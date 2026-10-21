"""Dynamic GRPO loop combining voting and discussion training with a joint loss.

Each iteration:
  1. HARVEST: the current policy plays full self-play games.  Both voting-phase
     and discussion-phase turns are captured from the same games in one pass.
  2. COMPLETIONS: for each harvested scenario the policy generates N completions
     (vote and defense scenarios share the same vLLM pass).
  3. VOTE REWARDS: rule-based vote rewards assigned with no GPU.
  4. DEFENSE EVALUATE + REWARD: judge model evaluates suspicion pre/post for
     each defense completion; team-aware reward is computed.
  5. TRAIN: joint GRPO training with combined loss
       L = lambda_vote * L_vote + lambda_defense * L_defense
     then merge for next iteration.

The env scoping fix: role metadata for defense scenarios is extracted immediately
after each game finishes (while env still belongs to that game), so every entry
carries the correct role/judge assignments for its own game.

Example:
    python scenarios/dynamic_joint_loop.py --base-model /models/policy \\
        --judge-model /models/judge --loop-count 2 --bf16
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path

import textarena as ta
import torch
from transformers import AutoTokenizer
from vllm import LLM

ROOT = Path(__file__).resolve().parents[1]
for subdir in ("src", "scripts"):
    module_dir = str(ROOT / subdir)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)

from grpo_training.cli import run_joint_training
from self_play_textarena import VLLMTextArenaAgent
from teamproject_fss2026.textarena_utils import extract_phase

from assign_defense_rewards import assign_team_aware_rewards
from assign_vote_rewards import assign_vote_rewards
from completion_generator import generate_completions
from defense_harvest import judge_context
from defense_judge import evaluate_all_suspicions, make_evaluator

HERE = Path(__file__).resolve().parent


def get_phase(env, observation: str) -> str:
    try:
        return str(env.state.game_state["phase"])
    except (AttributeError, KeyError, TypeError):
        pass
    try:
        return str(env.phase)
    except AttributeError:
        return extract_phase(observation)


def is_voting(phase_value: str) -> bool:
    return "vot" in str(phase_value).lower()


def is_discussion(phase_value: str) -> bool:
    return "discuss" in str(phase_value).lower()


def extract_roles(env) -> dict[int, dict[str, str]]:
    """Return {player_id: {"team": "Mafia"|"Village", "role": str}} from the env."""
    roles_raw: dict = {}
    try:
        roles_raw = env.state.game_state["roles"]
    except (AttributeError, KeyError, TypeError):
        pass
    if not roles_raw:
        try:
            roles_raw = env.roles
        except AttributeError:
            pass

    result: dict[int, dict[str, str]] = {}
    for player_id, role_obj in roles_raw.items():
        role_name = type(role_obj).__name__ if not isinstance(role_obj, str) else str(role_obj)
        team = "Mafia" if role_name == "Mafia" else "Village"
        result[int(player_id)] = {"team": team, "role": role_name}
    return result


def select_judge_ids(player_id: int, villager_ids: list[int], max_judges: int = 3) -> list[int]:
    eligible = [pid for pid in villager_ids if pid != player_id]
    if not eligible:
        return []
    return random.sample(eligible, min(max_judges, len(eligible)))


def harvest_and_generate_joint(
    model: str,
    stamp: str,
    raw_vote_dir: Path,
    raw_defense_dir: Path,
    vote_comp_dir: Path,
    defense_comp_dir: Path,
    args: argparse.Namespace,
) -> tuple[list[dict], list[dict], list[Path], list[Path]]:
    """Play full games, capture both vote and defense turns, generate completions.

    Returns (vote_entries, defense_entries, vote_files, defense_files).
    vote/defense_entries carry per-scenario metadata needed for later scoring.
    vote/defense_files are the paths to the generated JSONL completion files.

    Env scoping: extract_roles(env) is called inside the per-game loop block,
    immediately after each game ends, so annotations always use the correct game's
    role map — never the last game's env from a previous loop iteration.
    """
    llm = LLM(
        model=model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    tokenizer = AutoTokenizer.from_pretrained(model)
    agents = {pid: VLLMTextArenaAgent(llm, tokenizer) for pid in range(args.num_players)}

    vote_entries: list[dict] = []
    defense_entries: list[dict] = []
    vote_files: list[Path] = []
    defense_files: list[Path] = []
    vote_idx = 0
    defense_idx = 0

    try:
        for game_idx in range(args.games_per_iter):
            env = ta.make(env_id=args.env_id)
            env.reset(num_players=args.num_players)

            captured_vote: list[dict] = []
            captured_defense: list[dict] = []
            done = False
            turn_id = 0

            while not done:
                player_id, observation = env.get_observation()
                phase = get_phase(env, observation)
                if is_voting(phase):
                    captured_vote.append({
                        "player_id": player_id,
                        "turn_id": turn_id,
                        "observation": observation,
                    })
                elif is_discussion(phase):
                    captured_defense.append({
                        "player_id": player_id,
                        "turn_id": turn_id,
                        "observation": observation,
                    })
                agent_output = agents[player_id](observation)
                done, _ = env.step(action=agent_output["action"])
                turn_id += 1

            # Extract roles NOW — env still refers to this game, not the last game.
            role_map = extract_roles(env)
            if not role_map:
                print(f"[Harvest] game {game_idx + 1}: could not extract roles, dropping defense entries")
                captured_defense = []

            if role_map:
                all_player_ids = sorted(role_map.keys())
                villager_ids = [p for p in all_player_ids if role_map[p]["team"] == "Village"]
                mafia_ids = [p for p in all_player_ids if role_map[p]["team"] == "Mafia"]

                # Annotate all defense entries with this game's metadata before sampling.
                for entry in captured_defense:
                    pid = entry["player_id"]
                    entry["player_team"] = role_map.get(pid, {}).get("team", "Village")
                    entry["all_player_ids"] = all_player_ids
                    entry["villager_ids"] = villager_ids
                    entry["mafia_ids"] = mafia_ids
                    entry["judge_observation"] = judge_context(entry["observation"], pid)

            # Subsample and generate completions for vote turns.
            selected_vote = random.sample(
                captured_vote, min(len(captured_vote), args.situations_per_game)
            )
            for entry in selected_vote:
                stem = f"vote_{stamp}_g{game_idx}_p{entry['player_id']}_t{entry['turn_id']}"
                raw_path = raw_vote_dir / f"{stem}.txt"
                raw_path.write_text(entry["observation"], encoding="utf-8")
                out_file = vote_comp_dir / f"{stem}.jsonl"
                generate_completions(
                    scenario=str(raw_path),
                    model=model,
                    player_id=entry["player_id"],
                    num_completions=args.num_completions,
                    game_id=vote_idx,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    output=str(out_file),
                    llm=llm,
                    tokenizer=tokenizer,
                )
                entry["output"] = out_file
                vote_entries.append(entry)
                vote_files.append(out_file)
                vote_idx += 1

            # Subsample and generate completions for defense turns.
            selected_defense = random.sample(
                captured_defense, min(len(captured_defense), args.situations_per_game)
            )
            for entry in selected_defense:
                judge_ids = select_judge_ids(
                    entry["player_id"], entry.get("villager_ids", []), max_judges=3
                )
                entry["judge_ids"] = judge_ids
                stem = f"defense_{stamp}_g{game_idx}_p{entry['player_id']}_t{entry['turn_id']}"
                raw_path = raw_defense_dir / f"{stem}.txt"
                raw_path.write_text(entry["observation"], encoding="utf-8")
                out_file = defense_comp_dir / f"{stem}.jsonl"
                generate_completions(
                    scenario=str(raw_path),
                    model=model,
                    player_id=entry["player_id"],
                    num_completions=args.num_completions,
                    game_id=defense_idx,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    output=str(out_file),
                    llm=llm,
                    tokenizer=tokenizer,
                )
                entry["output"] = out_file
                defense_entries.append(entry)
                defense_files.append(out_file)
                defense_idx += 1

            print(
                f"[Harvest] game {game_idx + 1}/{args.games_per_iter}: "
                f"vote={len(selected_vote)}/{len(captured_vote)}, "
                f"defense={len(selected_defense)}/{len(captured_defense)}"
            )

    finally:
        del agents, llm, tokenizer
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    return vote_entries, defense_entries, vote_files, defense_files


def score_vote_completions(vote_files: list[Path]) -> None:
    """Assign rule-based vote rewards in-place. No GPU required."""
    for path in vote_files:
        assign_vote_rewards(str(path))


def score_defense_completions(
    defense_entries: list[dict], defense_files: list[Path], args: argparse.Namespace
) -> None:
    """Evaluate suspicion pre/post and assign team-aware rewards for defense completions."""
    evaluator = make_evaluator(args.judge_mode, args.judge_model, args.gpu_memory_utilization)
    try:
        for entry, output_path in zip(defense_entries, defense_files):
            judge_ids: list[int] = entry["judge_ids"]
            all_player_ids: list[int] = entry["all_player_ids"]

            suspicion_pre = evaluate_all_suspicions(
                observation=entry["observation"],
                public_game_state=entry["judge_observation"],
                response=None,
                judge_ids=judge_ids,
                all_player_ids=all_player_ids,
                evaluator=evaluator,
            )

            records = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            for record in records:
                record["player_team"] = entry["player_team"]
                record["judge_ids"] = judge_ids
                record["all_player_ids"] = all_player_ids
                record["villager_ids"] = entry["villager_ids"]
                record["mafia_ids"] = entry["mafia_ids"]
                record["judge_observation"] = entry["judge_observation"]
                record["suspicion_pre"] = suspicion_pre
                record["judge_mode"] = args.judge_mode
                record["judge_model"] = args.judge_model if args.judge_mode == "local" else ""
            output_path.write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
                encoding="utf-8",
            )

            assign_team_aware_rewards(records, output_path=output_path, evaluator=evaluator)

            pre_mean = sum(suspicion_pre.values()) / len(suspicion_pre) if suspicion_pre else 0.0
            print(
                f"[Judge] {output_path.stem}: "
                f"team={entry['player_team']}, "
                f"judges={judge_ids}, "
                f"suspicion_pre mean={pre_mean:.2f}"
            )
    finally:
        del evaluator
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def concat_jsonl(files: list[Path], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as target:
        for source in files:
            target.write(source.read_text(encoding="utf-8"))


def reward_summary(files: list[Path]) -> tuple[float, int, int]:
    rewards = [
        json.loads(line)["reward"]
        for source in files
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rewards:
        return 0.0, 0, 0
    return sum(rewards) / len(rewards), sum(r > 0 for r in rewards), len(rewards)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dynamic joint GRPO loop: trains on vote + defense turns from the same games"
    )
    parser.add_argument("--env-id", default="SecretMafia-v0")
    parser.add_argument("--num-players", type=int, default=8)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--judge-mode", choices=("local", "mock"), default="local")
    parser.add_argument("--judge-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--work-dir", type=Path, default=HERE / "runs" / "dynamic_joint_loop")
    parser.add_argument("--loop-count", type=int, default=4)
    parser.add_argument("--games-per-iter", type=int, default=5)
    parser.add_argument("--situations-per-game", type=int, default=4)
    parser.add_argument("--num-completions", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.4)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--lambda-vote", type=float, default=1.0,
                        help="Loss weight for voting objective.")
    parser.add_argument("--lambda-defense", type=float, default=1.0,
                        help="Loss weight for defense/discussion objective.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--max-completion-length", type=int, default=256)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    args = parser.parse_args()

    if min(args.loop_count, args.games_per_iter, args.situations_per_game) < 1:
        parser.error("loop, games, and situations must be >= 1")
    if args.num_completions < 2:
        parser.error("--num-completions must be >= 2 for GRPO")

    raw_vote_dir = args.work_dir / "raw_vote"
    raw_defense_dir = args.work_dir / "raw_defense"
    vote_comp_dir = args.work_dir / "completions_vote"
    defense_comp_dir = args.work_dir / "completions_defense"
    data_dir = args.work_dir / "data"
    ckpt_dir = args.work_dir / "checkpoints"
    for d in (raw_vote_dir, raw_defense_dir, vote_comp_dir, defense_comp_dir, data_dir, ckpt_dir):
        d.mkdir(parents=True, exist_ok=True)

    if args.judge_mode == "mock":
        print("[WARNING] Mock judge active: defense rewards reflect response length only.")

    policy_model = args.base_model
    trained_iterations = 0

    for iteration in range(1, args.loop_count + 1):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        print(f"\n{'='*70}\n[Iter {iteration}] using {policy_model}\n{'='*70}")

        vote_entries, defense_entries, vote_files, defense_files = harvest_and_generate_joint(
            model=policy_model,
            stamp=stamp,
            raw_vote_dir=raw_vote_dir,
            raw_defense_dir=raw_defense_dir,
            vote_comp_dir=vote_comp_dir,
            defense_comp_dir=defense_comp_dir,
            args=args,
        )

        if not vote_files and not defense_files:
            print("[WARNING] Nothing harvested this iteration. Stopping.")
            break

        if vote_files:
            score_vote_completions(vote_files)
            mean_r, pos, total = reward_summary(vote_files)
            print(f"[Vote] mean_reward={mean_r:.3f}, positive={pos}/{total}")

        if defense_files:
            score_defense_completions(defense_entries, defense_files, args)
            mean_r, pos, total = reward_summary(defense_files)
            print(f"[Defense] mean_reward={mean_r:.3f}, positive={pos}/{total}")

        vote_train = data_dir / f"vote_iter_{iteration}.jsonl"
        defense_train = data_dir / f"defense_iter_{iteration}.jsonl"
        if vote_files:
            concat_jsonl(vote_files, vote_train)
        if defense_files:
            concat_jsonl(defense_files, defense_train)

        adapter_out = ckpt_dir / f"iter_{iteration}" / "lora_adapter"
        merged_out = ckpt_dir / f"iter_{iteration}" / "merged_model"
        train_args = argparse.Namespace(
            model=policy_model,
            old_policy_model=policy_model,
            vote_data=str(vote_train) if vote_files else "",
            defense_data=str(defense_train) if defense_files else "",
            lambda_vote=args.lambda_vote,
            lambda_defense=args.lambda_defense,
            output_dir=str(adapter_out),
            epochs=args.epochs,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            max_prompt_length=args.max_prompt_length,
            max_completion_length=args.max_completion_length,
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            bf16=args.bf16,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            merge_for_vllm=True,
            merged_output_dir=str(merged_out),
        )
        run_joint_training(train_args)
        gc.collect()
        torch.cuda.empty_cache()

        prev_policy_model = policy_model
        policy_model = str(merged_out)
        trained_iterations += 1

        if iteration > 1 and Path(prev_policy_model).is_dir():
            shutil.rmtree(prev_policy_model, ignore_errors=True)
            print(f"[Cleanup] removed previous merged model: {prev_policy_model}")

    if trained_iterations:
        print(f"\n{'='*70}\n[Final] measuring trained model {policy_model}\n{'='*70}")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        vote_entries, defense_entries, vote_files, defense_files = harvest_and_generate_joint(
            model=policy_model,
            stamp=stamp,
            raw_vote_dir=raw_vote_dir,
            raw_defense_dir=raw_defense_dir,
            vote_comp_dir=vote_comp_dir,
            defense_comp_dir=defense_comp_dir,
            args=args,
        )
        if vote_files:
            score_vote_completions(vote_files)
            mean_r, pos, total = reward_summary(vote_files)
            print(f"[Final/Vote] mean_reward={mean_r:.3f}, positive={pos}/{total}")
        if defense_files:
            score_defense_completions(defense_entries, defense_files, args)
            mean_r, pos, total = reward_summary(defense_files)
            print(f"[Final/Defense] mean_reward={mean_r:.3f}, positive={pos}/{total}")

    print(f"\n[Done] final merged model: {policy_model}")


if __name__ == "__main__":
    main()
