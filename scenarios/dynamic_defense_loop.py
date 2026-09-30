"""Dynamic GRPO loop for Secret Mafia discussion turns with team-aware rewards.

Each iteration:
  1. HARVEST: the current policy plays full self-play games.  All discussion-phase
     turns are captured in memory, then randomly sampled (mirrors dynamic_vote_loop.py).
  2. EVALUATE PRE: three randomly chosen villager judges each rate the suspicion of
     all other players BEFORE the speaking player's completion.  One LLM call per
     judge (3 total per scenario).
  3. COMPLETIONS: for each harvested scenario the policy generates N completions.
  4. EVALUATE POST + REWARD: the same three judges re-rate all players after each
     completion.  The team-aware reward is computed from the changes in suspicion:
       - Mafia speaker:  Σ villager_suspicion_up + Σ mafia_suspicion_down
       - Village speaker: Σ villager_suspicion_down + Σ mafia_suspicion_up
  5. TRAIN: GRPO-train on this iteration's data only, then merge for next iteration.

Example:
    python scenarios/dynamic_defense_loop.py --base-model /models/policy \\
        --judge-model /models/judge --loop-count 2 --bf16
"""

from __future__ import annotations

import argparse
import gc
import json
import random
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

from grpo_training.cli import run_training
from self_play_textarena import VLLMTextArenaAgent
from teamproject_fss2026.textarena_utils import extract_phase

from assign_defense_rewards import assign_team_aware_rewards
from completion_generator import generate_completions
from defense_harvest import judge_context
from defense_judge import evaluate_all_suspicions, make_evaluator

HERE = Path(__file__).resolve().parent


def get_phase(env, observation: str) -> str:
    """Use the environment phase when available, otherwise parse the observation."""
    try:
        return str(env.state.game_state["phase"])
    except (AttributeError, KeyError, TypeError):
        pass
    try:
        return str(env.phase)
    except AttributeError:
        return extract_phase(observation)


def extract_roles(env) -> dict[int, dict[str, str]]:
    """Return {player_id: {"team": "Mafia"|"Village", "role": str}} from the env.

    Tries ``env.state.game_state["roles"]`` first (same pattern used in
    run_eval_games.py), then falls back to ``env.roles``.  Role class name
    "Mafia" maps to team "Mafia"; everything else maps to "Village".
    """
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
    """Pick other villagers as judges; the acting player remains in the scored set."""
    eligible = [pid for pid in villager_ids if pid != player_id]
    if not eligible:
        return []
    return random.sample(eligible, min(max_judges, len(eligible)))


def harvest_and_generate(
    model: str, tag: str, raw_dir: Path, completions_dir: Path, args: argparse.Namespace
) -> list[dict]:
    """Play full games, capture discussion turns in memory, generate completions.

    All discussion-phase turns are collected during each game, then randomly
    sampled after the game ends (same pattern as dynamic_vote_loop.py).  Role
    assignments are read from the environment after the game and used to:
      - annotate each scenario with player_team, villager_ids, mafia_ids
      - randomly select 3 villager judges for that scenario
    """
    llm = LLM(
        model=model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    tokenizer = AutoTokenizer.from_pretrained(model)
    agents = {pid: VLLMTextArenaAgent(llm, tokenizer) for pid in range(args.num_players)}
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    harvested: list[dict] = []
    try:
        for game_idx in range(args.games_per_iter):
            env = ta.make(env_id=args.env_id)
            env.reset(num_players=args.num_players)
            captured: list[dict] = []
            done = False
            turn_id = 0
            while not done:
                player_id, observation = env.get_observation()
                if "discuss" in get_phase(env, observation).lower():
                    captured.append({
                        "player_id": player_id,
                        "turn_id": turn_id,
                        "observation": observation,
                    })
                agent_output = agents[player_id](observation)
                done, _ = env.step(action=agent_output["action"])
                turn_id += 1

            # Extract role assignments now that the game has finished
            role_map = extract_roles(env)
            if not role_map:
                print(f"[Harvest] game {game_idx + 1}: could not extract roles, skipping")
                continue

            all_player_ids = sorted(role_map.keys())
            villager_ids = [p for p in all_player_ids if role_map[p]["team"] == "Village"]
            mafia_ids = [p for p in all_player_ids if role_map[p]["team"] == "Mafia"]

            # Annotate captured turns with team and role metadata
            for entry in captured:
                pid = entry["player_id"]
                team = role_map.get(pid, {}).get("team", "Village")
                entry["player_team"] = team
                entry["all_player_ids"] = all_player_ids
                entry["villager_ids"] = villager_ids
                entry["mafia_ids"] = mafia_ids
                entry["judge_observation"] = judge_context(entry["observation"], pid)

            # Randomly sample up to situations_per_game turns per game
            selected = random.sample(captured, min(len(captured), args.situations_per_game))

            for entry in selected:
                # Never let the speaking player act as a judge. Choose only other villagers.
                judge_ids = select_judge_ids(entry["player_id"], villager_ids, max_judges=3)
                entry["judge_ids"] = judge_ids

                stem = f"{tag}_{stamp}_g{game_idx}_p{entry['player_id']}_t{entry['turn_id']}"
                raw_path = raw_dir / f"{stem}.txt"
                raw_path.write_text(entry["observation"], encoding="utf-8")
                output = completions_dir / f"{stem}.jsonl"

                generate_completions(
                    scenario=str(raw_path), model=model, player_id=entry["player_id"],
                    num_completions=args.num_completions, game_id=len(harvested),
                    temperature=args.temperature, max_tokens=args.max_tokens,
                    output=str(output), llm=llm, tokenizer=tokenizer,
                )
                entry["stem"] = stem
                entry["output"] = output
                harvested.append(entry)

            print(
                f"[Harvest] game {game_idx + 1}/{args.games_per_iter}: "
                f"{len(captured)} discussion turns, kept {len(selected)}"
            )
    finally:
        del agents, llm, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    return harvested


def rate_and_save(
    harvested: list[dict], scenario_dir: Path, args: argparse.Namespace
) -> list[Path]:
    """Pre-rate all players, then score completions with team-aware rewards."""
    evaluator = make_evaluator(args.judge_mode, args.judge_model, args.gpu_memory_utilization)
    files: list[Path] = []
    try:
        for entry in harvested:
            judge_ids: list[int] = entry["judge_ids"]
            all_player_ids: list[int] = entry["all_player_ids"]

            # Pre-evaluation: 3 judge calls, each rates all other players
            suspicion_pre = evaluate_all_suspicions(
                observation=entry["observation"],
                public_game_state=entry["judge_observation"],
                response=None,
                judge_ids=judge_ids,
                all_player_ids=all_player_ids,
                evaluator=evaluator,
            )

            scenario_data = {
                "player_id": entry["player_id"],
                "player_team": entry["player_team"],
                "turn_id": entry["turn_id"],
                "observation": entry["observation"],
                "judge_observation": entry["judge_observation"],
                "judge_ids": judge_ids,
                "all_player_ids": all_player_ids,
                "villager_ids": entry["villager_ids"],
                "mafia_ids": entry["mafia_ids"],
                "suspicion_pre": suspicion_pre,
                "judge_mode": args.judge_mode,
                "judge_model": args.judge_model if args.judge_mode == "local" else "",
            }
            scenario_path = scenario_dir / f"{entry['stem']}.json"
            scenario_path.write_text(
                json.dumps(scenario_data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )

            # Attach suspicion_pre and all scenario metadata to every completion record
            output = entry["output"]
            records = [
                json.loads(line)
                for line in output.read_text(encoding="utf-8").splitlines()
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
                record["judge_mode"] = scenario_data["judge_mode"]
                record["judge_model"] = scenario_data["judge_model"]
            output.write_text(
                "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
                encoding="utf-8",
            )

            # Post-evaluation + team-aware reward (3 × num_completions judge calls)
            assign_team_aware_rewards(records, output_path=output, evaluator=evaluator)
            files.append(output)

            pre_mean = sum(suspicion_pre.values()) / len(suspicion_pre) if suspicion_pre else 0.0
            print(
                f"[Judge] {entry['stem']}: "
                f"team={entry['player_team']}, "
                f"judges={judge_ids}, "
                f"suspicion_pre mean={pre_mean:.2f}"
            )
    finally:
        del evaluator
        gc.collect()
        torch.cuda.empty_cache()
    return files


def harvest_complete_and_rate(
    model: str, tag: str, raw_dir: Path, scenario_dir: Path,
    completions_dir: Path, args: argparse.Namespace,
) -> list[Path]:
    harvested = harvest_and_generate(model, tag, raw_dir, completions_dir, args)
    if not harvested:
        return []
    return rate_and_save(harvested, scenario_dir, args)


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
        raise ValueError("No scored completions")
    return sum(rewards) / len(rewards), sum(reward > 0 for reward in rewards), len(rewards)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dynamic GRPO loop for accused Secret Mafia players")
    parser.add_argument("--env-id", default="SecretMafia-v0")
    parser.add_argument("--num-players", type=int, default=8)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--judge-mode", choices=("local", "mock"), default="local")
    parser.add_argument("--judge-model", default="Qwen/Qwen2.5-7B-Instruct", help="Local judge model path or HF id")
    parser.add_argument("--work-dir", type=Path, default=HERE / "runs" / "dynamic_defense_loop")
    parser.add_argument("--completions-dir", type=Path, default=HERE / "completions_defense_dynamic")
    parser.add_argument("--loop-count", type=int, default=4)
    parser.add_argument("--games-per-iter", type=int, default=5)
    parser.add_argument("--situations-per-game", type=int, default=4)
    # --accusation-window removed: any discussion turn is now a valid scenario
    parser.add_argument("--num-completions", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
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

    raw_dir = args.work_dir / "raw_observations"
    scenario_dir = args.work_dir / "observations"
    data_dir = args.work_dir / "data"
    checkpoints = args.work_dir / "checkpoints"
    for directory in (raw_dir, scenario_dir, data_dir, checkpoints, args.completions_dir):
        directory.mkdir(parents=True, exist_ok=True)

    if args.judge_mode == "mock":
        print("[WARNING] Mock judge active: scores reflect response length only.")

    policy_model = args.base_model
    trained_iterations = 0
    for iteration in range(1, args.loop_count + 1):
        print(f"[Iter {iteration}] self-play and defense generation with {policy_model}")
        files = harvest_complete_and_rate(
            policy_model, f"iter_{iteration}", raw_dir, scenario_dir, args.completions_dir, args
        )
        if not files:
            print("[WARNING] No discussion turns harvested; stopping training.")
            break
        mean_reward, improved, total = reward_summary(files)
        print(f"[Iter {iteration}] mean reward={mean_reward:.3f}; positive-reward completions: {improved}/{total}")

        train_file = data_dir / f"train_iter_{iteration}.jsonl"
        concat_jsonl(files, train_file)
        adapter_out = checkpoints / f"iter_{iteration}" / "lora_adapter"
        merged_out = checkpoints / f"iter_{iteration}" / "merged_model"
        train_args = argparse.Namespace(
            model=policy_model, old_policy_model=policy_model, data=str(train_file),
            output_dir=str(adapter_out), epochs=args.epochs, batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate, max_prompt_length=args.max_prompt_length,
            max_completion_length=args.max_completion_length, logging_steps=args.logging_steps,
            save_steps=args.save_steps, bf16=args.bf16, lora_r=args.lora_r,
            lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            merge_for_vllm=True, merged_output_dir=str(merged_out),
        )
        run_training(train_args)
        gc.collect()
        torch.cuda.empty_cache()
        policy_model = str(merged_out)
        trained_iterations += 1

    if trained_iterations:
        print(f"[Final] measuring merged policy {policy_model} on freshly harvested situations")
        final_files = harvest_complete_and_rate(
            policy_model, "final", raw_dir, scenario_dir, args.completions_dir, args
        )
        if final_files:
            mean_reward, improved, total = reward_summary(final_files)
            print(f"[Final] mean reward={mean_reward:.3f}; positive-reward completions: {improved}/{total}")
        else:
            print("[WARNING] No final discussion turns found; no final rating available.")
        print(f"[Final] merged model: {policy_model}")
        print("[Note] Each pass uses new situations; reward averages are not a fixed-set comparison.")


if __name__ == "__main__":
    main()
