"""Dynamic GRPO loop for defense speeches in real Secret Mafia discussions.

The current policy plays full self-play games. Recent accusations against the
active player are harvested as fresh one-turn scenarios. For every scenario the
policy generates multiple defense speeches; three local villager judges score
the suspicion before and after each speech. Training uses this iteration's
records only, then the merged policy starts the next iteration.

Example:
    python scenarios/dynamic_defense_loop.py --base-model /models/policy \
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

from assign_defense_rewards import assign_defense_rewards
from completion_generator import generate_completions
from defense_harvest import is_defense_situation, judge_context
from defense_judge import evaluate_suspicion, make_evaluator

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


def harvest_and_generate(
    model: str, tag: str, raw_dir: Path, completions_dir: Path, args: argparse.Namespace
) -> list[dict]:
    """Play full games and generate completions with one loaded policy model."""
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
            candidates: list[dict] = []
            done = False
            turn_id = 0
            while not done:
                player_id, observation = env.get_observation()
                if "discuss" in get_phase(env, observation).lower() and is_defense_situation(
                    observation, player_id, args.accusation_window
                ):
                    candidates.append({
                        "player_id": player_id,
                        "turn_id": turn_id,
                        "observation": observation,
                        "judge_observation": judge_context(observation, player_id),
                    })
                agent_output = agents[player_id](observation)
                done, _ = env.step(action=agent_output["action"])
                turn_id += 1

            selected = random.sample(candidates, min(len(candidates), args.situations_per_game))
            for entry in selected:
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
                f"{len(candidates)} accused discussion turns, kept {len(selected)}"
            )
    finally:
        del agents, llm, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    return harvested


def rate_and_save(
    harvested: list[dict], scenario_dir: Path, args: argparse.Namespace
) -> list[Path]:
    """Cache pre-ratings and score completions with one loaded judge model."""
    evaluator = make_evaluator(args.judge_mode, args.judge_model, args.gpu_memory_utilization)
    files: list[Path] = []
    try:
        for entry in harvested:
            public_context = entry["judge_observation"]
            baseline = evaluate_suspicion(public_context, None, evaluator)
            scenario_data = {
                "player_id": entry["player_id"],
                "turn_id": entry["turn_id"],
                "observation": entry["observation"],
                "judge_observation": public_context,
                "verdacht_pre": baseline,
                "judge_mode": args.judge_mode,
                "judge_model": args.judge_model if args.judge_mode == "local" else "",
            }
            scenario_path = scenario_dir / f"{entry['stem']}.json"
            scenario_path.write_text(
                json.dumps(scenario_data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )

            output = entry["output"]
            records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line.strip()]
            for record in records:
                record.update({
                    "verdacht_pre": baseline,
                    "judge_observation": public_context,
                    "judge_mode": scenario_data["judge_mode"],
                    "judge_model": scenario_data["judge_model"],
                })
            output.write_text(
                "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
                encoding="utf-8",
            )
            assign_defense_rewards(str(output), evaluator=evaluator)
            files.append(output)
            print(f"[Judge] {entry['stem']}: verdacht_pre={baseline:.2f}")
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
    parser.add_argument("--judge-model", default="", help="Fixed local judge model, required in local mode")
    parser.add_argument("--work-dir", type=Path, default=HERE / "runs" / "dynamic_defense_loop")
    parser.add_argument("--completions-dir", type=Path, default=HERE / "completions_defense_dynamic")
    parser.add_argument("--loop-count", type=int, default=4)
    parser.add_argument("--games-per-iter", type=int, default=5)
    parser.add_argument("--situations-per-game", type=int, default=4)
    parser.add_argument("--accusation-window", type=int, default=8,
                        help="Number of recent public messages searched for an accusation")
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

    if args.judge_mode == "local" and not args.judge_model:
        parser.error("--judge-model is required for --judge-mode local")
    if min(args.loop_count, args.games_per_iter, args.situations_per_game, args.accusation_window) < 1:
        parser.error("loop, games, situations, and accusation window must be >= 1")
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
            print("[WARNING] No accused discussion turns found; stopping training.")
            break
        mean_reward, improved, total = reward_summary(files)
        print(f"[Iter {iteration}] mean reward={mean_reward:.3f}; suspicion lowered: {improved}/{total}")

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
            print(f"[Final] mean reward={mean_reward:.3f}; suspicion lowered: {improved}/{total}")
        else:
            print("[WARNING] No final defense situations found; no final rating available.")
        print(f"[Final] merged model: {policy_model}")
        print("[Note] Each pass uses new situations; reward averages are not a fixed-set comparison.")


if __name__ == "__main__":
    main()
