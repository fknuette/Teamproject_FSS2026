"""Online GRPO loop for fixed defense speeches under suspicion.

Create observations first with generate_suspicion_observations.py, using the
same judge mode and model as this loop. Each iteration generates fresh speeches,
rates them with three villagers, trains through the repository's run_training,
and uses the merged policy for the next iteration.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import torch
from vllm import LLM
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
for subdir in ("src", "scripts"):
    module_dir = str(ROOT / subdir)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)

# Exactly the repository's training wrapper used by vote_generation_loop.py.
from grpo_training.cli import run_training
from completion_generator import generate_completions
from assign_defense_rewards import assign_defense_rewards
from defense_judge import make_evaluator

HERE = Path(__file__).resolve().parent


def check_scenarios(scenarios: list[Path], judge_mode: str, judge_model: str) -> None:
    for scenario in scenarios:
        data = json.loads(scenario.read_text(encoding="utf-8"))
        if not isinstance(data.get("observation"), str) or not data["observation"].strip():
            raise ValueError(f"Missing observation in {scenario}")
        baseline = data.get("verdacht_pre")
        if isinstance(baseline, bool) or not isinstance(baseline, (int, float)) or not 1 <= baseline <= 10:
            raise ValueError(f"Missing/invalid verdacht_pre in {scenario}")
        if not isinstance(data.get("player_id"), int) or isinstance(data["player_id"], bool):
            raise ValueError(f"Missing/invalid player_id in {scenario}")
        if data.get("judge_mode") != judge_mode or data.get("judge_model", "") != (
            judge_model if judge_mode == "local" else ""
        ):
            raise ValueError(
                f"Judge mismatch for {scenario}; regenerate baselines with "
                "generate_suspicion_observations.py using the same judge settings"
            )


def generate_and_rate(
    model: str, scenarios: list[Path], output_dir: Path, tag: str, args: argparse.Namespace
) -> list[Path]:
    """Generate with policy, unload it, then rate with a separate local judge."""
    llm = LLM(model=model, gpu_memory_utilization=args.gpu_memory_utilization)
    tokenizer = AutoTokenizer.from_pretrained(model)
    files = []
    try:
        for game_id, scenario in enumerate(scenarios):
            data = json.loads(scenario.read_text(encoding="utf-8"))
            output = output_dir / f"{tag}_{scenario.stem}.jsonl"
            generate_completions(
                scenario=str(scenario),
                model=model,
                player_id=data["player_id"],
                num_completions=args.num_completions,
                game_id=game_id,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                output=str(output),
                llm=llm,
                tokenizer=tokenizer,
            )
            files.append(output)
    finally:
        del llm, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    # In local mode, the judge is loaded only after vLLM policy inference ends.
    evaluator = make_evaluator(args.judge_mode, args.judge_model, args.gpu_memory_utilization)
    try:
        for output in files:
            assign_defense_rewards(str(output), evaluator=evaluator)
    finally:
        del evaluator
        gc.collect()
        torch.cuda.empty_cache()
    return files


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
    parser = argparse.ArgumentParser(description="Online GRPO training for fixed defense scenarios")
    parser.add_argument("--obs-dir", type=Path, default=HERE / "observations_defense")
    parser.add_argument("--completions-dir", type=Path, default=HERE / "completions_defense")
    parser.add_argument("--work-dir", type=Path, default=HERE / "runs" / "defense_loop")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--judge-mode", choices=("mock", "local"), default="mock")
    parser.add_argument("--judge-model", default="", help="Local vLLM evaluator model, fixed across iterations")
    parser.add_argument("--loop-count", type=int, default=4)
    parser.add_argument("--num-completions", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.6)
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

    if args.loop_count < 1 or args.num_completions < 2:
        parser.error("--loop-count must be >= 1 and --num-completions must be >= 2")
    if args.judge_mode == "local" and not args.judge_model:
        parser.error("--judge-model is required for --judge-mode local")

    scenarios = sorted(args.obs_dir.glob("*.json"))
    if not 1 <= len(scenarios) <= 3:
        raise SystemExit("Expected 1-3 .json scenarios; run generate_suspicion_observations.py first")
    check_scenarios(scenarios, args.judge_mode, args.judge_model)
    args.completions_dir.mkdir(parents=True, exist_ok=True)
    data_dir = args.work_dir / "data"
    checkpoints = args.work_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)

    if args.judge_mode == "mock":
        print("[WARNING] Mock judge active: scores are placeholders, not LLM assessments.")

    policy_model = args.base_model
    for iteration in range(1, args.loop_count + 1):
        print(f"[Iter {iteration}] generating defenses with {policy_model}")
        files = generate_and_rate(policy_model, scenarios, args.completions_dir, f"iter_{iteration}", args)
        mean_reward, improved, total = reward_summary(files)
        print(f"[Iter {iteration}] mean reward={mean_reward:.3f}; suspicion lowered: {improved}/{total}")

        train_file = data_dir / f"train_iter_{iteration}.jsonl"
        concat_jsonl(files, train_file)
        adapter_out = checkpoints / f"iter_{iteration}" / "lora_adapter"
        merged_out = checkpoints / f"iter_{iteration}" / "merged_model"
        train_args = argparse.Namespace(
            model=policy_model,
            old_policy_model=policy_model,
            data=str(train_file),
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
        run_training(train_args)
        gc.collect()
        torch.cuda.empty_cache()
        policy_model = str(merged_out)

    final_files = generate_and_rate(policy_model, scenarios, args.completions_dir, "final", args)
    mean_reward, improved, total = reward_summary(final_files)
    print(f"[Final] mean reward={mean_reward:.3f}; suspicion lowered: {improved}/{total}")
    print(f"[Final] merged model: {policy_model}")


if __name__ == "__main__":
    main()
