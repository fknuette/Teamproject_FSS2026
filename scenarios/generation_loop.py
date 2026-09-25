"""
Evaluate a given model on the fixed voting scenarios: generate completions,
auto-assign the self-vote rewards, and report the bad-vote rate.

Bad vote = negative reward = self-vote OR unparseable vote.
Lower is better.

Usage:
    python scenarios/evaluate_vote_model.py --model Qwen/Qwen2.5-7B-Instruct
    python scenarios/evaluate_vote_model.py --model /path/to/merged_model
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# make sibling modules + src importable regardless of where we're called from
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[0].parents[0]  # project root (scenarios/ -> Teamproject_FSS2026)
for _p in (str(HERE), str(ROOT / "src"), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from vllm import LLM
from transformers import AutoTokenizer
from completion_generator import generate_completions
from assign_vote_rewards import assign_vote_rewards


def negative_reward_rate(files: list[Path]) -> tuple[int, int]:
    """Count completions with negative reward (self-vote or invalid)."""
    n_neg = n_total = 0
    for f in files:
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            n_total += 1
            if json.loads(line).get("reward", 0.0) < 0:
                n_neg += 1
    return n_neg, n_total


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate a model's self-vote rate on the fixed scenarios")
    p.add_argument("--model", required=True, help="Model path or HF name to evaluate.")
    p.add_argument("--obs-dir", default=str(HERE / "observations_vote"),
                   help="Folder with the fixed .txt vote scenarios.")
    p.add_argument("--out-dir", default=str(HERE / "completions_vote_eval"),
                   help="Where the eval completions are written (separate from training dirs).")
    p.add_argument("--num-completions", type=int, default=16,
                   help="Completions per scenario. Higher = more stable rate. Default 16.")
    p.add_argument("--temperature", type=float, default=0.7,
                   help="Sampling temperature for evaluation. Default 0.7 (not 1.0) for a less "
                        "noisy estimate of typical behavior.")
    p.add_argument("--max-tokens", type=int, default=100, help="Votes are short.")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    args = p.parse_args()

    obs_dir = Path(args.obs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scenarios = sorted(obs_dir.glob("*.txt"))
    if not scenarios:
        raise SystemExit(f"No .txt scenarios found in {obs_dir}")

    print(f"Evaluating {args.model} on {len(scenarios)} scenarios "
          f"({args.num_completions} completions each, temp={args.temperature})")

    # load the model once, reuse for every scenario
    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_memory_utilization)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    per_scenario: list[tuple[str, int, int]] = []
    all_files: list[Path] = []

    for game_id, scenario in enumerate(scenarios):
        out_file = out_dir / f"{scenario.stem}.jsonl"
        generate_completions(
            scenario=str(scenario),
            model=args.model,
            num_completions=args.num_completions,
            game_id=game_id,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            output=str(out_file),
            llm=llm,
            tokenizer=tokenizer,
        )
        assign_vote_rewards(str(out_file))
        all_files.append(out_file)

        # per-scenario rate
        n_neg, n_total = negative_reward_rate([out_file])
        per_scenario.append((scenario.stem, n_neg, n_total))

    # --- report ---
    print(f"\n{'='*70}\nBad-vote rate per scenario (self-vote or invalid)\n{'='*70}")
    for name, n_neg, n_total in per_scenario:
        r = n_neg / n_total if n_total else 0.0
        print(f"  {name:<40} {n_neg:3d}/{n_total:3d} = {r:.1%}")

    total_neg, total_all = negative_reward_rate(all_files)
    overall = total_neg / total_all if total_all else 0.0
    print("=" * 70)
    print(f"  OVERALL: {total_neg}/{total_all} = {overall:.1%}")
    print("=" * 70)


if __name__ == "__main__":
    main()
