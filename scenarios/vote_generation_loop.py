"""
Online GRPO loop over the fixed VOTING scenarios.

Lives in scenarios/ next to completion_generator.py and assign_vote_rewards.py.
The "rollout" is generating completions for every .txt in observations_vote/,
then auto-assigning the self-vote rewards. Then GRPO-train, merge, repeat --
the merged model becomes the generator for the next iteration (= online).

    observations_vote/*.txt
        -> completions_vote/*.jsonl  (completions + auto rewards)
        -> concat -> train (run_training) -> merge
        -> next iteration generates with the improved model
"""

from __future__ import annotations

import argparse
import gc
import shutil
import sys
from pathlib import Path

import torch
from vllm import LLM
from transformers import AutoTokenizer

# scenarios/ is one level under the project root.
# grpo_training lives under scripts/, textarena_utils under src/ -> add BOTH.
ROOT = Path(__file__).resolve().parents[1]
for _sub in ("src", "scripts"):
    _path = str(ROOT / _sub)
    if _path not in sys.path:
        sys.path.insert(0, _path)

from grpo_training.cli import run_training
from completion_generator import generate_completions
from assign_vote_rewards import assign_vote_rewards

# All paths are relative to this script's folder (scenarios/), so the loop can be
# run from anywhere.
HERE = Path(__file__).resolve().parent


def concat_jsonl(files: list[Path], output_file: Path) -> None:
    """Merge per-scenario JSONL files into one training file for this iteration."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as out:
        for f in files:
            out.write(f.read_text(encoding="utf-8"))


def negative_reward_rate(files: list[Path]) -> tuple[int, int]:
    """
    Count completions with a negative reward across the given JSONL files.

    assign_vote_rewards already scored each completion, so we just read the
    reward field: negative (-1) means self-vote OR unparseable vote, positive
    (+1) means a valid vote for someone else. Returns (num_negative, num_total).
    """
    import json

    n_neg = 0
    n_total = 0
    for f in files:
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            n_total += 1
            if rec.get("reward", 0.0) < 0:
                n_neg += 1
    return n_neg, n_total


def run_completion_pass(
    model: str,
    scenarios: list[Path],
    completions_dir: Path,
    tag: str,
    args,
) -> list[Path]:
    """
    Generate completions for every scenario with `model`, assign vote rewards,
    and return the written JSONL paths. `tag` is used in the file names (e.g.
    "iter_1" or "final"). Loads the model once and frees it afterwards.
    """
    llm = LLM(model=model, gpu_memory_utilization=args.gpu_memory_utilization)
    tokenizer = AutoTokenizer.from_pretrained(model)

    files: list[Path] = []
    for game_id, scenario in enumerate(scenarios):
        out_file = completions_dir / f"{tag}_{scenario.stem}.jsonl"
        generate_completions(
            scenario=str(scenario),
            model=model,
            player_id=0,
            num_completions=args.num_completions,
            game_id=game_id,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            output=str(out_file),
            llm=llm,
            tokenizer=tokenizer,
        )
        assign_vote_rewards(str(out_file))
        files.append(out_file)

    del llm, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return files


def main() -> None:
    p = argparse.ArgumentParser(description="Online GRPO loop over fixed voting scenarios")
    p.add_argument("--obs-dir", type=str, default=str(HERE / "observations_vote"),
                   help="Folder with the fixed .txt vote observations.")
    p.add_argument("--completions-dir", type=str, default=str(HERE / "completions_vote"),
                   help="Where the per-scenario completion JSONLs are written.")
    p.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--work-dir", type=str, default=str(HERE / "runs" / "vote_loop"))
    p.add_argument("--loop-count", type=int, default=4)
    p.add_argument("--num-completions", type=int, default=8)
    p.add_argument("--temperature", type=float, default=1.0,
                   help="High by default: short vote answers need variance for GRPO signal.")
    p.add_argument("--max-tokens", type=int, default=100, help="Votes are short.")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    # training hyperparams passed straight through to run_training
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--gradient-accumulation-steps", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--max-prompt-length", type=int, default=1024)
    p.add_argument("--max-completion-length", type=int, default=256)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--save-steps", type=int, default=100)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    args = p.parse_args()

    work_dir = Path(args.work_dir)
    data_dir = work_dir / "data"
    ckpt_dir = work_dir / "checkpoints"
    data_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    obs_dir = Path(args.obs_dir)
    completions_dir = Path(args.completions_dir)
    completions_dir.mkdir(parents=True, exist_ok=True)

    scenarios = sorted(obs_dir.glob("*.txt"))
    if not scenarios:
        raise SystemExit(f"No .txt scenarios found in {obs_dir}")
    print(f"Found {len(scenarios)} vote scenarios in {obs_dir}")

    policy_model = args.base_model
    rate_history: list[tuple[str, int, int]] = []   # (label, n_self, n_total) per iteration

    for iter_idx in range(1, args.loop_count + 1):
        print(f"\n{'='*70}\n[Iter {iter_idx}] generating with {policy_model}\n{'='*70}")

        # --- 1. ROLLOUT: completions for every scenario (model loaded once) ---
        iter_files = run_completion_pass(
            model=policy_model,
            scenarios=scenarios,
            completions_dir=completions_dir,
            tag=f"iter_{iter_idx}",
            args=args,
        )

        # negative-reward rate for THIS iteration (before training), on the model
        # that generated the data -> shows the state going INTO this training step.
        n_neg, n_total = negative_reward_rate(iter_files)
        rate = n_neg / n_total if n_total else 0.0
        rate_history.append((f"iter_{iter_idx}", n_neg, n_total))
        print(f"[Bad-vote] iter {iter_idx}: {n_neg}/{n_total} = {rate:.1%}")

        # --- 2. one training file for THIS iteration (not cumulative) ---
        train_file = data_dir / f"train_iter_{iter_idx}.jsonl"
        concat_jsonl(iter_files, train_file)

        # --- 3. TRAIN (reuses your existing run_training) ---
        adapter_out = ckpt_dir / f"iter_{iter_idx}" / "lora_adapter"
        merged_out = ckpt_dir / f"iter_{iter_idx}" / "merged_model"
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
            merge_for_vllm=True,                 # merge inside run_training
            merged_output_dir=str(merged_out),
        )
        print(f"[Iter {iter_idx}] training on {train_file}")
        run_training(train_args)
        gc.collect()
        torch.cuda.empty_cache()

        # --- 4. next iteration generates with the improved model (= online) ---
        prev_policy_model = policy_model        # the model this iteration used
        policy_model = str(merged_out)          # point to the fresh merged model

        # Rolling cleanup: once we have the new merged model, the PREVIOUS one is
        # no longer needed for the loop (adapters stay as the real history).
        # Guards:
        #   - iter_idx > 1: never touch the base model (iter 1's prev is base).
        #   - is_dir(): a HF hub name ("Qwen/...") is not a dir -> never deleted;
        #     only a real local merged_model folder is removed.
        # NOTE: this runs AFTER run_training has fully finished; run_training loads
        # prev_policy_model three times internally, so it must NOT be deleted earlier.
        if iter_idx > 1 and Path(prev_policy_model).is_dir():
            shutil.rmtree(prev_policy_model, ignore_errors=True)
            print(f"[Cleanup] removed previous merged model: {prev_policy_model}")

    # --- FINAL PASS: the last iteration trained + merged a model, but no rollout
    # has measured it yet. Generate one more pass with the FINAL model so we see
    # the self-vote rate AFTER the last training step. policy_model currently
    # points at the last merged model.
    print(f"\n{'='*70}\n[Final] generating with trained model {policy_model}\n{'='*70}")
    final_files = run_completion_pass(
        model=policy_model,
        scenarios=scenarios,
        completions_dir=completions_dir,
        tag="final",
        args=args,
    )
    n_neg, n_total = negative_reward_rate(final_files)
    final_rate = n_neg / n_total if n_total else 0.0
    rate_history.append(("final", n_neg, n_total))
    print(f"[Bad-vote] final: {n_neg}/{n_total} = {final_rate:.1%}")

    # --- summary of the whole trajectory ---
    print(f"\n{'='*70}\nNegative-vote rate over the run (self-vote or invalid)\n{'='*70}")
    for label, ns, nt in rate_history:
        r = ns / nt if nt else 0.0
        print(f"  {label:>8}: {ns:3d}/{nt:3d} = {r:.1%}")
    print("=" * 70)

    print("\nVote online loop finished.")


if __name__ == "__main__":
    main()