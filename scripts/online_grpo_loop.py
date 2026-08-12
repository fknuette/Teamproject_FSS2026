from __future__ import annotations

import argparse
import gc
from pathlib import Path
import shutil
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from argument_parser import build_parser
from self_play_textarena import run_self_play
from grpo_training.cli import run_training
from grpo_training.models import merge_lora_adapter


def concat_jsonl(files: list[Path], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as out:
        for f in files:
            with f.open("r", encoding="utf-8") as src:
                shutil.copyfileobj(src, out)


def main() -> None:
    """Main entry point for the online GRPO loop."""
    parser = build_parser(context="online_grpo_loop")
    args = parser.parse_args()
    loop_count = args.iterations if args.iterations is not None else args.loop_count
    work_dir = Path(args.work_dir)
    traces_dir = work_dir / "traces"
    datasets_dir = work_dir / "datasets"
    ckpt_dir = work_dir / "checkpoints"
    traces_dir.mkdir(parents=True, exist_ok=True)
    datasets_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    policy_model = args.base_model

    all_trace_files: list[Path] = []

    for iter_idx in range(1, loop_count + 1):
        iter_trace = traces_dir / f"iter_{iter_idx}.jsonl"

        rollout_args = argparse.Namespace(
            env_id=args.env_id,
            model=policy_model,
            num_games=args.games_per_iter,
            output=str(iter_trace),
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )

        print(f"[Iter {iter_idx}] Rollout with model={policy_model}")
        run_self_play(rollout_args)
        gc.collect()
        torch.cuda.empty_cache()
        all_trace_files.append(iter_trace)
        merged_dataset = datasets_dir / f"train_until_iter_{iter_idx}.jsonl"
        concat_jsonl(all_trace_files, merged_dataset)

        current_dataset = datasets_dir / f"train_iter_{iter_idx}.jsonl"
        concat_jsonl([iter_trace], current_dataset)

        adapter_out = ckpt_dir / f"iter_{iter_idx}" / "lora_adapter"
        merged_out = ckpt_dir / f"iter_{iter_idx}" / "merged_model"

        train_args = argparse.Namespace(
            model=policy_model,
            old_policy_model=policy_model,  # Model that generated the rollout data
            data=str(current_dataset),
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
            merge_for_vllm=False,
            merged_output_dir="",
        )

        print(f"[Iter {iter_idx}] GRPO training on {current_dataset}")
        run_training(train_args)
        gc.collect()
        torch.cuda.empty_cache()

        if args.skip_merge:
            print(f"[Iter {iter_idx}] Merge skipped; next rollout still uses {policy_model}")
            continue

        print(f"[Iter {iter_idx}] Merging LoRA adapter into full model for next vLLM rollout")
        """
        merge_lora_adapter(
            base_model=policy_model,
            adapter_dir=str(adapter_out),
            merged_output_dir=str(merged_out),
        )
        policy_model = str(merged_out)
        """
    print("Online GRPO loop finished.")


if __name__ == "__main__":
    main()
