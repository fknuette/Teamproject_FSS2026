from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from self_play_textarena import run_self_play
from grpo_training.cli import run_training
from grpo_training.models import merge_lora_adapter


def concat_jsonl(files: list[Path], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as out:
        for f in files:
            with f.open("r", encoding="utf-8") as src:
                shutil.copyfileobj(src, out)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Online loop: rollout -> reward data -> GRPO LoRA -> next rollout")

    parser.add_argument("--env-id", type=str, default="SecretMafia-v0")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--opponent-model", type=str, default="")
    parser.add_argument("--loop-count", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--games-per-iter", type=int, default=3)
    parser.add_argument("--work-dir", type=str, default="runs/online_grpo")

    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--max-completion-length", type=int, default=256)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--bf16", action="store_true")

    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--skip-merge", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    loop_count = args.iterations if args.iterations is not None else args.loop_count

    work_dir = Path(args.work_dir)
    traces_dir = work_dir / "traces"
    datasets_dir = work_dir / "datasets"
    ckpt_dir = work_dir / "checkpoints"
    traces_dir.mkdir(parents=True, exist_ok=True)
    datasets_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    opponent_model = args.opponent_model or args.base_model
    policy_model = args.base_model

    all_trace_files: list[Path] = []

    for iter_idx in range(1, loop_count + 1):
        iter_trace = traces_dir / f"iter_{iter_idx}.jsonl"

        rollout_args = argparse.Namespace(
            env_id=args.env_id,
            model_a=policy_model,
            model_b=opponent_model,
            num_games=args.games_per_iter,
            output=str(iter_trace),
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            tensor_parallel_size=args.tensor_parallel_size,
        )

        print(f"[Iter {iter_idx}] Rollout with policy={policy_model} vs opponent={opponent_model}")
        run_self_play(rollout_args)

        all_trace_files.append(iter_trace)
        merged_dataset = datasets_dir / f"train_until_iter_{iter_idx}.jsonl"
        concat_jsonl(all_trace_files, merged_dataset)

        adapter_out = ckpt_dir / f"iter_{iter_idx}" / "lora_adapter"
        merged_out = ckpt_dir / f"iter_{iter_idx}" / "merged_model"

        train_args = argparse.Namespace(
            model=policy_model,
            old_policy_model=policy_model,  # Model that generated the rollout data
            data=str(merged_dataset),
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

        print(f"[Iter {iter_idx}] GRPO training on {merged_dataset}")
        run_training(train_args)

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
