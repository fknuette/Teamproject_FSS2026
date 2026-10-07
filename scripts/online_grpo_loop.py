from __future__ import annotations

import argparse
import gc
from pathlib import Path
import shutil
import sys
import subprocess
import time

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


def _iter_sort_key(iter_name: str) -> tuple[int, str]:
    """Natural numeric ordering for names like iter_1, iter_10."""
    prefix, _, suffix = iter_name.partition("_")
    try:
        return (0, int(suffix))
    except ValueError:
        return (1, iter_name)


def prune_merged_models(ckpt_dir: Path, keep: int = 2) -> None:
    """Remove oldest `merged_model` subdirectories in *ckpt_dir* to keep at most *keep*.

    Only the `merged_model` subdirectory is removed — the outer `iter_*` directory
    and any LoRA adapter artifacts are left untouched. Registry entries remain.
    """
    merged_dirs: list[Path] = []
    for iter_dir in sorted(ckpt_dir.glob("iter_*"), key=lambda p: _iter_sort_key(p.name)):
        merged = iter_dir / "merged_model"
        if merged.is_dir():
            merged_dirs.append(merged)

    if len(merged_dirs) <= keep:
        return

    num_remove = len(merged_dirs) - keep
    to_remove = merged_dirs[:num_remove]
    for path in to_remove:
        try:
            shutil.rmtree(path, ignore_errors=True)
            print(f"[Prune] Removed old merged model: {path}")
        except Exception:
            print(f"[Prune] Failed to remove {path}; continuing")


def build_in_training_eval_command(
    work_dir: Path,
    checkpoint_dir: Path,
    iter_idx: int,
    *,
    gpu_memory_utilization: float = 0.25,
    eval_window_size: int = 1,
    min_games_per_team_role: int = 3,
    baseline_checkpoint: str = "Qwen/Qwen2.5-7B-Instruct",
    eval_output_dir: Path | None = None,
    eval_registry_path: Path | None = None,
    eval_script: Path | None = None,
) -> tuple[list[str], Path, Path]:
    """Build a per-iteration TrueSkill eval command.

    All eval configuration lives in the CLI arguments and is forwarded here to keep
    the train-time evaluator aligned with the standalone eval entrypoint.
    """
    eval_root = Path(eval_output_dir) if eval_output_dir is not None else work_dir / "evals" / "trueskill"
    output_dir = eval_root / f"iter_{iter_idx}"
    registry_path = Path(eval_registry_path) if eval_registry_path is not None else eval_root / "checkpoint_registry.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    registry_path.parent.mkdir(parents=True, exist_ok=True)

    if eval_script is None:
        eval_script = Path(__file__).resolve().parent / "evaluation" / "eval_main.py"

    cmd = [
        sys.executable,
        str(eval_script),
        "--mode",
        "trueskill",
        "--eval-checkpoint",
        f"iter_{iter_idx}",
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--output-dir",
        str(output_dir),
        "--registry-path",
        str(registry_path),
        "--eval-window-size",
        str(max(0, int(eval_window_size))),
        "--min-games-per-team-role",
        str(max(1, int(min_games_per_team_role))),
        "--baseline-checkpoint",
        str(baseline_checkpoint),
        "--no-reset-registry",
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
    ]
    return cmd, output_dir, registry_path


def main() -> None:
    """Main entry point for the online GRPO loop."""
    parser = build_parser(context="online_grpo_loop")
    args = parser.parse_args()
    loop_count = args.iterations if args.iterations is not None else args.loop_count
    work_dir = Path(args.work_dir)
    traces_dir = work_dir / "traces"
    datasets_dir = work_dir / "datasets"
    ckpt_dir = work_dir / "checkpoints"

    # If evaluation will be performed inside training and this is a fresh
    # training (no existing iter_* checkpoints), remove the previous runs
    # directory and any existing registry so evaluation starts fresh.
    if getattr(args, "eval_inside_training", False):
        try:
            if work_dir.exists():
                shutil.rmtree(work_dir)
                print(f"[Registry] Removed previous runs directory for fresh in-training eval: {work_dir}")
        except Exception:
            print(f"[Registry] Failed to remove previous runs directory: {work_dir}; continuing")

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
            num_players=args.num_players,
            num_mafia=args.num_mafia,
            output=str(iter_trace),
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )

        print(f"[Iter {iter_idx}] Rollout with model={policy_model}")
        run_self_play(rollout_args)
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
            reference_model=args.base_model,
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
            merge_for_vllm=True,
            merged_output_dir=str(merged_out),
        )

        print(f"[Iter {iter_idx}] GRPO training on {current_dataset}")
        run_training(train_args)
        prev_policy_model = policy_model
        policy_model = str(merged_out)

        # Prune merged models to keep only the most recent N merged models
        prune_merged_models(ckpt_dir, keep=2)

        # Run synchronous TrueSkill evaluation for the newly created checkpoint
        if getattr(args, "eval_inside_training", False):
            # Only run evaluation every `eval_frequency` iterations.
            eval_freq = max(1, int(getattr(args, "eval_frequency", 1)))
            if iter_idx % eval_freq != 0:
                print(f"[Eval] Skipping evaluation for iter_{iter_idx} (eval-frequency={eval_freq})")
                # continue main loop without running external eval
                continue
            try:
                eval_root = Path(args.eval_output_dir) if getattr(args, "eval_output_dir", None) else work_dir / "evals" / "trueskill"
                registry_path = Path(args.eval_registry_path) if getattr(args, "eval_registry_path", None) else eval_root / "checkpoint_registry.json"
                eval_window_size = max(0, int(getattr(args, "eval_window_size", 1)))
                min_games_per_team_role = max(1, int(getattr(args, "min_games_per_team_role", 3)))
                eval_cmd, eval_output_dir, registry_path = build_in_training_eval_command(
                    work_dir=work_dir,
                    checkpoint_dir=ckpt_dir,
                    iter_idx=iter_idx,
                    gpu_memory_utilization=0.25,
                    eval_window_size=eval_window_size,
                    min_games_per_team_role=min_games_per_team_role,
                    baseline_checkpoint=getattr(args, "baseline_checkpoint", "Qwen/Qwen2.5-7B-Instruct"),
                    eval_output_dir=eval_root,
                    eval_registry_path=registry_path,
                )
                print(f"[Eval] Running TrueSkill eval subprocess: {' '.join(eval_cmd)}")
                print(f"[Eval] Results dir: {eval_output_dir}")
                print(f"[Eval] Registry path: {registry_path}")
                # ensure some cleanup before spawning eval subprocess
                gc.collect()
                torch.cuda.empty_cache()
                time.sleep(1)
                proc = subprocess.run(eval_cmd, cwd=str(Path(__file__).resolve().parents[1]), check=False)
                if proc.returncode != 0:
                    print(f"[Eval] Evaluation exited with code {proc.returncode}")
                else:
                    print(f"[Eval] Evaluation completed for iter_{iter_idx}")
            except Exception as exc:
                print(f"[Eval] Failed to run evaluation subprocess: {exc}")
        else:
            print("[Eval] Skipping external evaluation because evaluation is handled inside training.")
        
    # After full run, optionally archive the runs directory into `runs/final`.
    if getattr(args, "archive_runs", False):
        try:
            if work_dir.exists():
                from datetime import datetime

                final_dir = work_dir.parent / "final"
                final_dir.mkdir(parents=True, exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                dest = final_dir / f"run_{ts}"
                shutil.move(str(work_dir), str(dest))
                print(f"[Archive] Moved completed runs to {dest}")
        except Exception:
            print(f"[Archive] Failed to archive runs directory: {work_dir}; continuing")

    print("Online GRPO loop finished.")


if __name__ == "__main__":
    main()
