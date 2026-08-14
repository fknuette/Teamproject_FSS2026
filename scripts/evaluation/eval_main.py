"""Main orchestration for evaluation runs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]  # scripts/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_factory import AgentFactory
from checkpoint_registry import CheckpointRegistry, ROLES
from matchmaker import SimplePairMatchmaker
from matchmaker import RandomMatchmaker
from result_analyzer import evaluate_simple_matchmaking_winrate, evaluate_trueskill_eval_winrate
from run_eval_games import run_eval_games
from trueskill_rater import update_ratings_from_results


def main() -> None:
    """Main entry point for evaluation."""
    parser = argparse.ArgumentParser(description="Run evaluation games")

    # Determine default paths
    root_dir = Path(__file__).resolve().parents[2]  # scripts/ parent
    default_checkpoint_dir = root_dir / "scripts" / "runs" / "online_grpo" / "checkpoints"
    default_output_dir = root_dir / "scripts" / "runs" / "online_grpo" / "evals"

    parser.add_argument(
        "--eval-checkpoint",
        type=str,
        default=None,
        help="Checkpoint to evaluate (trueskill mode: defaults to latest iter_* in --checkpoint-dir)",
    )
    parser.add_argument(
        "--baseline-checkpoint",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Baseline (untrained) model; used as opponent in trueskill mode and fixed baseline in simple mode",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=str(default_checkpoint_dir),
        help=f"Directory to scan for iter_* checkpoints (trueskill mode; default: {default_checkpoint_dir})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=f"Directory to save eval results (default: {default_output_dir}/<mode>)",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.3,
        help="GPU memory utilization ratio for vLLM startup (0-1)",
    )
    parser.add_argument(
        "--mode",
        choices=["simple", "trueskill"],
        default="trueskill",
        help="Evaluation mode: 'simple' (fixed baseline) or 'trueskill' (persistent ratings)",
    )
    parser.add_argument(
        "--registry-path",
        type=str,
        default=None,
        help="Path to checkpoint_registry.json (trueskill mode; defaults to <output-dir>/checkpoint_registry.json)",
    )
    parser.add_argument(
        "--reset-registry",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Delete the existing registry before the run starts. Default is True; use --no-reset-registry to keep it.",
    )
    parser.add_argument(
        "--min-games-per-team-role",
        type=int,
        default=3,
        help="Minimum games where eval plays as Mafia and as Village (trueskill mode)",
    )
    parser.add_argument(
        "--register-checkpoint",
        type=str,
        default=None,
        help="Register this checkpoint path into the registry before running (trueskill mode)",
    )
    parser.add_argument(
        "--full",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    args = parser.parse_args()

    # Resolve eval checkpoint: auto-select latest iter_* if not explicitly provided
    if args.eval_checkpoint is None:
        discovered = _discover_checkpoints(Path(args.checkpoint_dir))
        if not discovered:
            raise SystemExit(
                f"No usable iter_* checkpoints found in {args.checkpoint_dir!r}. "
                "Pass --eval-checkpoint explicitly."
            )
        args.eval_checkpoint = discovered[-1][0]
        print(f"Auto-selected eval checkpoint: {discovered[-1][0]} ({args.eval_checkpoint})")

    # Resolve output directory based on mode if not explicitly provided
    if args.output_dir is None:
        args.output_dir = str(default_output_dir / args.mode)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "simple":
        _run_simple_mode(args, output_dir)
    else:
        _run_trueskill_mode(args, output_dir)


def _run_simple_mode(args, output_dir: Path) -> None:
    """Run evaluation in simple mode (eval vs fixed baseline)."""
    matchmaker = SimplePairMatchmaker(
        baseline_checkpoint=args.baseline_checkpoint,
    )
    checkpoint_path = _resolve_checkpoint_path(
        Path(args.checkpoint_dir), args.eval_checkpoint
    )
    matchups_dict = matchmaker.get_matchups(str(checkpoint_path))

    print(f"Evaluating checkpoint: {args.eval_checkpoint}")
    print(f"vs Baseline: {args.baseline_checkpoint}")
    print(f"Output directory: {output_dir}")
    print()

    output_path = output_dir / "results.jsonl"
    print("Running evaluation...")
    run_eval_games(
        matchups_dict=matchups_dict,
        output_path=output_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    print(f"Results saved to {output_path}")
    print()
    print(evaluate_simple_matchmaking_winrate(output_path))
    print()
    print("Evaluation complete!")


def _resolve_checkpoint_path(checkpoint_dir: Path, checkpoint_id: str) -> Path:
    """Prefer a merged model and otherwise return the existing LoRA checkpoint."""
    iter_dir = checkpoint_dir / checkpoint_id
    merged_model = iter_dir / "merged_model"
    if merged_model.is_dir():
        print(f"Using merged model: {merged_model}")
        return merged_model

    lora_checkpoint = iter_dir / "lora_adapter" / "final"
    print(f"No merged model found; using LoRA checkpoint: {lora_checkpoint}")
    return lora_checkpoint


def _discover_checkpoints(checkpoint_dir: Path) -> list[tuple[str, str]]:
    """Return (checkpoint_id, preferred model path) pairs for usable iter_* dirs."""
    results = []
    for iter_dir in sorted(checkpoint_dir.glob("iter_*")):
        model_path = _resolve_checkpoint_path(checkpoint_dir, iter_dir.name)
        if model_path.is_dir():
            results.append((iter_dir.name, str(model_path)))
    return results


def prepare_registry_path(registry_path: Path, *, reset_registry: bool) -> Path:
    """Return the registry path after optionally deleting an existing file."""
    if reset_registry and registry_path.exists():
        registry_path.unlink()
        print(f"Reset registry at: {registry_path}")
    return registry_path


def _run_trueskill_mode(args, output_dir: Path) -> None:
    """Run evaluation in TrueSkill mode with a persistent checkpoint registry."""
    # import pdb; pdb.set_trace()
    registry_path = (
        Path(args.registry_path)
        if args.registry_path
        else output_dir / CheckpointRegistry.DEFAULT_REGISTRY_FILENAME
    )
    registry_path = prepare_registry_path(registry_path, reset_registry=args.reset_registry)
    registry = CheckpointRegistry(registry_path)

    # Always ensure the baseline (untrained) model is in the registry
    if args.baseline_checkpoint not in registry.all_ids():
        registry.register(args.baseline_checkpoint, args.baseline_checkpoint)
        registry.save()
        print(f"Registered baseline: {args.baseline_checkpoint}")

    if args.full:
        discovered = _discover_checkpoints(Path(args.checkpoint_dir))
        newly_discovered = [
            (ckpt_id, ckpt_path)
            for ckpt_id, ckpt_path in discovered
            if ckpt_id not in registry.all_ids()
        ]
        for ckpt_id, ckpt_path in newly_discovered:
            ##registry.register(ckpt_id, ckpt_path)
            #registry.save()
            run_single_eval_game(ckpt_id, args, registry, output_dir)

    else:
        run_single_eval_game(args.eval_checkpoint, args, registry, output_dir)

def run_single_eval_game(eval_checkpoint, args, registry, output_dir: Path):
    if eval_checkpoint not in registry.all_ids():
        checkpoint_path = _resolve_checkpoint_path(
            Path(args.checkpoint_dir), eval_checkpoint
        )
        registry.register(eval_checkpoint, str(checkpoint_path))
        print(f"Discovered and registered checkpoint: {eval_checkpoint}")
        registry.save()

    matchmaker = RandomMatchmaker(
        registry=registry,
        min_games_per_team_role=args.min_games_per_team_role,
    )
    matchups_dict = matchmaker.get_matchups(eval_checkpoint)#str(Path(args.checkpoint_dir) / args.eval_checkpoint / "lora_adapter" / "final"))
    total_games = sum(matchups_dict.values())
    print(f"TrueSkill mode — evaluating checkpoint: {eval_checkpoint}")
    #print(f"Checkpoint dir: {args.checkpoint_dir}  ({len(discovered)} discovered)")
    #print(f"Registry: {registry_path}  ({len(registry.all_ids())} checkpoints)")
    print(f"Scheduled games: {total_games}")
    print()

    output_path = output_dir / "results.jsonl"
    print("Running evaluation...")
    results = run_eval_games(
        matchups_dict=matchups_dict,
        output_path=output_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        registry=registry,
    )

    print(f"Results saved to {output_path}")
    print()
    print("Updating TrueSkill ratings...")
    update_ratings_from_results(results, registry, fixed_baseline_checkpoint=args.baseline_checkpoint)
    #print(f"Registry saved to {registry_path}")
    print()
    _print_trueskill_leaderboard(registry)
    print()
    print(evaluate_trueskill_eval_winrate(output_path, eval_checkpoint))
    print()
    print("Evaluation complete!")


def _print_trueskill_leaderboard(registry: CheckpointRegistry) -> None:
    """Print a leaderboard sorted by average conservative skill estimate."""
    all_ids = registry.all_ids()
    if not all_ids:
        print("(no checkpoints in registry)")
        return

    def conservative(ckpt_id: str) -> float:
        entry = registry.get(ckpt_id)
        estimates = [entry.ratings[r].mu - 3 * entry.ratings[r].sigma for r in ROLES]
        return sum(estimates) / len(estimates)

    ranked = sorted(all_ids, key=conservative, reverse=True)

    col_w = 12
    header = f"{'Rank':<5} {'Checkpoint':<50}"
    for role in ROLES:
        header += f"  {role:>{col_w}}"
    print(header)
    print("-" * len(header))

    for rank, ckpt_id in enumerate(ranked, start=1):
        entry = registry.get(ckpt_id)
        short_name = ckpt_id.split("/")[-1] if "/" in ckpt_id else ckpt_id
        row = f"{rank:<5} {short_name:<50}"
        for role in ROLES:
            r = entry.ratings[role]
            cell = f"{r.mu:.1f}±{r.sigma:.1f}"
            row += f"  {cell:>{col_w}}"
        print(row)


if __name__ == "__main__":
    main()
