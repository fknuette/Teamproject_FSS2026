"""Main orchestration for evaluation runs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]  # scripts/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_factory import AgentFactory
from matchmaker import SimplePairMatchmaker
from result_analyzer import evaluate_simple_matchmaking_winrate
from self_play_textarena import run_eval_games


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
        default=str(default_checkpoint_dir / "iter_1" / "lora_adapter" / "final"),
        help=f"Checkpoint being evaluated (default: {default_checkpoint_dir}/iter_1/lora_adapter/final)",
    )
    parser.add_argument(
        "--baseline-checkpoint",
        type=str,
        default="Qwen/Qwen3-8B",
        help="Baseline checkpoint to play against",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(default_output_dir / "iter_1_vs_base"),
        help=f"Directory to save eval results (default: {default_output_dir}/iter_1_vs_base)",
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
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create matchmaker with defaults
    matchmaker = SimplePairMatchmaker(
        baseline_checkpoint=args.baseline_checkpoint,
    )
    
    # Get matchups
    matchups_dict = matchmaker.get_matchups(args.eval_checkpoint)
    
    print(f"Evaluating checkpoint: {args.eval_checkpoint}")
    print(f"vs Baseline: {args.baseline_checkpoint}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Run evaluation games
    output_path = output_dir / "results.jsonl"
    
    print(f"Running evaluation...")
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


if __name__ == "__main__":
    main()
