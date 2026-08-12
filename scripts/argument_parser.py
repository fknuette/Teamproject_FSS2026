"""
Centralized argument parser with context-based argument groups.

Argument groups are selectively added based on context:
- loop: Online loop orchestration args
- rollout: Inference parameters (temperature, top-p, etc.)
- self_play_args: Self-play specific args (model-a, model-b, num-games, output)
- training: Training parameters (epochs, batch-size, learning-rate, etc.)
- lora: LoRA configuration (rank, alpha, dropout)
- model_data: Model and data paths
- old_policy: Old policy model checkpoint
- merge: Model merge options

Usage:
    build_parser(context="online_grpo_loop")  # Full parser
    build_parser(context="cli")               # Training only
    build_parser(context="self_play")         # Self-play only
"""

import argparse


def build_parser(context: str | None = None) -> argparse.ArgumentParser:
    """
    Build an argument parser with context-based argument groups.
    
    Args:
        context: One of "online_grpo_loop", "cli", "self_play", or None for full parser.
        
    Returns:
        An ArgumentParser instance configured for the given context.
    """
    
    parser = argparse.ArgumentParser(description="GRPO Training Pipeline")
    
    # Add arguments based on context
    if context == "cli":
        _add_training_args(parser)
        _add_lora_args(parser)
        _add_model_data_args(parser)
        _add_old_policy_args(parser)
        _add_merge_args(parser)
    elif context == "self_play":
        _add_rollout_args(parser)
        _add_self_play_args(parser)
    else:
        _add_loop_args(parser)
        _add_rollout_args(parser)
        _add_training_args(parser)
        _add_lora_args(parser)
        _add_model_data_args(parser)
        _add_old_policy_args(parser)
        _add_merge_args(parser)
    
    return parser


def _add_loop_args(parser: argparse.ArgumentParser) -> None:
    """Loop orchestration arguments."""
    parser.add_argument("--env-id", type=str, default="SecretMafia-v0")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--loop-count", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--games-per-iter", type=int, default=3)
    parser.add_argument("--work-dir", type=str, default="runs/online_grpo")
    parser.add_argument("--skip-merge", action="store_true")


def _add_rollout_args(parser: argparse.ArgumentParser) -> None:
    """Rollout/inference arguments (vLLM)."""
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.3,
        help="GPU memory utilization ratio for vLLM startup (0-1)",
    )


def _add_self_play_args(parser: argparse.ArgumentParser) -> None:
    """Self-play specific arguments."""
    parser.add_argument("--model", type=str, required=True, help="Model to use for self-play")
    parser.add_argument("--num-games", type=int, default=50, help="Number of games")
    parser.add_argument("--output", type=str, default="data/selfplay_traces.jsonl", help="Output path")
    parser.add_argument("--env-id", type=str, default="TicTacToe-v0", help="Environment ID")


def _add_training_args(parser: argparse.ArgumentParser) -> None:
    """Training arguments."""
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size per device")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8,
                       help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--max-prompt-length", type=int, default=1024,
                       help="Maximum prompt length in tokens")
    parser.add_argument("--max-completion-length", type=int, default=256,
                       help="Maximum completion length in tokens")
    parser.add_argument("--logging-steps", type=int, default=10, help="Log every N optimizer steps")
    parser.add_argument("--save-steps", type=int, default=100, help="Save checkpoint every N steps")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 precision")


def _add_lora_args(parser: argparse.ArgumentParser) -> None:
    """LoRA arguments."""
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha (scaling factor)")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")


def _add_model_data_args(parser: argparse.ArgumentParser) -> None:
    """Model and data arguments."""
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B",
                       help="Base model name or path (e.g., Qwen/Qwen3-8B)")
    parser.add_argument("--data", type=str, default="runs/online_grpo/datasets/train_until_iter_1.jsonl",
                       help="Path to training data JSONL")
    parser.add_argument("--output-dir", type=str, default="runs/online_grpo/checkpoints/test",
                       help="Output directory for LoRA adapter")


def _add_old_policy_args(parser: argparse.ArgumentParser) -> None:
    """Old policy model argument for GRPO ratio computation."""
    parser.add_argument("--old-policy-model", type=str, default=None,
                       help="Checkpoint that generated the rollout data (for GRPO ratio computation)")


def _add_merge_args(parser: argparse.ArgumentParser) -> None:
    """Merge arguments."""
    parser.add_argument("--merge-for-vllm", action="store_true",
                       help="Merge LoRA into base model after training")
    parser.add_argument("--merged-output-dir", type=str, default="",
                       help="Output directory for merged model")

