"""
CLI and main entry point for GRPO training.
"""

import argparse
from pathlib import Path

from transformers import AutoTokenizer

from grpo_training.data import GRPODataset
from grpo_training.models import (
    setup_model_with_lora,
    setup_reference_model,
    setup_old_policy_model,
    merge_lora_adapter,
)
from grpo_training.trainer import GRPOTrainer


def run_training(args: argparse.Namespace) -> None:
    """
    Main training function.
    
    Can be called from online_grpo_loop.py or directly via CLI.
    """
    
    # Determine old policy model path (fall back to args.model if not specified)
    old_policy_model_path = args.old_policy_model if args.old_policy_model else args.model
    
    #Setup policy model with LoRA
    policy_model, tokenizer = setup_model_with_lora(
        model_name=args.model,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    
    # Setup frozen old policy model (generated the rollout data)
    old_policy_model = setup_old_policy_model(old_policy_model_path)
    
    # Setup frozen reference model for KL regularization
    ref_model = setup_reference_model(args.model)
    
    # Nur Tokenizer laden - dauert Sekunden, braucht kein GPU
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Setup dataset
    train_dataset = GRPODataset(
        data_path=args.data,
        tokenizer=tokenizer,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
    )
    
    # Check if we have data
    if len(train_dataset) == 0:
        print("[WARNING] No training samples found! Skipping training.")
        return
    # Setup trainer
    trainer = GRPOTrainer(
        policy_model=policy_model,
        old_policy_model=old_policy_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        bf16=args.bf16,
    )
    
    # Train
    trainer.train()
    
    # Optionally merge LoRA into base model
    if args.merge_for_vllm and args.merged_output_dir:
        merge_lora_adapter(
            base_model=args.model,
            adapter_dir=str(Path(args.output_dir) / "final"),
            merged_output_dir=args.merged_output_dir,
        )


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for CLI usage."""
    parser = argparse.ArgumentParser(
        description="GRPO Training with LoRA for TextArena self-play traces"
    )
    
    # Model arguments
    # parser.add_argument("--model", type=str, required=True,
    #                     help="Base model name or path (e.g., Qwen/Qwen3-8B)")
    # parser.add_argument("--data", type=str, required=True,
    #                     help="Path to training data JSONL")
    # parser.add_argument("--output-dir", type=str, required=True,
    #                     help="Output directory for LoRA adapter")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B",
                    help="Base model name or path (e.g., Qwen/Qwen3-8B)")
    parser.add_argument("--data", type=str, default="runs/online_grpo/datasets/train_until_iter_1.jsonl",
                    help="Path to training data JSONL")
    parser.add_argument("--output-dir", type=str, default="runs/online_grpo/checkpoints/test",
                    help="Output directory for LoRA adapter")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Batch size per device")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--max-prompt-length", type=int, default=1024,
                        help="Maximum prompt length in tokens")
    parser.add_argument("--max-completion-length", type=int, default=256,
                        help="Maximum completion length in tokens")
    parser.add_argument("--logging-steps", type=int, default=10,
                        help="Log every N optimizer steps")
    parser.add_argument("--save-steps", type=int, default=100,
                        help="Save checkpoint every N optimizer steps")
    parser.add_argument("--bf16", action="store_true",
                        help="Use bfloat16 precision")
    
    # LoRA arguments
    parser.add_argument("--lora-r", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32,
                        help="LoRA alpha (scaling factor)")
    parser.add_argument("--lora-dropout", type=float, default=0.05,
                        help="LoRA dropout")
    
    # Old policy model (checkpoint that generated the rollout data)
    parser.add_argument("--old-policy-model", type=str, default=None,
                        help="Checkpoint that generated the rollout data (for GRPO ratio computation)")
    
    # Merge arguments
    parser.add_argument("--merge-for-vllm", action="store_true",
                        help="Merge LoRA into base model after training")
    parser.add_argument("--merged-output-dir", type=str, default="",
                        help="Output directory for merged model")
    
    return parser


def main() -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()
    run_training(args)