"""
CLI and main entry point for GRPO training.
"""

import argparse
import gc
from pathlib import Path

import torch
from transformers import AutoTokenizer

from argument_parser import build_parser
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
    
    
    if args.merge_for_vllm and args.merged_output_dir:
        merge_lora_adapter(
            base_model=args.model,
            adapter_dir=str(Path(args.output_dir) / "final"),
            merged_output_dir=args.merged_output_dir,
        )
        print(f"[INFO] Merged model saved to {args.merged_output_dir}")

    # Cleanup GPU memory before returning to the outer loop
    del trainer
    del policy_model
    del tokenizer
    del old_policy_model
    del ref_model
    del train_dataset
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    """CLI entry point for training."""
    parser = build_parser(context="cli")
    args = parser.parse_args()
    run_training(args)