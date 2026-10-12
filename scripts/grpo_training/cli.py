"""
CLI and main entry point for GRPO training.
"""

import argparse
import gc
import itertools
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from argument_parser import build_parser
from grpo_training.data import GRPODataset, collate_fn
from grpo_training.loss import grpo_loss
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
    ref_model = setup_reference_model(
        getattr(args, "reference_model", args.model)
    )
    
    # Reuse tokenizer from policy model setup
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


def run_joint_training(args: argparse.Namespace) -> None:
    """Joint GRPO training over vote + defense JSONL datasets.

    Combined loss: total = lambda_vote * L_vote + lambda_defense * L_defense
    Each type uses its own GRPODataset so group-relative advantage normalization
    is independent.  Both losses are summed before a single backward pass.

    Extra args vs run_training:
        args.vote_data:     str   path to vote JSONL (empty string = skip)
        args.defense_data:  str   path to defense JSONL (empty string = skip)
        args.lambda_vote:   float loss weight for vote
        args.lambda_defense float loss weight for defense
    """
    old_policy_path = args.old_policy_model if args.old_policy_model else args.model

    policy_model, tokenizer = setup_model_with_lora(
        model_name=args.model,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    old_policy_model = setup_old_policy_model(old_policy_path)
    ref_model = setup_reference_model(getattr(args, "reference_model", args.model))

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lambda_vote = getattr(args, "lambda_vote", 1.0)
    lambda_defense = getattr(args, "lambda_defense", 1.0)

    def _make_loader(data_path: str):
        if not data_path or not Path(data_path).exists():
            return None
        ds = GRPODataset(data_path, tokenizer, args.max_prompt_length, args.max_completion_length)
        if len(ds) == 0:
            return None
        return DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                          collate_fn=collate_fn, num_workers=0)

    vote_loader = _make_loader(getattr(args, "vote_data", ""))
    defense_loader = _make_loader(getattr(args, "defense_data", ""))

    if vote_loader is None and defense_loader is None:
        print("[WARNING] Both vote and defense datasets are empty; skipping joint training.")
        del policy_model, tokenizer, old_policy_model, ref_model
        gc.collect()
        torch.cuda.empty_cache()
        return

    n_steps_per_epoch = max(len(vote_loader) if vote_loader else 0,
                            len(defense_loader) if defense_loader else 0)

    optimizer = torch.optim.AdamW(
        policy_model.parameters(), lr=args.learning_rate,
        betas=(0.9, 0.999), weight_decay=0.01,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    clip_epsilon = getattr(args, "clip_epsilon", 0.2)
    kl_coef = getattr(args, "kl_coef", 0.1)
    grad_accum = args.gradient_accumulation_steps

    policy_model.train()
    global_step = 0

    print(f"\n[JointTraining] Starting")
    print(f"  vote_data:    {getattr(args, 'vote_data', '')} "
          f"({'active' if vote_loader else 'empty/skipped'})")
    print(f"  defense_data: {getattr(args, 'defense_data', '')} "
          f"({'active' if defense_loader else 'empty/skipped'})")
    print(f"  lambda_vote={lambda_vote}, lambda_defense={lambda_defense}")
    print(f"  steps/epoch={n_steps_per_epoch}, epochs={args.epochs}")

    for epoch in range(args.epochs):
        vote_cycle = itertools.cycle(vote_loader) if vote_loader else None
        defense_cycle = itertools.cycle(defense_loader) if defense_loader else None

        for step in range(n_steps_per_epoch):
            total_loss = None
            vote_loss_val = 0.0
            defense_loss_val = 0.0

            if vote_cycle is not None:
                v_batch = next(vote_cycle)
                v_loss, v_metrics = grpo_loss(
                    policy_model=policy_model,
                    old_policy_model=old_policy_model,
                    ref_model=ref_model,
                    input_ids=v_batch["input_ids"].cuda(),
                    attention_mask=v_batch["attention_mask"].cuda(),
                    labels=v_batch["labels"].cuda(),
                    advantages=v_batch["advantages"].cuda(),
                    clip_epsilon=clip_epsilon,
                    kl_coef=kl_coef,
                )
                total_loss = lambda_vote * v_loss
                vote_loss_val = v_metrics["total_loss"]

            if defense_cycle is not None:
                d_batch = next(defense_cycle)
                d_loss, d_metrics = grpo_loss(
                    policy_model=policy_model,
                    old_policy_model=old_policy_model,
                    ref_model=ref_model,
                    input_ids=d_batch["input_ids"].cuda(),
                    attention_mask=d_batch["attention_mask"].cuda(),
                    labels=d_batch["labels"].cuda(),
                    advantages=d_batch["advantages"].cuda(),
                    clip_epsilon=clip_epsilon,
                    kl_coef=kl_coef,
                )
                total_loss = lambda_defense * d_loss if total_loss is None else total_loss + lambda_defense * d_loss
                defense_loss_val = d_metrics["total_loss"]

            if total_loss is None:
                continue

            (total_loss / grad_accum).backward()

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % args.logging_steps == 0:
                    print(
                        f"[JointTraining] epoch={epoch + 1} step={global_step} "
                        f"joint_loss={total_loss.item():.4f} "
                        f"vote={vote_loss_val:.4f} defense={defense_loss_val:.4f}"
                    )

                if global_step % args.save_steps == 0:
                    ckpt = output_dir / f"checkpoint-{global_step}"
                    policy_model.save_pretrained(str(ckpt))
                    tokenizer.save_pretrained(str(ckpt))

    final_dir = output_dir / "final"
    policy_model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"[JointTraining] Completed. Final adapter saved to {final_dir}")

    if args.merge_for_vllm and args.merged_output_dir:
        merge_lora_adapter(
            base_model=args.model,
            adapter_dir=str(final_dir),
            merged_output_dir=args.merged_output_dir,
        )
        print(f"[INFO] Merged joint model saved to {args.merged_output_dir}")

    del policy_model, tokenizer, old_policy_model, ref_model
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    """CLI entry point for training."""
    parser = build_parser(context="cli")
    args = parser.parse_args()
    run_training(args)
