"""
GRPO Trainer class.
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from grpo_training.data import GRPODataset, collate_fn
from grpo_training.loss import grpo_loss
from grpo_training.models import setup_model_with_lora, setup_reference_model, setup_old_policy_model, merge_lora_adapter


class GRPOTrainer:
    """GRPO Trainer for LLM fine-tuning with LoRA."""
    
    def __init__(
        self,
        policy_model,
        old_policy_model: AutoModelForCausalLM,
        ref_model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        train_dataset: GRPODataset,
        output_dir: str,
        epochs: int = 1,
        batch_size: int = 2,
        gradient_accumulation_steps: int = 8,
        learning_rate: float = 1e-5,
        clip_epsilon: float = 0.2,
        kl_coef: float = 0.1,
        max_grad_norm: float = 1.0,
        logging_steps: int = 10,
        save_steps: int = 100,
        bf16: bool = True,
    ):
        self.policy_model = policy_model
        self.old_policy_model = old_policy_model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training hyperparameters
        self.epochs = epochs
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.learning_rate = learning_rate
        self.clip_epsilon = clip_epsilon
        self.kl_coef = kl_coef
        self.max_grad_norm = max_grad_norm
        self.logging_steps = logging_steps
        self.save_steps = save_steps
        self.bf16 = bf16
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.policy_model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )
        
        # Setup dataloader
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=True,
        )
        
        # Training state
        self.global_step = 0
        self.epoch = 0
    
    def train(self) -> None:
        """Run the GRPO training loop."""
        
        print(f"\n[Training] Starting GRPO training")
        print(f"  Epochs: {self.epochs}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Gradient accumulation steps: {self.gradient_accumulation_steps}")
        print(f"  Effective batch size: {self.batch_size * self.gradient_accumulation_steps}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Clip epsilon: {self.clip_epsilon}")
        print(f"  KL coefficient: {self.kl_coef}")
        print(f"  Total samples: {len(self.train_dataset)}")
        print(f"  Steps per epoch: {len(self.train_dataloader)}")
        
        self.policy_model.train()
        
        for epoch in range(self.epochs):
            self.epoch = epoch
            print(f"\n[Epoch {epoch + 1}/{self.epochs}]")
            
            epoch_metrics = {
                "policy_loss": [],
                "kl_div": [],
                "total_loss": [],
                "ratio_mean": [],
            }
            
            progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch + 1}")
            
            for step, batch in enumerate(progress_bar):
                # Move batch to GPU
                input_ids = batch["input_ids"].cuda()
                attention_mask = batch["attention_mask"].cuda()
                labels = batch["labels"].cuda()
                advantages = batch["advantages"].cuda()
                
                # Compute GRPO loss
                loss, metrics = grpo_loss(
                    policy_model=self.policy_model,
                    old_policy_model=self.old_policy_model,
                    ref_model=self.ref_model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    advantages=advantages,
                    clip_epsilon=self.clip_epsilon,
                    kl_coef=self.kl_coef,
                )
                
                # Scale loss for gradient accumulation
                scaled_loss = loss / self.gradient_accumulation_steps
                scaled_loss.backward()
                
                # Track metrics
                for key in epoch_metrics:
                    if key in metrics:
                        epoch_metrics[key].append(metrics[key])
                
                # Gradient accumulation step
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(
                        self.policy_model.parameters(),
                        self.max_grad_norm,
                    )
                    
                    # Optimizer step
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1
                    
                    # Logging
                    if self.global_step % self.logging_steps == 0:
                        recent_metrics = {
                            k: sum(v[-self.logging_steps * self.gradient_accumulation_steps:]) / 
                               len(v[-self.logging_steps * self.gradient_accumulation_steps:])
                            for k, v in epoch_metrics.items() if v
                        }
                        progress_bar.set_postfix({
                            "loss": f"{recent_metrics.get('total_loss', 0):.4f}",
                            "kl": f"{recent_metrics.get('kl_div', 0):.4f}",
                            "ratio": f"{recent_metrics.get('ratio_mean', 0):.3f}",
                        })
                    
                    # Save checkpoint
                    if self.global_step % self.save_steps == 0:
                        self._save_checkpoint()
            
            # End of epoch summary
            print(f"\n[Epoch {epoch + 1}] Summary:")
            for key, values in epoch_metrics.items():
                if values:
                    print(f"  {key}: {sum(values) / len(values):.4f}")
        
        # Save final model
        self._save_checkpoint(final=True)
        print(f"\n[Training] Completed! Final model saved to {self.output_dir / 'final'}")
    
    def _save_checkpoint(self, final: bool = False) -> None:
        """Save LoRA adapter checkpoint."""
        if final:
            save_path = self.output_dir / "final"
        else:
            save_path = self.output_dir / f"checkpoint-{self.global_step}"
        
        print(f"\n[Saving] Checkpoint to {save_path}")
        self.policy_model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)