"""
GRPO (Group Relative Policy Optimization) Training with LoRA

This script implements GRPO training for LLMs playing Secret Mafia.
It loads self-play traces, computes group-relative advantages, and
updates the policy using LoRA adapters.

Expected JSONL format:
    {"game_id": 0, "observation": "...", "response": "...", "reward": 1.0, "player_id": 4, "turn_id": 0}
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
import sys

print("Importing torch...")
import torch
print("Torch imported!")
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel
from tqdm import tqdm
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class TrainingSample:
    """A single training sample with observation, response, and advantage."""
    observation: str
    response: str
    advantage: float
    game_id: int
    player_id: int = 0
    turn_id: int = 0


# =============================================================================
# Dataset
# =============================================================================

class GRPODataset(Dataset):
    """
    Dataset for GRPO training.
    
    Loads JSONL traces, groups by game_id, and computes normalized advantages
    within each group (game). Winners get positive advantages, losers negative.
    
    Expected JSONL fields:
        - game_id (required): For grouping
        - observation (required): Input for the model
        - response (required): Model output to learn from
        - reward (required): 1.0 for win, 0.0 for loss
        - player_id (optional): For debugging
        - turn_id (optional): For debugging
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: AutoTokenizer,
        max_prompt_length: int = 1024,
        max_completion_length: int = 256,
    ):
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.max_completion_length = max_completion_length
        self.samples: list[TrainingSample] = []
        
        self._load_and_process(data_path)
    
    def _load_and_process(self, data_path: str) -> None:
        """Load JSONL and compute group-relative advantages."""
        
        # Load all records
        records: list[dict] = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        
        print(f"[GRPODataset] Loaded {len(records)} turn records")
        
        # Group by game_id
        games: dict[int, list[dict]] = {}
        for record in records:
            game_id = record["game_id"]
            if game_id not in games:
                games[game_id] = []
            games[game_id].append(record)
        
        print(f"[GRPODataset] Found {len(games)} games")
        
        # Compute advantages within each game
        for game_id, game_records in games.items():
            # Get rewards for this game
            rewards = [r["reward"] for r in game_records]
            
            # Normalize rewards within the game (group-relative)
            mean_reward = sum(rewards) / len(rewards)
            variance = sum((r - mean_reward) ** 2 for r in rewards) / len(rewards)
            std_reward = math.sqrt(variance) if variance > 0 else 1.0
            
            # Avoid division by zero
            if std_reward < 1e-8:
                std_reward = 1.0
            
            for record in game_records:
                # Compute normalized advantage
                advantage = (record["reward"] - mean_reward) / std_reward
                
                # Create training sample
                sample = TrainingSample(
                    observation=record["observation"],
                    response=record["response"],
                    advantage=advantage,
                    game_id=game_id,
                    player_id=record.get("player_id", 0),
                    turn_id=record.get("turn_id", 0),
                )
                self.samples.append(sample)
        
        # Filter out samples with empty responses
        original_count = len(self.samples)
        self.samples = [s for s in self.samples if s.response.strip()]
        filtered_count = original_count - len(self.samples)
        
        if filtered_count > 0:
            print(f"[GRPODataset] Filtered {filtered_count} samples with empty responses")
        
        print(f"[GRPODataset] Created {len(self.samples)} training samples")
        
        # Print advantage statistics
        if self.samples:
            advantages = [s.advantage for s in self.samples]
            print(f"[GRPODataset] Advantage stats: "
                  f"mean={sum(advantages)/len(advantages):.3f}, "
                  f"min={min(advantages):.3f}, "
                  f"max={max(advantages):.3f}")
            
            # Print win/loss distribution
            wins = sum(1 for s in self.samples if s.advantage > 0)
            losses = len(self.samples) - wins
            print(f"[GRPODataset] Distribution: {wins} positive, {losses} negative advantages")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        
        # Tokenize observation (prompt)
        prompt_tokens = self.tokenizer(
            sample.observation,
            truncation=True,
            max_length=self.max_prompt_length,
            return_tensors="pt",
        )
        
        # Tokenize response (completion)
        response_tokens = self.tokenizer(
            sample.response,
            truncation=True,
            max_length=self.max_completion_length,
            add_special_tokens=False,
            return_tensors="pt",
        )
        
        # Combine prompt + response for full sequence
        input_ids = torch.cat([
            prompt_tokens["input_ids"].squeeze(0),
            response_tokens["input_ids"].squeeze(0),
        ], dim=0)
        
        attention_mask = torch.cat([
            prompt_tokens["attention_mask"].squeeze(0),
            response_tokens["attention_mask"].squeeze(0),
        ], dim=0)
        
        # Create labels: -100 for prompt tokens (don't compute loss), actual ids for response
        labels = input_ids.clone()
        prompt_length = prompt_tokens["input_ids"].shape[1]
        labels[:prompt_length] = -100  # Ignore prompt tokens in loss calculation
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "advantage": torch.tensor(sample.advantage, dtype=torch.float32),
            "prompt_length": prompt_length,
        }


def collate_fn(batch: list[dict]) -> dict:
    """Collate function with dynamic padding."""
    
    # Find max length in this batch
    max_length = max(item["input_ids"].shape[0] for item in batch)
    
    # Prepare lists for stacking
    input_ids = []
    attention_mask = []
    labels = []
    advantages = []
    prompt_lengths = []
    
    for item in batch:
        seq_len = item["input_ids"].shape[0]
        padding_length = max_length - seq_len
        
        # Pad input_ids with pad_token_id (or 0)
        padded_input_ids = F.pad(item["input_ids"], (0, padding_length), value=0)
        input_ids.append(padded_input_ids)
        
        # Pad attention_mask with 0 (ignore padding)
        padded_attention_mask = F.pad(item["attention_mask"], (0, padding_length), value=0)
        attention_mask.append(padded_attention_mask)
        
        # Pad labels with -100 (ignore in loss)
        padded_labels = F.pad(item["labels"], (0, padding_length), value=-100)
        labels.append(padded_labels)
        
        advantages.append(item["advantage"])
        prompt_lengths.append(item["prompt_length"])
    
    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "labels": torch.stack(labels),
        "advantages": torch.stack(advantages),
        "prompt_lengths": torch.tensor(prompt_lengths),
    }


# =============================================================================
# Model Setup
# =============================================================================

def setup_model_with_lora(
    model_name: str,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    device: str = "cuda",
) -> tuple[PeftModel, AutoTokenizer]:
    """
    Load base model and add LoRA adapters.
    
    Args:
        model_name: HuggingFace model name or path
        lora_r: LoRA rank
        lora_alpha: LoRA alpha (scaling)
        lora_dropout: Dropout for LoRA layers
        device: Device to load model on
    
    Returns:
        model: PeftModel with LoRA adapters (trainable)
        tokenizer: AutoTokenizer
    """
    print(f"[Setup] Loading base model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # Add LoRA adapters
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer


def setup_reference_model(
    model_name: str,
    device: str = "cuda",
) -> AutoModelForCausalLM:
    """
    Load frozen reference model for KL divergence calculation.
    
    Args:
        model_name: HuggingFace model name or path
        device: Device to load model on
    
    Returns:
        Frozen reference model
    """
    print(f"[Setup] Loading reference model: {model_name}")
    
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    
    # Freeze all parameters
    for param in ref_model.parameters():
        param.requires_grad = False
    
    ref_model.eval()
    return ref_model


# =============================================================================
# GRPO Loss Computation
# =============================================================================

def compute_log_probs_per_token(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-token log probabilities for the response tokens.
    
    Args:
        model: Language model
        input_ids: Input token IDs [batch_size, seq_len]
        attention_mask: Attention mask [batch_size, seq_len]
        labels: Labels with -100 for prompt tokens [batch_size, seq_len]
    
    Returns:
        token_log_probs: Per-token log probs [batch_size, seq_len-1]
        mask: Mask for response tokens [batch_size, seq_len-1]
    """
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = outputs.logits  # [batch_size, seq_len, vocab_size]
    
    # Shift for next-token prediction
    # logits[:, :-1, :] predicts labels[:, 1:]
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    
    # Compute log softmax
    log_probs = F.log_softmax(shift_logits, dim=-1)
    
    # Gather log probs for actual tokens
    token_log_probs = log_probs.gather(
        dim=-1,
        index=shift_labels.unsqueeze(-1).clamp(min=0)
    ).squeeze(-1)
    
    # Mask out prompt tokens (where labels == -100)
    mask = (shift_labels != -100).float()
    token_log_probs = token_log_probs * mask
    
    return token_log_probs, mask


def compute_sequence_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    Compute sequence-level log probability (sum of token log probs).
    
    Args:
        model: Language model
        input_ids: Input token IDs
        attention_mask: Attention mask
        labels: Labels with -100 for prompt
    
    Returns:
        Sequence log probabilities [batch_size]
    """
    token_log_probs, mask = compute_log_probs_per_token(
        model, input_ids, attention_mask, labels
    )
    
    # Sum log probs over sequence (only response tokens)
    sequence_log_probs = (token_log_probs * mask).sum(dim=-1)
    
    return sequence_log_probs


def grpo_loss(
    policy_model: torch.nn.Module,
    ref_model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    advantages: torch.Tensor,
    clip_epsilon: float = 0.2,
    kl_coef: float = 0.1,
) -> tuple[torch.Tensor, dict]:
    """
    Compute GRPO loss.
    
    Loss = -E[advantage * min(ratio, clip(ratio))] + kl_coef * KL(policy || ref)
    
    Args:
        policy_model: Trainable policy model (with LoRA)
        ref_model: Frozen reference model
        input_ids: Input token IDs
        attention_mask: Attention mask
        labels: Labels with -100 for prompt
        advantages: Group-relative advantages
        clip_epsilon: PPO clipping parameter
        kl_coef: KL divergence coefficient
    
    Returns:
        loss: Scalar loss tensor
        metrics: Dict with logging metrics
    """
    
    # Compute log probs from policy model
    policy_log_probs = compute_sequence_log_probs(
        policy_model, input_ids, attention_mask, labels
    )
    
    # Compute log probs from reference model (no grad)
    with torch.no_grad():
        ref_log_probs = compute_sequence_log_probs(
            ref_model, input_ids, attention_mask, labels
        )
    
    # Compute probability ratio: exp(log_pi - log_pi_ref)
    log_ratio = policy_log_probs - ref_log_probs
    ratio = torch.exp(log_ratio)
    
    # Clipped ratio
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    
    # Policy loss (negative because we minimize)
    policy_loss_unclipped = -advantages * ratio
    policy_loss_clipped = -advantages * clipped_ratio
    policy_loss = torch.max(policy_loss_unclipped, policy_loss_clipped).mean()
    
    # KL divergence penalty (approximate)
    kl_div = log_ratio.mean()
    
    # Total loss
    total_loss = policy_loss + kl_coef * kl_div
    
    # Metrics for logging
    metrics = {
        "policy_loss": policy_loss.item(),
        "kl_div": kl_div.item(),
        "total_loss": total_loss.item(),
        "ratio_mean": ratio.mean().item(),
        "ratio_std": ratio.std().item(),
        "advantage_mean": advantages.mean().item(),
    }
    
    return total_loss, metrics


# =============================================================================
# Training Loop
# =============================================================================

class GRPOTrainer:
    """GRPO Trainer for LLM fine-tuning with LoRA."""
    
    def __init__(
        self,
        policy_model: PeftModel,
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


# =============================================================================
# Merge LoRA Adapter
# =============================================================================

def merge_lora_adapter(
    base_model: str,
    adapter_dir: str,
    merged_output_dir: str,
) -> None:
    """
    Merge LoRA adapter weights into the base model.
    
    Creates a standalone model that can be used with vLLM
    without needing to load adapters separately.
    
    Args:
        base_model: HuggingFace model name or path
        adapter_dir: Path to LoRA adapter
        merged_output_dir: Output path for merged model
    """
    print(f"\n[Merge] Loading base model: {base_model}")
    print(f"[Merge] Loading adapter from: {adapter_dir}")
    
    # Load base model on CPU for merging
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(model, adapter_dir)
    
    # Merge adapter into base model
    print("[Merge] Merging adapter weights...")
    model = model.merge_and_unload()
    
    # Save merged model
    print(f"[Merge] Saving merged model to: {merged_output_dir}")
    Path(merged_output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(merged_output_dir)
    tokenizer.save_pretrained(merged_output_dir)
    
    print("[Merge] Done!")


# =============================================================================
# Main Entry Points
# =============================================================================

def run_training(args: argparse.Namespace) -> None:
    """
    Main training function.
    
    Can be called from online_grpo_loop.py or directly via CLI.
    """
    
    #Setup policy model with LoRA
    policy_model, tokenizer = setup_model_with_lora(
        model_name=args.model,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    
    # Setup frozen reference model
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


if __name__ == "__main__":
    main()