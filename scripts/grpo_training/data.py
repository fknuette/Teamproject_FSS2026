"""
Data structures and dataset for GRPO training.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


@dataclass
class TrainingSample:
    """A single training sample with observation, response, and advantage."""
    observation: str
    response: str
    advantage: float
    game_id: int
    player_id: int = 0
    turn_id: int = 0


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