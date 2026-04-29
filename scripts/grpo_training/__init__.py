"""GRPO Training package for TextArena self-play."""

from grpo_training.data import GRPODataset, TrainingSample, collate_fn
from grpo_training.models import (
    setup_model_with_lora,
    setup_reference_model,
    setup_old_policy_model,
    merge_lora_adapter,
)
from grpo_training.loss import grpo_loss
from grpo_training.trainer import GRPOTrainer

__all__ = [
    "GRPODataset",
    "TrainingSample",
    "collate_fn",
    "setup_model_with_lora",
    "setup_reference_model",
    "setup_old_policy_model",
    "merge_lora_adapter",
    "grpo_loss",
    "GRPOTrainer",
]