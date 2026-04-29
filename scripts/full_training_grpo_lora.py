"""
GRPO (Group Relative Policy Optimization) Training with LoRA

This script implements GRPO training for LLMs playing Secret Mafia.
It loads self-play traces, computes group-relative advantages, and
updates the policy using LoRA adapters.

Expected JSONL format:
    {"game_id": 0, "observation": "...", "response": "...", "reward": 1.0, "player_id": 4, "turn_id": 0}
"""

from grpo_training.cli import main

if __name__ == "__main__":
    main()