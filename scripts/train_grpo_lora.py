from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from datasets import Dataset
from peft import LoraConfig
from peft import PeftModel
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

from teamproject_fss2026.textarena_utils import parse_model_response


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_dataset(rows: list[dict]) -> Dataset:
    prompts = []
    reference_actions = []
    target_rewards = []

    for row in rows:
        prompts.append(row["prompt"])
        reference_actions.append(row["action"])
        target_rewards.append(float(row.get("final_reward", 0.0)))

    return Dataset.from_dict(
        {
            "prompt": prompts,
            "reference_action": reference_actions,
            "target_reward": target_rewards,
        }
    )


def reward_from_action_match(completions, reference_action, target_reward, **kwargs):
    del kwargs
    rewards = []
    for completion, expected_action, base_reward in zip(completions, reference_action, target_reward):
        if isinstance(completion, dict):
            completion_text = completion.get("content", "")
        else:
            completion_text = str(completion)
        parsed = parse_model_response(completion_text)
        reward = float(base_reward) if parsed.action.strip() == str(expected_action).strip() else 0.0
        if parsed.reasoning:
            reward += 0.05
        rewards.append(reward)
    return rewards


def run_training(args: argparse.Namespace) -> None:
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found: {data_path}")

    rows = load_jsonl(data_path)
    dataset = build_dataset(rows)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        bf16=args.bf16,
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=args.model,
        args=training_args,
        train_dataset=dataset,
        reward_funcs=reward_from_action_match,
        peft_config=lora_config,
    )

    trainer.train()
    trainer.save_model(args.output_dir)


def merge_lora_adapter(base_model: str, adapter_dir: str, merged_output_dir: str) -> None:
    adapter_path = Path(adapter_dir)
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_path}")

    merged_path = Path(merged_output_dir)
    merged_path.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype="auto", device_map="auto")
    model_with_adapter = PeftModel.from_pretrained(base, str(adapter_path))
    merged = model_with_adapter.merge_and_unload()
    merged.save_pretrained(str(merged_path))
    tokenizer.save_pretrained(str(merged_path))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train LoRA with GRPO on TextArena traces.")
    parser.add_argument("--model", type=str, required=True, help="Base model (e.g., Qwen/Qwen2.5-3B-Instruct)")
    parser.add_argument("--data", type=str, default="data/selfplay_traces.jsonl")
    parser.add_argument("--output-dir", type=str, default="outputs/grpo-lora")

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)

    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--max-completion-length", type=int, default=256)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--bf16", action="store_true")

    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--merge-for-vllm", action="store_true")
    parser.add_argument("--merged-output-dir", type=str, default="")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_training(args)
    if args.merge_for_vllm:
        merged_out = args.merged_output_dir or f"{args.output_dir}/merged"
        merge_lora_adapter(
            base_model=args.model,
            adapter_dir=args.output_dir,
            merged_output_dir=merged_out,
        )


if __name__ == "__main__":
    main()
