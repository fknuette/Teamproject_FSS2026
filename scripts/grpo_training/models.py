"""
Model setup functions for GRPO training.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel
from pathlib import Path


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


def setup_old_policy_model(
    model_name_or_path: str,
    device: str = "cuda",
) -> AutoModelForCausalLM:
    """
    Load old policy model (the one that generated the rollout data).
    
    Used for computing the importance sampling ratio in GRPO.
    
    Args:
        model_name_or_path: HuggingFace model name or path
        device: Device to load model on
    
    Returns:
        Frozen old policy model
    """
    print(f"[Setup] Loading old policy model: {model_name_or_path}")
    
    old_policy_model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    
    # Freeze all parameters
    old_policy_model.requires_grad_(False)
    old_policy_model.eval()
    return old_policy_model


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