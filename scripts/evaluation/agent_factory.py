"""Factory for creating agents from checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict
import sys

ROOT = Path(__file__).resolve().parents[1]  # scripts/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vllm import LLM
from transformers import AutoTokenizer

from self_play_textarena import VLLMTextArenaAgent
import torch


class AgentFactory:
    """Factory for creating VLLMTextArenaAgent instances from checkpoint paths.
    
    Supports both full models and LoRA adapters.
    Caches LLM and tokenizer instances to avoid redundant loading.
    """
    
    def __init__(
        self,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.6,
    ):
        """Initialize the factory.
        
        Args:
            tensor_parallel_size: Number of GPUs for tensor parallelism.
            gpu_memory_utilization: Ratio of GPU memory to reserve for vLLM.
        """
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self._llm_cache: Dict[str, LLM] = {}
        self._tokenizer_cache: Dict[str, AutoTokenizer] = {}
        self._merged_cache: Dict[str, Path] = {}  # Maps adapter path to merged model path
    
    def create_agent(self, checkpoint_path: str) -> VLLMTextArenaAgent:
        """Create a VLLMTextArenaAgent from a checkpoint path.
        
        Args:
            checkpoint_path: Path to the checkpoint directory. Can be:
                - LoRA adapter: directory with adapter_config.json
                - Full model: directory with config.json
                - HuggingFace model: "Qwen/Qwen2.5-7B-Instruct"
            
        Returns:
            VLLMTextArenaAgent ready to play.
        """
        # Load or retrieve from cache
        llm = self._get_or_load_llm(checkpoint_path)
        tokenizer = self._get_or_load_tokenizer(checkpoint_path)
        
        return VLLMTextArenaAgent(llm, tokenizer)
    
    def _is_lora_checkpoint(self, checkpoint_path: str) -> bool:
        """Check if checkpoint is a LoRA adapter."""
        checkpoint_dir = Path(checkpoint_path)
        return (checkpoint_dir / "adapter_config.json").exists()
    
    def _merge_lora_checkpoint(self, checkpoint_path: str) -> str:
        """Merge LoRA adapter with base model and save to cache.
        
        Returns path to merged model.
        """
        if checkpoint_path in self._merged_cache:
            return str(self._merged_cache[checkpoint_path])
        
        print(f"Detected LoRA adapter at {checkpoint_path}")
        print("Merging LoRA weights with base model...")
        
        try:
            from peft import AutoPeftModelForCausalLM
        except ImportError:
            raise ImportError(
                "peft is required to load LoRA adapters. "
                "Install it with: pip install peft"
            )
        
        checkpoint_dir = Path(checkpoint_path)
        
        # Read adapter config to get base model name
        with open(checkpoint_dir / "adapter_config.json") as f:
            adapter_config = json.load(f)
        
        base_model_name = adapter_config.get("base_model_name_or_path")
        if not base_model_name:
            raise ValueError(
                f"Could not find base_model_name_or_path in "
                f"{checkpoint_dir / 'adapter_config.json'}"
            )
        
        print(f"Base model: {base_model_name}")
        
        # Load model with LoRA applied on CPU to avoid OOM during merge
        # The merge operation loads both base model and adapter, so we do it in CPU RAM
        model = AutoPeftModelForCausalLM.from_pretrained(
            checkpoint_path,
            device_map="cpu",
        )
        
        # Merge LoRA into base model (still on CPU)
        merged_model = model.merge_and_unload()
        
        # Save merged model to cache subdirectory
        merged_cache_dir = checkpoint_dir / ".merged_for_vllm"
        merged_cache_dir.mkdir(exist_ok=True)
        
        print(f"Saving merged model to {merged_cache_dir}")
        merged_model.save_pretrained(merged_cache_dir, safe_serialization=True)
        
        # Also save tokenizer from adapter
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        tokenizer.save_pretrained(merged_cache_dir)
        
        self._merged_cache[checkpoint_path] = merged_cache_dir
        print(f"Merged model ready at {merged_cache_dir}")
        
        return str(merged_cache_dir)
    
    def _get_or_load_llm(self, checkpoint_path: str) -> LLM:
        """Get LLM from cache or load it."""
        if checkpoint_path not in self._llm_cache:
            print(f"Loading LLM from {checkpoint_path}")
            
            # Check if it's a LoRA adapter and merge if needed
            model_path = checkpoint_path
            if self._is_lora_checkpoint(checkpoint_path):
                model_path = self._merge_lora_checkpoint(checkpoint_path)
            
            # Load with vLLM
            self._llm_cache[checkpoint_path] = LLM(
                model=model_path,
                tensor_parallel_size=self.tensor_parallel_size,
                gpu_memory_utilization=self.gpu_memory_utilization,
            )
        
        return self._llm_cache[checkpoint_path]
    
    def _get_or_load_tokenizer(self, checkpoint_path: str) -> AutoTokenizer:
        """Get tokenizer from cache or load it."""
        if checkpoint_path not in self._tokenizer_cache:
            print(f"Loading tokenizer from {checkpoint_path}")
            self._tokenizer_cache[checkpoint_path] = AutoTokenizer.from_pretrained(
                checkpoint_path
            )
        return self._tokenizer_cache[checkpoint_path]
    
    def clear_cache(self) -> None:
        """Clear cached LLMs and tokenizers."""
        self._llm_cache.clear()
        self._tokenizer_cache.clear()
