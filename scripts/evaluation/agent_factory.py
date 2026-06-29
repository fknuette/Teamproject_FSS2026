"""Factory for creating agents from checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple
import sys

ROOT = Path(__file__).resolve().parents[1]  # scripts/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vllm import LLM
from transformers import AutoTokenizer

from self_play_textarena import VLLMTextArenaAgent


class AgentFactory:
    """Factory for creating VLLMTextArenaAgent instances from checkpoint paths.
    
    Caches LLM and tokenizer instances to avoid redundant loading.
    """
    
    def __init__(self, tensor_parallel_size: int = 1):
        """Initialize the factory.
        
        Args:
            tensor_parallel_size: Number of GPUs for tensor parallelism.
        """
        self.tensor_parallel_size = tensor_parallel_size
        self._llm_cache: Dict[str, LLM] = {}
        self._tokenizer_cache: Dict[str, AutoTokenizer] = {}
    
    def create_agent(self, checkpoint_path: str) -> VLLMTextArenaAgent:
        """Create a VLLMTextArenaAgent from a checkpoint path.
        
        Args:
            checkpoint_path: Path to the checkpoint directory. Can be:
                - Local path: "runs/online_grpo/checkpoints/iter_1/lora_adapter/final"
                - HuggingFace model: "Qwen/Qwen3-8B"
            
        Returns:
            VLLMTextArenaAgent ready to play.
        """
        # Load or retrieve from cache
        llm = self._get_or_load_llm(checkpoint_path)
        tokenizer = self._get_or_load_tokenizer(checkpoint_path)
        
        return VLLMTextArenaAgent(llm, tokenizer)
    
    def _get_or_load_llm(self, checkpoint_path: str) -> LLM:
        """Get LLM from cache or load it."""
        if checkpoint_path not in self._llm_cache:
            print(f"Loading LLM from {checkpoint_path}")
            self._llm_cache[checkpoint_path] = LLM(
                model=checkpoint_path,
                tensor_parallel_size=self.tensor_parallel_size,
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
