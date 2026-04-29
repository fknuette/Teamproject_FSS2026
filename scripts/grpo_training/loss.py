"""
GRPO loss computation functions.
"""

import torch
import torch.nn.functional as F


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
    old_policy_model: torch.nn.Module,
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
    
    ratio = policy_log_probs / old_policy_log_probs
    KL = KL(policy || ref)
    
    Args:
        policy_model: Trainable policy model (with LoRA)
        old_policy_model: Frozen old policy model (generated rollout data)
        ref_model: Frozen reference model for KL regularization
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
    
    # Compute log probs from old policy model (no grad)
    with torch.no_grad():
        old_log_probs = compute_sequence_log_probs(
            old_policy_model, input_ids, attention_mask, labels
        )
    
    # Compute log probs from reference model (no grad)
    with torch.no_grad():
        ref_log_probs = compute_sequence_log_probs(
            ref_model, input_ids, attention_mask, labels
        )
    
    # Compute probability ratio: exp(log_pi - log_pi_old)
    # This is used for the GRPO advantage clipping
    log_ratio = policy_log_probs - old_log_probs
    ratio = torch.exp(log_ratio)
    
    # Clipped ratio
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    
    # Policy loss (negative because we minimize)
    policy_loss_unclipped = -advantages * ratio
    policy_loss_clipped = -advantages * clipped_ratio
    policy_loss = torch.max(policy_loss_unclipped, policy_loss_clipped).mean()
    
    # KL divergence penalty using reference model (separate from ratio)
    log_ratio_ref = policy_log_probs - ref_log_probs
    kl_div = ((torch.exp(log_ratio_ref) - 1.0) - log_ratio_ref).mean()
    
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