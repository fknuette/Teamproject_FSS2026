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
    Compute GRPO loss (token-level PPO-clip objective + KL penalty).

    Loss = -E[ min(ratio * A, clip(ratio) * A) ] + kl_coef * KL(policy || ref)

    Ratio und Clipping werden pro Token gebildet und über die Antwort-Tokens
    gemittelt. Das hält jede einzelne Token-Ratio nahe 1, sodass das
    Clip-Fenster [1-eps, 1+eps] wirksam bleibt.

    Args:
        policy_model: Trainable policy model (with LoRA)
        old_policy_model: Frozen old policy model (generated rollout data)
        ref_model: Frozen reference model for KL regularization
        input_ids: Input token IDs
        attention_mask: Attention mask
        labels: Labels with -100 for prompt
        advantages: Group-relative advantages [batch_size]
        clip_epsilon: PPO clipping parameter
        kl_coef: KL divergence coefficient

    Returns:
        loss: Scalar loss tensor
        metrics: Dict with logging metrics
    """

    # Per-token log probs of the response under each model.
    # mask marks the response tokens (1) vs. prompt/padding (0).
    policy_tok, mask = compute_log_probs_per_token(
        policy_model, input_ids, attention_mask, labels
    )
    # Old policy and reference are frozen -> no gradient needed.
    with torch.no_grad():
        old_tok, _ = compute_log_probs_per_token(
            old_policy_model, input_ids, attention_mask, labels
        )
        ref_tok, _ = compute_log_probs_per_token(
            ref_model, input_ids, attention_mask, labels
        )

    # Per-token importance ratio: how much more/less likely the current policy
    # emits each token compared to the policy that generated the rollout.
    log_ratio = policy_tok - old_tok                 # [batch_size, seq_len-1]
    ratio = torch.exp(log_ratio)

    # Every token in a response shares that response's advantage.
    adv = advantages.unsqueeze(1)                    # [batch_size, 1]

    # PPO clipped surrogate: take the pessimistic (max of the negated) objective
    # so a token whose ratio ran far from 1 can't dominate the update.
    policy_loss_unclipped = -adv * ratio
    policy_loss_clipped = -adv * torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    per_token_loss = torch.max(policy_loss_unclipped, policy_loss_clipped)

    # Average only over real response tokens (ignore prompt/padding).
    valid = mask.sum().clamp(min=1.0)
    policy_loss = (per_token_loss * mask).sum() / valid

    # KL penalty keeps the policy close to the frozen reference model.
    # Sequence-level log prob = sum of the (masked) per-token log probs.
    policy_log_probs = policy_tok.sum(dim=-1)        # [batch_size]
    ref_log_probs = ref_tok.sum(dim=-1)              # [batch_size]

    log_ratio_ref = policy_log_probs - ref_log_probs
    kl_div = ((torch.exp(log_ratio_ref) - 1.0) - log_ratio_ref).mean()

    # Total objective: clipped policy loss regularized by the KL term.
    total_loss = policy_loss + kl_coef * kl_div

    # Masked statistics for logging (ratio is per token, so ignore padding).
    ratio_mean = (ratio * mask).sum() / valid
    ratio_var = ((ratio - ratio_mean) ** 2 * mask).sum() / valid
    metrics = {
        "policy_loss": policy_loss.item(),
        "kl_div": kl_div.item(),
        "total_loss": total_loss.item(),
        "ratio_mean": ratio_mean.item(),
        "ratio_std": ratio_var.sqrt().item(),
        "advantage_mean": advantages.mean().item(),
    }

    return total_loss, metrics