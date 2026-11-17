"""Optional training-time repetition objective, after Welleck et al. (2019)."""
import torch
import torch.nn.functional as F


def token_unlikelihood(logits, inputs, targets, window=32):
    """Penalize unique recent context tokens, excluding the correct next token.

    Unlike decoding penalties this changes learned weights. This bounded-context
    variant is experimental; validation always measures plain cross-entropy.
    """
    candidates = F.pad(inputs, (window-1, 0), value=-1).unfold(1, window, 1).sort(-1).values
    valid = (candidates >= 0) & (candidates != targets.unsqueeze(-1)) & (targets.unsqueeze(-1) >= 0)
    unique = torch.ones_like(valid)
    unique[..., 1:] = candidates[..., 1:] != candidates[..., :-1]
    valid &= unique
    selected = logits.gather(-1, candidates.clamp_min(0))
    probability = (selected - logits.logsumexp(-1, keepdim=True)).exp().clamp_max(1-1e-6)
    losses = -torch.log1p(-probability) * valid
    return losses.sum() / (targets >= 0).sum().clamp_min(1)
