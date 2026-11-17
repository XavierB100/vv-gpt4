"""VV-GPT Transformer, carried forward from XavierB100/vv-gpt3 (802d1a8).

The original decoder remains the custom-model lab's foundation. VV-GPT4 adds
cached multi-token causality and explicit end-of-text stopping.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from typing import Dict, Optional
from dataclasses import dataclass

@dataclass
class GPTConfig:
    """Configuration for GPT model"""
    # Model architecture
    vocab_size: int = 50257
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    architecture: str = 'classic'

    def __post_init__(self):
        if self.architecture not in ('classic', 'revised'):
            raise ValueError('Unknown Transformer architecture.')
        if self.architecture == 'revised':
            if self.n_embd % self.n_head or (self.n_embd // self.n_head) % 2:
                raise ValueError('Rotary attention requires an even head dimension.')
            self.bias = False

    # Model size presets
    @classmethod
    def get_preset(cls, size: str, vocab_size: int, block_size: int = 1024):
        """Get predefined model configurations"""
        presets = {
            'nano': {'n_layer': 3, 'n_head': 3, 'n_embd': 144},
            'micro': {'n_layer': 4, 'n_head': 4, 'n_embd': 192},
            'mini': {'n_layer': 6, 'n_head': 6, 'n_embd': 192},
            'tiny': {'n_layer': 6, 'n_head': 6, 'n_embd': 384},
            'small': {'n_layer': 8, 'n_head': 8, 'n_embd': 512},
            'medium': {'n_layer': 12, 'n_head': 12, 'n_embd': 768},
            'large': {'n_layer': 16, 'n_head': 16, 'n_embd': 1024},
        }

        if size not in presets:
            raise ValueError(f"Unknown size: {size}. Available: {list(presets.keys())}")

        config = cls(vocab_size=vocab_size, block_size=block_size)
        for key, value in presets[size].items():
            setattr(config, key, value)

        return config

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class RMSNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x):
        normalized = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-5)
        return normalized.to(x.dtype) * self.weight


def norm(config):
    return RMSNorm(config.n_embd) if config.architecture == 'revised' else LayerNorm(config.n_embd, config.bias)


class RotaryEmbedding(nn.Module):
    def __init__(self, dimension, context):
        super().__init__()
        frequencies = 1.0 / (10000 ** (torch.arange(0, dimension, 2).float() / dimension))
        angles = torch.outer(torch.arange(context).float(), frequencies)
        self.register_buffer('cos', angles.cos(), persistent=False)
        self.register_buffer('sin', angles.sin(), persistent=False)

    def forward(self, x, offset=0):
        c = self.cos[offset:offset+x.size(-2)].to(x.dtype)
        s = self.sin[offset:offset+x.size(-2)].to(x.dtype)
        even, odd = x[..., ::2], x[..., 1::2]
        return torch.stack((even*c-odd*s, even*s+odd*c), dim=-1).flatten(-2)

class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with improvements"""

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # Key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.rotary = RotaryEmbedding(config.n_embd // config.n_head, config.block_size) if config.architecture == 'revised' else None

        # Causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                .view(1, 1, config.block_size, config.block_size))

    def forward(self, x, use_cache=False, past_key_value=None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # Calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        if self.rotary is not None:
            offset = past_key_value[0].size(-2) if past_key_value is not None else 0
            q, k = self.rotary(q, offset), self.rotary(k, offset)

        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)

        present = (k, v) if use_cache else None

        # Causal self-attention using PyTorch Flash Attention
        # (automatically handles causality and dropout efficiently)
        # Is causal only if query length > 1 (during training or prompt processing)
        is_causal = (past_key_value is None and q.size(2) > 1)

        mask = None
        if past_key_value is not None and T > 1:
            past_length = past_key_value[0].size(-2)
            mask = torch.arange(k.size(-2), device=x.device)[None, :] <= (
                past_length + torch.arange(T, device=x.device)[:, None]
            )

        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # Output projection
        y = self.resid_dropout(self.c_proj(y))

        if use_cache:
            return y, present
        return y

class MLP(nn.Module):
    """Multi-layer perceptron with GELU activation"""

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class GatedMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Three matrices instead of two; 8d/3 preserves approximately the 8d² FFN budget.
        hidden = 32 * math.ceil((8 * config.n_embd / 3) / 32)
        self.gate = nn.Linear(config.n_embd, hidden, bias=False)
        self.up = nn.Linear(config.n_embd, hidden, bias=False)
        self.c_proj = nn.Linear(hidden, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(F.silu(self.gate(x)) * self.up(x)))

class Block(nn.Module):
    """Transformer block with pre-normalization"""

    def __init__(self, config):
        super().__init__()
        self.ln_1 = norm(config)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = norm(config)
        self.mlp = GatedMLP(config) if config.architecture == 'revised' else MLP(config)

    def forward(self, x, use_cache=False, past_key_value=None):
        # Pre-normalization (more stable training)
        if use_cache:
            attn_out, present = self.attn(self.ln_1(x), use_cache=True, past_key_value=past_key_value)
            x = x + attn_out
            x = x + self.mlp(self.ln_2(x))
            return x, present
        else:
            x = x + self.attn(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
            return x

class GPT(nn.Module):
    """Enhanced GPT Model"""

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = norm(config),
        ))
        if config.architecture == 'classic':
            self.transformer['wpe'] = nn.Embedding(config.block_size, config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: share weights between input embedding and output projection
        # This typically reduces parameter count by ~20% with negligible performance hit
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize all weights
        self.apply(self._init_weights)

        # Apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # Report number of parameters
        print("Number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and 'wpe' in self.transformer:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        """Initialize weights following GPT-2 paper"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, use_cache=False, past_key_values=None):
        device = idx.device
        b, t = idx.size()

        if past_key_values is not None:
            past_length = past_key_values[0][0].size(-2)
        else:
            past_length = 0

        assert t + past_length <= self.config.block_size, f"Cannot forward sequence of length {t + past_length}, block size is only {self.config.block_size}"

        pos = torch.arange(past_length, t + past_length, dtype=torch.long, device=device) # shape (t)

        # Forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        x = self.transformer.drop(tok_emb + self.transformer.wpe(pos) if 'wpe' in self.transformer else tok_emb)

        presents = [] if use_cache else None

        for i, block in enumerate(self.transformer.h):
            past_kv = past_key_values[i] if past_key_values is not None else None
            if use_cache:
                x, present = block(x, use_cache=True, past_key_value=past_kv)
                presents.append(present)
            else:
                x = block(x)

        x = self.transformer.ln_f(x)

        if targets is not None:
            # If we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # Inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        if use_cache:
            return logits, loss, presents
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None,
                 repetition_penalty=1.0, no_repeat_ngram_size=0, generator=None, eos_token_id=None):
        """
        Fast generation using KV caching.
        """
        if idx.size(1) == 0:
            raise ValueError('A non-empty prompt is required.')
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('Temperature must be positive and finite.')
        past_key_values = None
        prompt_length = idx.size(1)
        if eos_token_id is None and self.config.vocab_size == 50257:
            eos_token_id = 50256
        for _ in range(max_new_tokens):
            if past_key_values is not None:
                # If the sequence context is growing too long we unfortunately can't cache past block_size
                if idx.size(1) >= self.config.block_size:
                    # Clear cache when we exceed context (fallback to slow generation for one step)
                    idx_cond = idx[:, -self.config.block_size:]
                    past_key_values = None
                else:
                    # We only need to pass the most recent token when caching is active
                    idx_cond = idx[:, -1:]
            else:
                idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]

            # Forward the model to get the logits for the index in the sequence
            logits, _, past_key_values = self(idx_cond, use_cache=True, past_key_values=past_key_values)

            # Pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature

            # Penalize generated content, not the user's prompt. A bounded window avoids
            # suppressing every common token in a long continuation.
            for row in range(idx.size(0)):
                generated = idx[row, prompt_length:].tolist()
                if generated and repetition_penalty != 1:
                    seen = list(set(generated[-64:]))
                    values = logits[row, seen]
                    logits[row, seen] = torch.where(values < 0, values * repetition_penalty, values / repetition_penalty)
                banned = banned_ngram_tokens(generated, no_repeat_ngram_size)
                if banned:
                    logits[row, list(banned)] = -float('inf')

            # Optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Optionally crop probabilities to only the top p options (nucleus sampling)
            if top_p is not None:
                # Sort logits in descending order
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                # Scatter sorted tensors to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')

            # Apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # Sample from the distribution
            if generator is None:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                idx_next = torch.multinomial(probs.cpu(), num_samples=1, generator=generator).to(idx.device)
            # Append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

            if eos_token_id is not None and bool((idx_next == eos_token_id).all()):
                break

        return idx


def banned_ngram_tokens(tokens, size):
    if size < 2 or len(tokens) < size-1:
        return set()
    prefix = tuple(tokens[-(size-1):])
    return {tokens[i+size-1] for i in range(len(tokens)-size+1)
            if tuple(tokens[i:i+size-1]) == prefix}


def repeat_fraction(tokens, size=4):
    grams = [tuple(tokens[i:i+size]) for i in range(len(tokens)-size+1)]
    return 1-len(set(grams))/len(grams) if grams else 0.0


# Example usage
if __name__ == "__main__":
    # Test different model sizes
    print("Testing different model configurations...")

    # Create a tiny model for testing
    config = GPTConfig.get_preset('tiny', vocab_size=100, block_size=256)
    print(f"Tiny model config: {config}")

    model = GPT(config)

    # Test forward pass
    x = torch.randint(0, config.vocab_size, (2, 10))  # batch=2, seq=10
    logits, loss = model(x, x)  # Use same tensor as targets for testing

    print(f"Input shape: {x.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")

    # Test generation
    print("\nTesting generation...")
    model.eval()
    with torch.no_grad():
        context = torch.zeros((1, 1), dtype=torch.long)
        generated = model.generate(context, max_new_tokens=20, temperature=1.0)
        print(f"Generated tokens: {generated[0].tolist()}")

    print("Model test successful!")
