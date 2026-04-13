"""Benchmark MLX transformer layer for comparison with our MPSGraph implementation."""

import mlx.core as mx
import mlx.nn as nn
import time
import numpy as np

# Llama-3.2-3B dimensions
HIDDEN = 3072
HEAD_DIM = 128
N_Q_HEADS = 24
N_KV_HEADS = 8
INTERMEDIATE = 8192
N_LAYERS = 28
EPS = 1e-5

class LlamaAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(HIDDEN, N_Q_HEADS * HEAD_DIM, bias=False)
        self.k_proj = nn.Linear(HIDDEN, N_KV_HEADS * HEAD_DIM, bias=False)
        self.v_proj = nn.Linear(HIDDEN, N_KV_HEADS * HEAD_DIM, bias=False)
        self.o_proj = nn.Linear(N_Q_HEADS * HEAD_DIM, HIDDEN, bias=False)
        self.scale = HEAD_DIM ** -0.5

    def __call__(self, x, cos, sin):
        B, S, _ = x.shape
        q = self.q_proj(x).reshape(B, S, N_Q_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, S, N_KV_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, S, N_KV_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)

        # RoPE
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # GQA: repeat KV heads
        if N_Q_HEADS != N_KV_HEADS:
            n_rep = N_Q_HEADS // N_KV_HEADS
            k = mx.repeat(k, n_rep, axis=1)
            v = mx.repeat(v, n_rep, axis=1)

        # Attention
        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        mask = nn.MultiHeadAttention.create_additive_causal_mask(S)
        scores = scores + mask
        weights = mx.softmax(scores, axis=-1)
        out = (weights @ v).transpose(0, 2, 1, 3).reshape(B, S, -1)
        return self.o_proj(out)

class LlamaMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(HIDDEN, INTERMEDIATE, bias=False)
        self.up_proj = nn.Linear(HIDDEN, INTERMEDIATE, bias=False)
        self.down_proj = nn.Linear(INTERMEDIATE, HIDDEN, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))

class LlamaLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm = nn.RMSNorm(HIDDEN, eps=EPS)
        self.self_attn = LlamaAttention()
        self.post_attention_layernorm = nn.RMSNorm(HIDDEN, eps=EPS)
        self.mlp = LlamaMLP()

    def __call__(self, x, cos, sin):
        h = x + self.self_attn(self.input_layernorm(x), cos, sin)
        out = h + self.mlp(self.post_attention_layernorm(h))
        return out

def apply_rope(x, cos, sin):
    # x: (B, H, S, D)
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return mx.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)

def bench_single_layer(seq_len, n_warmup=20, n_iter=100):
    print(f"\n=== MLX Single Layer (seq={seq_len}) ===")
    layer = LlamaLayer()
    mx.eval(layer.parameters())

    x = mx.random.normal((1, seq_len, HIDDEN)).astype(mx.float16)
    cos = mx.random.normal((1, 1, seq_len, HEAD_DIM // 2)).astype(mx.float16)
    sin = mx.random.normal((1, 1, seq_len, HEAD_DIM // 2)).astype(mx.float16)

    # Warmup
    for _ in range(n_warmup):
        out = layer(x, cos, sin)
        mx.eval(out)

    # Benchmark
    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter_ns()
        out = layer(x, cos, sin)
        mx.eval(out)
        times.append((time.perf_counter_ns() - t0) / 1e6)

    med = np.median(times)
    mn = np.min(times)
    print(f"  Median: {med:.3f} ms")
    print(f"  Min:    {mn:.3f} ms")
    return med

def bench_multi_layer(seq_len, n_layers, n_warmup=10, n_iter=50):
    print(f"\n=== MLX {n_layers}-Layer Forward Pass (seq={seq_len}) ===")
    layers = [LlamaLayer() for _ in range(n_layers)]
    mx.eval([l.parameters() for l in layers])

    x = mx.random.normal((1, seq_len, HIDDEN)).astype(mx.float16)
    cos = mx.random.normal((1, 1, seq_len, HEAD_DIM // 2)).astype(mx.float16)
    sin = mx.random.normal((1, 1, seq_len, HEAD_DIM // 2)).astype(mx.float16)

    # Warmup
    for _ in range(n_warmup):
        h = x
        for layer in layers:
            h = layer(h, cos, sin)
        mx.eval(h)

    # Benchmark
    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter_ns()
        h = x
        for layer in layers:
            h = layer(h, cos, sin)
        mx.eval(h)
        times.append((time.perf_counter_ns() - t0) / 1e6)

    med = np.median(times)
    mn = np.min(times)
    tok_per_s = seq_len / (med / 1000)
    print(f"  Median: {med:.2f} ms ({tok_per_s:.0f} tok/s)")
    print(f"  Min:    {mn:.2f} ms")
    return med

if __name__ == "__main__":
    print("MLX Transformer Layer Benchmark")
    print(f"Hidden={HIDDEN}, Heads={N_Q_HEADS}/{N_KV_HEADS}, Inter={INTERMEDIATE}")

    for seq in [8, 16, 32, 64]:
        bench_single_layer(seq)

    print("\n" + "=" * 60)
    for seq in [8, 32, 64]:
        bench_multi_layer(seq, N_LAYERS)
