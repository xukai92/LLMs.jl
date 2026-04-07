"""
MLX baseline benchmarks for comparison with our Julia implementation.

Measures kernel-level and end-to-end performance of mlx-lm.

Usage:
    uv run --with mlx-lm scripts/benchmark_mlx_baseline.py
"""

import time
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_sampler
import sys, os

MODEL_DIR = os.path.expanduser(
    "~/.cache/huggingface/hub/models--mlx-community--Llama-3.2-3B-Instruct-4bit/"
    "snapshots/7f0dc925e0d0afb0322d96f9255cfddf2ba5636e"
)

def bench(f, warmup=10, iters=100):
    """Benchmark a function, return median and min time in seconds."""
    for _ in range(warmup):
        f()
        mx.eval(mx.zeros(1))  # sync

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        f()
        mx.eval(mx.zeros(1))  # force sync
        t1 = time.perf_counter()
        times.append(t1 - t0)

    times.sort()
    median = times[len(times)//2]
    mn = times[0]
    return median, mn

def bench_kernels(model):
    """Benchmark individual operations at Llama-3.2-3B dimensions."""
    print("\n" + "="*80)
    print("MLX KERNEL BENCHMARKS — Llama-3.2-3B dimensions")
    print("="*80)

    # Get first layer for benchmarking
    layer = model.model.layers[0]

    # Hidden state (B=1, hidden=3072)
    x = mx.random.normal((1, 3072)).astype(mx.float16)

    # RMSNorm
    med, mn = bench(lambda: mx.eval(layer.input_layernorm(x.astype(mx.float32)).astype(mx.float16)))
    print(f"  RMSNorm:              median={med*1e3:.3f}ms  min={mn*1e3:.3f}ms")

    # Quantized matmul (q_proj: 3072→3072)
    med, mn = bench(lambda: mx.eval(layer.self_attn.q_proj(x)))
    print(f"  q_proj (3072→3072):   median={med*1e3:.3f}ms  min={mn*1e3:.3f}ms")

    med, mn = bench(lambda: mx.eval(layer.self_attn.k_proj(x)))
    print(f"  k_proj (3072→1024):   median={med*1e3:.3f}ms  min={mn*1e3:.3f}ms")

    med, mn = bench(lambda: mx.eval(layer.self_attn.v_proj(x)))
    print(f"  v_proj (3072→1024):   median={med*1e3:.3f}ms  min={mn*1e3:.3f}ms")

    med, mn = bench(lambda: mx.eval(layer.self_attn.o_proj(x)))
    print(f"  o_proj (3072→3072):   median={med*1e3:.3f}ms  min={mn*1e3:.3f}ms")

    # MLP projections
    med, mn = bench(lambda: mx.eval(layer.mlp.gate_proj(x)))
    print(f"  gate_proj (3072→8192):median={med*1e3:.3f}ms  min={mn*1e3:.3f}ms")

    med, mn = bench(lambda: mx.eval(layer.mlp.up_proj(x)))
    print(f"  up_proj (3072→8192):  median={med*1e3:.3f}ms  min={mn*1e3:.3f}ms")

    ffn_out = mx.random.normal((1, 8192)).astype(mx.float16)
    med, mn = bench(lambda: mx.eval(layer.mlp.down_proj(ffn_out)))
    print(f"  down_proj (8192→3072):median={med*1e3:.3f}ms  min={mn*1e3:.3f}ms")

    # SwiGLU
    gate = mx.random.normal((1, 8192)).astype(mx.float16)
    up = mx.random.normal((1, 8192)).astype(mx.float16)
    med, mn = bench(lambda: mx.eval(nn.silu(gate) * up))
    print(f"  SwiGLU:               median={med*1e3:.3f}ms  min={mn*1e3:.3f}ms")

def bench_e2e(model, tokenizer):
    """End-to-end generation benchmarks."""
    print("\n" + "="*80)
    print("MLX END-TO-END BENCHMARKS")
    print("="*80)

    sampler = make_sampler(temp=0.0)

    # TTFT
    print("\nTTFT (Time To First Token):")
    base_text = "Hello world. " * 200
    base_ids = tokenizer.encode(base_text)

    for plen in [16, 128, 512]:
        ids = base_ids[:plen]
        prompt = mx.array(ids)

        # Warmup
        for _, _ in zip(range(1), generate_step(prompt, model, max_tokens=1, sampler=sampler)):
            pass

        t0 = time.perf_counter()
        for _, _ in zip(range(1), generate_step(prompt, model, max_tokens=1, sampler=sampler)):
            pass
        ttft = time.perf_counter() - t0
        print(f"  prompt={plen:4d} tokens: TTFT={ttft:.3f}s  ({plen/ttft:.0f} tok/s prefill)")

    # Decode throughput
    print("\nDecode throughput:")
    prompt_text = "The quick brown fox jumps over the lazy dog. " * 10
    prompt_ids = tokenizer.encode(prompt_text)
    prompt = mx.array(prompt_ids)

    for gen_len in [20, 50, 128]:
        # Warmup
        for i, (tok, _) in enumerate(generate_step(prompt, model, max_tokens=gen_len, sampler=sampler)):
            if i >= gen_len - 1:
                break

        t0 = time.perf_counter()
        t_first = None
        for i, (tok, _) in enumerate(generate_step(prompt, model, max_tokens=gen_len, sampler=sampler)):
            if t_first is None:
                t_first = time.perf_counter() - t0
            if i >= gen_len - 1:
                break
        t_total = time.perf_counter() - t0

        decode_time = t_total - t_first
        decode_tps = (gen_len - 1) / decode_time if decode_time > 0 else 0
        overall_tps = gen_len / t_total
        print(f"  gen={gen_len:3d} tokens: TTFT={t_first:.3f}s  decode={decode_tps:.1f} tok/s  overall={overall_tps:.1f} tok/s  ({t_total:.2f}s total)")

def main():
    print(f"Loading model from {MODEL_DIR}...")
    model, tokenizer = load(MODEL_DIR)
    print(f"Model loaded. Peak memory: {mx.metal.get_peak_memory() / 1e9:.2f} GB")

    bench_kernels(model)
    bench_e2e(model, tokenizer)

    print(f"\nPeak memory: {mx.metal.get_peak_memory() / 1e9:.2f} GB")

if __name__ == "__main__":
    main()
