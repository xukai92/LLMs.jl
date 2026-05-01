# LLMs.jl

End-to-end LLM inference server in Julia for Apple Silicon, using MPSGraph (Apple's graph compilation engine — the same backend that powers MLX) for fused GPU execution.

## Features

- **MPSGraph-based forward pass** — entire transformer layer (attention + MLP) compiled as a single fused graph via ObjectiveC.jl, eliminating intermediate memory round-trips
- Custom Metal.jl kernels for all ops (quantized matmul, FP16 matmul, RMSNorm, RoPE, SwiGLU, softmax, flash attention)
- MLX-compatible 4-bit quantization format (packed UInt32 weights + Float16 scales/biases, group_size=64)
- Radix-tree prefix caching (SGLang-style)
- OpenAI API-compatible HTTP server with SSE streaming

## Supported models

| Model | Status |
|-------|--------|
| Llama-3.2-3B-Instruct-4bit | Supported |
| Qwen3.5-4B-Instruct-4bit | Planned ([#9]) |

## Quick start

```bash
# Install dependencies
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Run tests
julia --project=. -e 'using Pkg; Pkg.test()'

# Generate text
julia --project=. scripts/generate.jl

# Start OpenAI-compatible server
julia --project=. scripts/serve.jl
```

## Performance

All benchmarks on M3 Max with Llama-3.2-3B-Instruct-4bit.

### Quality: MATH50

50 problems sampled from [MATH500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500), verified by extracting `\boxed{}` answers.

| | Accuracy | Throughput |
|-|----------|------------|
| MLX (baseline) | 32.0% (16/50) | 153 tok/s |
| MLX (first 10, max_tokens=512) | 50.0% (5/10) | 154.3 tok/s |
| LLMs.jl q4-opt (first 10, max_tokens=512) | 50.0% (5/10) | 36.5 tok/s |

Run the MLX baseline:

```bash
uv run --with mlx --with mlx-lm python3 scripts/bench_math50.py --max-tokens 512
```

Run the LLMs.jl quantized backend:

```bash
julia --project=. scripts/bench_math50.jl --backend q4-opt --max-tokens 512
```

Use `--limit N` on either benchmark for shorter smoke runs. The Julia runner writes per-problem JSONL and summary artifacts under `results/`.

Note: Llama uses MLX's non-traditional split-half RoPE convention. Using adjacent-pair RoPE caused long chat/math prompts to collapse into repetitive punctuation/digits while simple short prompts still looked plausible.

### End-to-end throughput (MPSGraph, FP16, 28-layer forward pass)

| Mode | LLMs.jl | MLX | Speedup |
|------|---------|-----|---------|
| Prefill seq=8 | 401 tok/s (19.9ms) | 183 tok/s (43.8ms) | **2.20x** |
| Prefill seq=32 | 1019 tok/s (31.4ms) | 601 tok/s (53.2ms) | **1.69x** |
| Prefill seq=64 | 1480 tok/s (43.3ms) | 1182 tok/s (54.1ms) | **1.25x** |
| Decode B=1 | 47 tok/s (21ms) | 29 tok/s (34ms) | **1.62x** |

The fused graph compiles all 28 layers into a single MPSGraph dispatch, eliminating per-layer overhead. Decode includes full KV cache attention.

### Kernel microbenchmarks (3072x3072)

**FP16 matmul:**

| Batch | Julia (ms) | MLX (ms) | Julia/MLX |
|-------|-----------|---------|-----------|
| 4 | 0.43 | 0.40 | 1.08x |
| 32 | 0.41 | 0.48 | **0.85x** |
| 64 | 0.54 | 0.69 | **0.78x** |
| 128 | 0.71 | 0.77 | **0.93x** |
| 256 | 1.04 | 0.67 | 1.55x |

**Quantized 4-bit matmul:**

| Batch | Julia (ms) | MLX (ms) | Julia/MLX |
|-------|-----------|---------|-----------|
| 8 | 0.42 | 0.40 | **1.04x** |
| 32 | 0.43 | 0.27 | 1.61x |
| 64 | 0.52 | 0.31 | 1.69x |
| 128 | 0.74 | 0.42 | 1.77x |
| 256 | 1.10 | 0.63 | 1.75x |

Per-kernel, Julia beats or matches MLX at small batch sizes. The remaining gap at B>=256 is from load instruction overhead in compiled IR ([details](docs/metal-matmul-optimization.md)). The MPSGraph approach sidesteps this by using Apple's optimized kernels directly.

## Architecture

```
src/
  metal_graph.jl             # MPSGraph wrapper (ObjectiveC.jl FFI)
  graph_forward.jl           # Fused transformer graphs (prefill + decode)
  metal/                     # Custom GPU kernels
    fp16_matmul.jl           #   FP16 simdgroup GEMM (ptr+vec2)
    quantized_matmul_sg.jl   #   4-bit quantized simdgroup GEMM
    quantized_matmul_v2.jl   #   4-bit scalar GEMM (best for B<=8)
    flash_attention.jl       #   Flash attention (online softmax)
    rmsnorm.jl               #   RMSNorm + fused residual
    rope.jl                  #   RoPE with Llama3 frequency scaling
    swiglu.jl                #   SwiGLU activation
    softmax.jl               #   Numerically stable softmax
    fused_mlp.jl             #   Fused gate+up+SwiGLU
    fused_qkv.jl             #   Fused Q+K+V projection
    argmax.jl                #   GPU argmax
  model.jl                   # Model loading (safetensors, MLX format)
  forward_optimized.jl       # Kernel-based forward pass
  forward_fp16.jl            # FP16 forward pass (dequantized)
  kvcache.jl                 # Pre-allocated KV cache
  buffer_pool.jl             # GPU buffer pool with sized() views
  prefix_cache.jl            # Radix tree prefix cache
  server.jl                  # HTTP server with SSE streaming
  tokenizer.jl               # Tokenizer (via Python transformers)
  gpucompiler_patch.jl       # GPUCompiler.jl trap elimination patch
  metal_simd_patch.jl        # Metal.jl mixed-precision MAC patch
```

## Roadmap

- [#20] — Integrate MPSGraph forward pass with model loading and generation
- [#21] — Quantized weights in MPSGraph forward pass
- [#22] — MPSGraphExecutable pre-compilation
- [#9] — Qwen3.5-4B-Instruct-4bit support
- [#10] — PrecompileTools.jl for fast startup
- [#11] — LoRA adaptor support
- [#18] — SQuat KV cache quantization

## Upstream patches

Several patches to JuliaGPU packages are included as monkey-patches:

| Patch | File | Upstream |
|-------|------|---------|
| `@inline convert_origin` | Metal.jl simd.jl | [#8] |
| Mixed-precision `simdgroup_multiply_accumulate` | `metal_simd_patch.jl` | [#7] |
| Trap elimination on macOS 15+ | `gpucompiler_patch.jl` | [#6] |

## License

MIT

[#5]: https://github.com/xukai92/LLMs.jl/issues/5
[#6]: https://github.com/xukai92/LLMs.jl/issues/6
[#7]: https://github.com/xukai92/LLMs.jl/issues/7
[#8]: https://github.com/xukai92/LLMs.jl/issues/8
[#9]: https://github.com/xukai92/LLMs.jl/issues/9
[#10]: https://github.com/xukai92/LLMs.jl/issues/10
[#11]: https://github.com/xukai92/LLMs.jl/issues/11
[#18]: https://github.com/xukai92/LLMs.jl/issues/18
[#20]: https://github.com/xukai92/LLMs.jl/issues/20
[#21]: https://github.com/xukai92/LLMs.jl/issues/21
[#22]: https://github.com/xukai92/LLMs.jl/issues/22
