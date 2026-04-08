# LLMs.jl

End-to-end LLM inference server in Julia with custom Metal GPU kernels for Apple Silicon.

## Features

- Custom Metal.jl kernels for all hot-path ops (quantized matmul, FP16 matmul, RMSNorm, RoPE, SwiGLU, softmax, flash attention)
- MLX-compatible 4-bit quantization format (packed UInt32 weights + Float16 scales/biases, group_size=64)
- Simdgroup 8x8 matrix ops for tiled GEMM (4x4 SG grid, K x8 unrolling)
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

All benchmarks on M3 Max.

### End-to-end generation (Llama-3.2-3B-Instruct-4bit)

| | Julia | MLX | Julia/MLX |
|-|-------|-----|-----------|
| Decode throughput | 20 tok/s | 167 tok/s | 0.12x |
| TTFT (102 tokens) | ~0.5s | 0.08s | ~6x |

The e2e gap is larger than kernel-level due to per-dispatch overhead (~30us per `@metal` call x ~280 dispatches per token). See [#5] and [#10] for planned improvements.

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

Julia beats or matches MLX at B=4-128 for FP16 and B=8 for quantized. The remaining gap at larger batch sizes is from load instruction overhead in the compiled IR ([details](docs/metal-matmul-optimization.md)).

## Architecture

```
src/
  metal/                   # GPU kernels
    fp16_matmul.jl         #   FP16 simdgroup GEMM (ptr+vec2, vec2 Float16)
    quantized_matmul_sg.jl #   4-bit quantized simdgroup GEMM
    quantized_matmul_v2.jl #   4-bit scalar GEMM (best for B<=8)
    flash_attention.jl     #   Flash attention (online softmax)
    rmsnorm.jl             #   RMSNorm + fused residual
    rope.jl                #   RoPE with Llama3 frequency scaling
    swiglu.jl              #   SwiGLU activation
    softmax.jl             #   Numerically stable softmax
    fused_mlp.jl           #   Fused gate+up+SwiGLU
    fused_qkv.jl           #   Fused Q+K+V projection
    argmax.jl              #   GPU argmax
  model.jl                 # Model loading (safetensors, MLX quantization format)
  forward_optimized.jl     # Inference forward pass with fused kernels
  kvcache.jl               # Pre-allocated KV cache
  prefix_cache.jl          # Radix tree prefix cache
  server.jl                # HTTP server with SSE streaming
  tokenizer.jl             # Tokenizer (via Python transformers)
  gpucompiler_patch.jl     # GPUCompiler.jl trap elimination patch
  metal_simd_patch.jl      # Metal.jl mixed-precision MAC patch
```

## Roadmap

- [#5] — `threadgroup_async_copy` to close the ~1.7x kernel gap to MLX at large batch sizes
- [#9] — Qwen3.5-4B-Instruct-4bit support
- [#10] — PrecompileTools.jl for fast startup
- [#11] — LoRA adaptor support

## Upstream patches

Several patches to JuliaGPU packages are needed and included as monkey-patches:

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
