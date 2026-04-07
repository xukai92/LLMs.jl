# LLMs.jl — Julia LLM Inference Server for Apple Silicon

## Project overview
End-to-end LLM inference server in Julia with custom Metal GPU kernels,
targeting Apple Silicon (unified memory). OpenAI API-compatible.

## Architecture
- Custom Metal.jl kernels for all hot-path ops (quantized matmul, RMSNorm, RoPE, SwiGLU, softmax, attention)
- MLX quantization format: 4-bit packed uint32 weights + float16 scales/biases, group_size=64
- Radix-tree prefix caching (SGLang-style RadixAttention, not PagedAttention)
- Continuous batching scheduler
- Initially targeting Qwen3.5-4B-Instruct-4bit, then Llama-3.2-3B-Instruct-4bit

## Build & test
```bash
julia --project=. -e 'using Pkg; Pkg.test()'        # run all tests
julia --project=. test/test_safetensors.jl            # run specific test
julia --project=. scripts/inspect_weights.jl          # inspect model weights
julia --project=. scripts/bandwidth_bench.jl          # Metal bandwidth benchmark
```

## Conventions
- All Metal kernels live in `src/metal/`
- Each kernel has a CPU reference implementation for testing
- Tests compare Metal output vs CPU reference, assert relative error < 1e-3 for fp16
- Use `mlx_lm` (Python) as correctness oracle for end-to-end generation
- Float16 is the native compute type on Metal

## Key dependencies
- Metal.jl: GPU kernel programming via @metal macro
- KernelAbstractions.jl: vendor-agnostic kernel abstraction (fallback)
- JSON3.jl: JSON parsing (config files, safetensors headers)
- HTTP.jl: HTTP server for OpenAI-compatible API
