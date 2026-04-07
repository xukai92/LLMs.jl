"""
Kernel-level microbenchmarks for all Phase 1 kernels.

Measures wall time, effective memory bandwidth, and GFLOPS at
Llama-3.2-3B-Instruct-4bit dimensions.

Usage:
    julia --project=. scripts/benchmark_kernels.jl
"""

using LLMs
using Metal
using Printf
using Statistics

# ── Llama-3.2-3B dimensions ──
const HIDDEN = 3072
const INTERMEDIATE = 8192
const HEAD_DIM = 128
const N_Q_HEADS = 24
const N_KV_HEADS = 8
const VOCAB = 128256
const GROUP_SIZE = 64

# ── Benchmark helpers ──

function bench(f; warmup=10, iters=100)
    for _ in 1:warmup
        f()
        Metal.synchronize()
    end

    times = Float64[]
    for _ in 1:iters
        t0 = time_ns()
        f()
        Metal.synchronize()
        t1 = time_ns()
        push!(times, (t1 - t0) / 1e9)
    end

    med = median(times)
    mn = minimum(times)
    return (; median=med, min=mn, times)
end

struct BenchResult
    name::String
    median_ms::Float64
    min_ms::Float64
    bandwidth_gbs::Float64   # effective GB/s (bytes moved / min time)
    gflops::Float64          # if applicable, else 0
    bytes_moved::Int         # total bytes read + written
    flops::Int               # total FLOPs (0 if N/A)
end

function print_results(results::Vector{BenchResult})
    println("\n" * "="^100)
    println("KERNEL MICROBENCHMARKS — Llama-3.2-3B dimensions (M3 Max)")
    println("="^100)
    @printf("%-25s %10s %10s %12s %10s %12s\n",
            "Kernel", "Median(ms)", "Min(ms)", "BW(GB/s)", "GFLOPS", "Bytes")
    println("-"^100)

    total_median = 0.0
    for r in results
        @printf("%-25s %10.3f %10.3f %12.1f %10.1f %12s\n",
                r.name, r.median_ms, r.min_ms, r.bandwidth_gbs,
                r.gflops, format_bytes(r.bytes_moved))
        total_median += r.median_ms
    end
    println("-"^100)
    @printf("%-25s %10.3f\n", "TOTAL (per token est.)", total_median)
    println("="^100)

    # Breakdown
    println("\nTime breakdown (% of total):")
    for r in results
        pct = r.median_ms / total_median * 100
        bar = "█"^round(Int, pct / 2)
        @printf("  %-25s %5.1f%%  %s\n", r.name, pct, bar)
    end
end

format_bytes(b) = b > 1e9 ? @sprintf("%.1f GB", b/1e9) :
                  b > 1e6 ? @sprintf("%.1f MB", b/1e6) :
                  @sprintf("%.1f KB", b/1e3)

# ── Individual kernel benchmarks ──

function bench_rmsnorm(; batch=1)
    x = MtlArray(randn(Float16, HIDDEN, batch))
    w = MtlArray(rand(Float16, HIDDEN) .+ Float16(0.5))
    out = MtlArray(zeros(Float16, HIDDEN, batch))

    r = bench() do
        metal_rmsnorm!(out, x, w, 1f-5)
    end

    bytes = (HIDDEN * batch * 2 + HIDDEN * 2 + HIDDEN * batch * 2)  # read x + w, write out
    return BenchResult("RMSNorm", r.median*1e3, r.min*1e3,
                       bytes / r.min / 1e9, 0.0, bytes, 0)
end

function bench_rope(; seq_len=1)
    x = MtlArray(randn(Float16, HEAD_DIM, N_Q_HEADS, seq_len))
    cos_t, sin_t = compute_rope_freqs(HEAD_DIM, 4096)
    cos_m = MtlArray(cos_t)
    sin_m = MtlArray(sin_t)

    r = bench() do
        metal_rope!(x, cos_m, sin_m, 1)
    end

    half = HEAD_DIM ÷ 2
    bytes = (HEAD_DIM * N_Q_HEADS * seq_len * 2 * 2 +  # read+write x
             half * seq_len * 4 * 2)  # read cos+sin tables
    flops = HEAD_DIM * N_Q_HEADS * seq_len * 6  # 4 muls + 2 adds per pair
    return BenchResult("RoPE (Q+K, seq=$seq_len)", r.median*1e3, r.min*1e3,
                       bytes / r.min / 1e9, flops / r.min / 1e9, bytes, flops)
end

function bench_swiglu(; batch=1)
    gate = MtlArray(randn(Float16, INTERMEDIATE, batch))
    up = MtlArray(randn(Float16, INTERMEDIATE, batch))
    out = MtlArray(zeros(Float16, INTERMEDIATE, batch))

    r = bench() do
        metal_swiglu!(out, gate, up)
    end

    bytes = (INTERMEDIATE * batch * 2 * 2 + INTERMEDIATE * batch * 2)
    flops = INTERMEDIATE * batch * 5  # exp + div + mul + mul + add
    return BenchResult("SwiGLU", r.median*1e3, r.min*1e3,
                       bytes / r.min / 1e9, flops / r.min / 1e9, bytes, flops)
end

function bench_softmax(; seq_kv=128)
    x = MtlArray(randn(Float16, N_Q_HEADS, seq_kv))
    out = MtlArray(zeros(Float16, N_Q_HEADS, seq_kv))

    r = bench() do
        metal_softmax!(out, x)
    end

    bytes = (N_Q_HEADS * seq_kv * 2 * 2)  # read x, write out (3 passes but cached)
    return BenchResult("Softmax (seq=$seq_kv)", r.median*1e3, r.min*1e3,
                       bytes / r.min / 1e9, 0.0, bytes, 0)
end

function bench_attention(; seq_q=1, seq_kv=128)
    Q = MtlArray(randn(Float16, HEAD_DIM, N_Q_HEADS, seq_q))
    K = MtlArray(randn(Float16, HEAD_DIM, N_KV_HEADS, seq_kv))
    V = MtlArray(randn(Float16, HEAD_DIM, N_KV_HEADS, seq_kv))
    out = MtlArray(zeros(Float16, HEAD_DIM, N_Q_HEADS, seq_q))

    r = bench() do
        metal_attention!(out, Q, K, V, Float32(1.0 / sqrt(128.0));
                        causal=true, causal_offset=seq_kv - seq_q)
    end

    # FLOPs: Q@K^T = 2*seq_q*seq_kv*head_dim*n_q_heads, softmax ~ 5*seq_q*seq_kv*n_q_heads,
    # attn@V = 2*seq_q*seq_kv*head_dim*n_q_heads
    flops = N_Q_HEADS * seq_q * (2 * seq_kv * HEAD_DIM + 5 * seq_kv + 2 * seq_kv * HEAD_DIM)
    # Bytes: read Q, K, V, scores temp, write out
    bytes = ((HEAD_DIM * N_Q_HEADS * seq_q + HEAD_DIM * N_KV_HEADS * seq_kv * 2) * 2 +
             N_Q_HEADS * seq_q * seq_kv * 4 +  # scores buffer (Float32)
             HEAD_DIM * N_Q_HEADS * seq_q * 2)

    return BenchResult("Attention (q=$seq_q,kv=$seq_kv)",
                       r.median*1e3, r.min*1e3,
                       bytes / r.min / 1e9, flops / r.min / 1e9, bytes, flops)
end

function bench_quantized_matmul(; O=HIDDEN, I=HIDDEN, B=1, label="")
    packed_cols = I ÷ 8
    n_groups = I ÷ GROUP_SIZE
    packed = MtlArray(rand(UInt32, O, packed_cols))
    scales = MtlArray(rand(Float16, O, n_groups))
    biases = MtlArray(rand(Float16, O, n_groups))
    x = MtlArray(randn(Float16, I, B))
    out = MtlArray(zeros(Float16, O, B))

    r = bench() do
        metal_quantized_matmul!(out, x, packed, scales, biases; group_size=GROUP_SIZE)
    end

    # Bytes: read packed weights + scales + biases + x, write out
    weight_bytes = O * packed_cols * 4  # uint32
    scale_bias_bytes = O * n_groups * 2 * 2  # scales + biases, float16
    x_bytes = I * B * 2
    out_bytes = O * B * 2
    bytes = weight_bytes + scale_bias_bytes + x_bytes + out_bytes

    # FLOPs: O * I * B * 2 (mul + add per element) + dequant overhead
    flops = O * I * B * 2 + O * I * 2  # dequant: scale*val+bias per weight
    name = isempty(label) ? "QMatmul $(O)×$(I) B=$B" : label

    return BenchResult(name, r.median*1e3, r.min*1e3,
                       bytes / r.min / 1e9, flops / r.min / 1e9, bytes, flops)
end

# ── Per-layer estimate ──

function bench_single_layer_decode()
    println("\nBenchmarking individual kernels at Llama-3.2-3B decode dimensions (B=1)...\n")
    results = BenchResult[]

    # RMSNorm × 2 per layer
    r = bench_rmsnorm(batch=1)
    push!(results, BenchResult("RMSNorm ×2", r.median_ms*2, r.min_ms*2,
                                r.bandwidth_gbs, 0.0, r.bytes_moved*2, 0))

    # RoPE for Q and K
    push!(results, bench_rope(seq_len=1))

    # SwiGLU
    push!(results, bench_swiglu(batch=1))

    # Softmax (inside attention, seq_kv varies)
    push!(results, bench_softmax(seq_kv=128))

    # Attention (decode: q=1, kv=128 cached positions)
    push!(results, bench_attention(seq_q=1, seq_kv=128))

    # Quantized matmul projections per layer:
    #   q_proj: 3072→3072, k_proj: 3072→1024, v_proj: 3072→1024
    #   o_proj: 3072→3072
    #   gate: 3072→8192, up: 3072→8192, down: 8192→3072
    proj_configs = [
        (HIDDEN, HIDDEN, "q_proj 3072→3072"),
        (N_KV_HEADS * HEAD_DIM, HIDDEN, "k_proj 3072→1024"),
        (N_KV_HEADS * HEAD_DIM, HIDDEN, "v_proj 3072→1024"),
        (HIDDEN, HIDDEN, "o_proj 3072→3072"),
        (INTERMEDIATE, HIDDEN, "gate_proj 3072→8192"),
        (INTERMEDIATE, HIDDEN, "up_proj 3072→8192"),
        (HIDDEN, INTERMEDIATE, "down_proj 8192→3072"),
    ]

    total_matmul_ms = 0.0
    total_matmul_bytes = 0
    total_matmul_flops = 0

    for (O, I, label) in proj_configs
        r = bench_quantized_matmul(O=O, I=I, B=1, label=label)
        total_matmul_ms += r.median_ms
        total_matmul_bytes += r.bytes_moved
        total_matmul_flops += r.flops
    end

    # Summarize matmul as one entry for the overview
    push!(results, BenchResult("QMatmul ×7 (all projs)",
                                total_matmul_ms, total_matmul_ms * 0.95,
                                total_matmul_bytes / (total_matmul_ms/1e3) / 1e9,
                                total_matmul_flops / (total_matmul_ms/1e3) / 1e9,
                                total_matmul_bytes, total_matmul_flops))

    print_results(results)

    # Also print individual matmul breakdown
    println("\n\nQuantized matmul breakdown (per projection):")
    println("-"^80)
    @printf("%-30s %10s %10s %12s\n", "Projection", "Median(ms)", "BW(GB/s)", "Bytes")
    println("-"^80)
    for (O, I, label) in proj_configs
        r = bench_quantized_matmul(O=O, I=I, B=1)
        @printf("%-30s %10.3f %12.1f %12s\n", label, r.median_ms, r.bandwidth_gbs, format_bytes(r.bytes_moved))
    end

    return results
end

# ── End-to-end decode throughput ──

function bench_e2e_decode(model, tok; gen_lengths=[20, 50, 128])
    println("\n" * "="^80)
    println("END-TO-END DECODE BENCHMARK")
    println("="^80)

    prompt = "The quick brown fox jumps over the lazy dog. "^10  # ~100 tokens
    prompt_ids = encode(tok, prompt)
    println("Prompt: $(length(prompt_ids)) tokens\n")

    for gen_len in gen_lengths
        # Warmup
        cache = KVCache(model.config; max_seq_len=length(prompt_ids) + gen_len + 16)
        prompt_gpu = MtlArray(Int32.(prompt_ids))
        logits = forward(model, prompt_gpu, cache)
        Metal.synchronize()

        # Timed decode
        t0 = time()
        # Prefill
        cache2 = KVCache(model.config; max_seq_len=length(prompt_ids) + gen_len + 16)
        prompt_gpu2 = MtlArray(Int32.(prompt_ids))
        logits = forward(model, prompt_gpu2, cache2)
        Metal.synchronize()
        t_prefill = time() - t0

        # Decode
        t_decode_start = time()
        next_token = argmax_last_col_cpu(logits)
        for i in 2:gen_len
            token_gpu = MtlArray(Int32[next_token])
            logits = forward(model, token_gpu, cache2)
            Metal.synchronize()
            next_token = argmax_last_col_cpu(logits)
        end
        t_decode = time() - t_decode_start
        t_total = time() - t0

        ttft = t_prefill
        decode_tps = (gen_len - 1) / t_decode  # first token is from prefill
        overall_tps = gen_len / t_total

        @printf("  gen=%3d tokens: TTFT=%.3fs  decode=%.1f tok/s  overall=%.1f tok/s  (%.2fs total)\n",
                gen_len, ttft, decode_tps, overall_tps, t_total)
    end
end

function bench_e2e_ttft(model, tok; prompt_lengths=[16, 128, 512])
    println("\n" * "="^80)
    println("TTFT (Time To First Token) BENCHMARK")
    println("="^80)

    base = "Hello world. "^200  # long base text
    base_ids = encode(tok, base)

    for plen in prompt_lengths
        ids = base_ids[1:min(plen, length(base_ids))]

        # Warmup
        cache = KVCache(model.config; max_seq_len=length(ids) + 16)
        gpu_ids = MtlArray(Int32.(ids))
        forward(model, gpu_ids, cache)
        Metal.synchronize()

        # Timed
        t0 = time()
        cache2 = KVCache(model.config; max_seq_len=length(ids) + 16)
        gpu_ids2 = MtlArray(Int32.(ids))
        logits = forward(model, gpu_ids2, cache2)
        Metal.synchronize()
        _ = argmax_last_col_cpu(logits)
        ttft = time() - t0

        tps = length(ids) / ttft
        @printf("  prompt=%4d tokens: TTFT=%.3fs  (%.0f tok/s prefill)\n",
                length(ids), ttft, tps)
    end
end

# ── Main ──

function main()
    println("Metal device: $(Metal.device().name)")
    println()

    # Kernel microbenchmarks
    results = bench_single_layer_decode()

    # Estimate per-token time for full model (28 layers)
    total_per_layer = sum(r.median_ms for r in results)
    est_per_token = total_per_layer * 28
    println("\nEstimated per-token time (28 layers): $(round(est_per_token, digits=1)) ms")
    println("Estimated decode throughput: $(round(1000/est_per_token, digits=1)) tok/s")

    # End-to-end (only if model is available)
    model_dir = expanduser("~/.cache/huggingface/hub/models--mlx-community--Llama-3.2-3B-Instruct-4bit/snapshots/7f0dc925e0d0afb0322d96f9255cfddf2ba5636e")
    if isdir(model_dir)
        println("\nLoading model for end-to-end benchmarks...")
        model = load_llama_model(model_dir)
        tok = Tokenizer(model_dir)

        bench_e2e_ttft(model, tok)
        bench_e2e_decode(model, tok)
    else
        println("\nModel not found at $model_dir — skipping end-to-end benchmarks")
    end
end

main()
