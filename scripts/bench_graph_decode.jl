"""Benchmark decode (seq=1) with KV cache via fused MPSGraph."""

using Metal
using Statistics

include(joinpath(@__DIR__, "..", "src", "metal_graph.jl"))
include(joinpath(@__DIR__, "..", "src", "graph_forward.jl"))
using .MetalGraphModule
using .GraphForward

const HIDDEN = 3072
const HEAD_DIM = 128
const N_Q = 24
const N_KV = 8
const INTER = 8192
const N_LAYERS = 28
const EPS = 1f-5
const MAX_LEN = 512

function bench_decode(cache_len::Int; n_warmup=30, n_iter=200)
    q_dim = N_Q * HEAD_DIM; kv_dim = N_KV * HEAD_DIM
    qkv_dim = q_dim + 2 * kv_dim; half_hd = HEAD_DIM ÷ 2

    println("\n=== Fused Decode (cache=$cache_len, max=$MAX_LEN) ===")
    println("Building graph...")
    t0 = time_ns()
    dec = build_decode_full(HIDDEN, HEAD_DIM, N_Q, N_KV, INTER, MAX_LEN, N_LAYERS, EPS)
    println("  Built in $(round((time_ns()-t0)/1e9, digits=2))s")

    # Build feeds
    feeds = Dict{GraphTensor, MtlArray}(
        dec.x_ph => MtlArray(randn(Float16, HIDDEN, 1)),
        dec.cos_ph => MtlArray(rand(Float16, half_hd, 1)),
        dec.sin_ph => MtlArray(rand(Float16, half_hd, 1)),
        dec.final_norm_w_ph => MtlArray(ones(Float16, HIDDEN, 1)),
    )

    # Attention mask: 0 for positions 1..cache_len, -1e4 for cache_len+1..max_len
    mask = fill(Float16(-1e4), 1, MAX_LEN)
    mask[1, 1:cache_len] .= Float16(0)
    feeds[dec.attn_mask_ph] = MtlArray(mask)

    for l in 1:N_LAYERS
        feeds[dec.input_norm_ws[l]] = MtlArray(ones(Float16, HIDDEN, 1))
        feeds[dec.w_qkvs[l]] = MtlArray(randn(Float16, qkv_dim, HIDDEN) * Float16(0.02))
        feeds[dec.w_os[l]] = MtlArray(randn(Float16, HIDDEN, q_dim) * Float16(0.02))
        # KV cache: random for positions 1..cache_len, zero for rest
        kc = zeros(Float16, HEAD_DIM, N_KV, MAX_LEN)
        kc[:, :, 1:cache_len] .= randn(Float16, HEAD_DIM, N_KV, cache_len) * Float16(0.02)
        feeds[dec.k_caches[l]] = MtlArray(kc)
        vc = zeros(Float16, HEAD_DIM, N_KV, MAX_LEN)
        vc[:, :, 1:cache_len] .= randn(Float16, HEAD_DIM, N_KV, cache_len) * Float16(0.02)
        feeds[dec.v_caches[l]] = MtlArray(vc)
        feeds[dec.post_norm_ws[l]] = MtlArray(ones(Float16, HIDDEN, 1))
        feeds[dec.w_gate_ups[l]] = MtlArray(randn(Float16, 2*INTER, HIDDEN) * Float16(0.02))
        feeds[dec.w_downs[l]] = MtlArray(randn(Float16, HIDDEN, INTER) * Float16(0.02))
    end

    out_buf = MtlArray(zeros(Float16, HIDDEN, 1))
    outputs = Dict{GraphTensor, MtlArray}(dec.out => out_buf)
    # Also allocate output buffers for new K/V (needed by execute_gpu!)
    for l in 1:N_LAYERS
        outputs[dec.new_ks[l]] = MtlArray(zeros(Float16, HEAD_DIM, N_KV, 1))
        outputs[dec.new_vs[l]] = MtlArray(zeros(Float16, HEAD_DIM, N_KV, 1))
    end

    println("Warming up ($n_warmup iterations)...")
    for _ in 1:n_warmup
        execute_gpu!(dec.compiled, feeds, outputs)
        Metal.synchronize()
    end

    println("Benchmarking ($n_iter iterations)...")
    times = Float64[]
    for _ in 1:n_iter
        t0 = time_ns()
        execute_gpu!(dec.compiled, feeds, outputs)
        Metal.synchronize()
        push!(times, (time_ns() - t0) / 1e6)
    end

    med = median(times)
    mn = minimum(times)
    println("  Median: $(round(med, digits=2)) ms = $(round(Int, 1000/med)) tok/s")
    println("  Min:    $(round(mn, digits=2)) ms")
    return med
end

println("MPSGraph Decode Benchmark (Llama-3.2-3B, 28 layers)")
println("Max KV cache length: $MAX_LEN")

results = Dict{Int, Float64}()
for cache_len in [1, 32, 128, 256, 512]
    results[cache_len] = bench_decode(cache_len)
end

println("\n" * "="^60)
println("Summary — Decode throughput (tok/s):")
println("| Cache Len | MPSGraph | MLX (est.) |")
println("|-----------|----------|------------|")
for cl in [1, 32, 128, 256, 512]
    tps = round(Int, 1000 / results[cl])
    println("|    $cl$(repeat(" ", 7-length(string(cl)))) | $tps tok/s | ~33 tok/s |")
end
