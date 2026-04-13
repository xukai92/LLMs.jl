"""Benchmark MPSGraph transformer layer vs Julia kernels.

Tests single-layer and multi-layer (simulating full forward pass) performance.
Compares: MPSGraph fused graph vs Julia @metal dispatches.
"""

using Metal
using Statistics

include(joinpath(@__DIR__, "..", "src", "metal_graph.jl"))
include(joinpath(@__DIR__, "..", "src", "graph_forward.jl"))
using .MetalGraphModule
using .GraphForward

# Llama-3.2-3B dimensions
const HIDDEN = 3072
const HEAD_DIM = 128
const N_Q_HEADS = 24
const N_KV_HEADS = 8
const INTERMEDIATE = 8192
const N_LAYERS = 28
const EPS = 1f-5

function bench_graph_layer(seq_len::Int; n_warmup=20, n_iter=100)
    q_dim = N_Q_HEADS * HEAD_DIM
    kv_dim = N_KV_HEADS * HEAD_DIM
    qkv_dim = q_dim + 2 * kv_dim
    half_hd = HEAD_DIM ÷ 2

    println("\n=== MPSGraph Transformer Layer (seq=$seq_len) ===")
    println("Building graph...")

    layer = build_transformer_layer(HIDDEN, HEAD_DIM, N_Q_HEADS, N_KV_HEADS,
                                     INTERMEDIATE, seq_len, EPS; emit_normed=false)

    # Allocate GPU data
    x = MtlArray(randn(Float16, HIDDEN, seq_len))
    residual = MtlArray(randn(Float16, HIDDEN, seq_len))
    w_qkv = MtlArray(randn(Float16, qkv_dim, HIDDEN) * Float16(0.02))
    w_o = MtlArray(randn(Float16, HIDDEN, q_dim) * Float16(0.02))
    cos_t = MtlArray(rand(Float16, half_hd, seq_len))
    sin_t = MtlArray(rand(Float16, half_hd, seq_len))
    post_norm_w = MtlArray(ones(Float16, HIDDEN, 1))
    w_gu = MtlArray(randn(Float16, 2*INTERMEDIATE, HIDDEN) * Float16(0.02))
    w_down = MtlArray(randn(Float16, HIDDEN, INTERMEDIATE) * Float16(0.02))
    out_buf = MtlArray(zeros(Float16, HIDDEN, seq_len))

    feeds = Dict{GraphTensor, MtlArray}(
        layer.x_ph => x,
        layer.residual_ph => residual,
        layer.w_qkv_ph => w_qkv,
        layer.w_o_ph => w_o,
        layer.cos_ph => cos_t,
        layer.sin_ph => sin_t,
        layer.post_norm_w_ph => post_norm_w,
        layer.w_gate_up_ph => w_gu,
        layer.w_down_ph => w_down,
    )
    outputs = Dict{GraphTensor, MtlArray}(layer.out_residual => out_buf)

    # Warmup
    println("Warming up ($n_warmup iterations)...")
    for _ in 1:n_warmup
        execute_gpu!(layer.compiled, feeds, outputs)
    end
    Metal.synchronize()

    # Benchmark
    println("Benchmarking ($n_iter iterations)...")
    times = Float64[]
    for _ in 1:n_iter
        t0 = time_ns()
        execute_gpu!(layer.compiled, feeds, outputs)
        Metal.synchronize()
        push!(times, (time_ns() - t0) / 1e6)
    end

    med = median(times)
    mn = minimum(times)
    println("  Median: $(round(med, digits=3)) ms")
    println("  Min:    $(round(mn, digits=3)) ms")
    return med
end

function bench_multi_layer(seq_len::Int, n_layers::Int; n_warmup=10, n_iter=50)
    q_dim = N_Q_HEADS * HEAD_DIM
    kv_dim = N_KV_HEADS * HEAD_DIM
    qkv_dim = q_dim + 2 * kv_dim
    half_hd = HEAD_DIM ÷ 2

    println("\n=== MPSGraph $n_layers-Layer Forward Pass (seq=$seq_len) ===")
    println("Building graph...")

    layer = build_transformer_layer(HIDDEN, HEAD_DIM, N_Q_HEADS, N_KV_HEADS,
                                     INTERMEDIATE, seq_len, EPS; emit_normed=false)

    # Shared weights (same graph reused per layer, different weight feeds)
    w_qkv = MtlArray(randn(Float16, qkv_dim, HIDDEN) * Float16(0.02))
    w_o = MtlArray(randn(Float16, HIDDEN, q_dim) * Float16(0.02))
    cos_t = MtlArray(rand(Float16, half_hd, seq_len))
    sin_t = MtlArray(rand(Float16, half_hd, seq_len))
    norm_w = MtlArray(ones(Float16, HIDDEN, 1))
    w_gu = MtlArray(randn(Float16, 2*INTERMEDIATE, HIDDEN) * Float16(0.02))
    w_down = MtlArray(randn(Float16, HIDDEN, INTERMEDIATE) * Float16(0.02))

    buf_a = MtlArray(randn(Float16, HIDDEN, seq_len))
    buf_b = MtlArray(zeros(Float16, HIDDEN, seq_len))
    normed_buf = MtlArray(zeros(Float16, HIDDEN, seq_len))

    # For simplicity, we'll use the same graph instance and just swap buffers
    # In production, each layer would have its own weight feeds

    println("Warming up ($n_warmup iterations)...")
    for _ in 1:n_warmup
        residual = buf_a
        for l in 1:n_layers
            feeds = Dict{GraphTensor, MtlArray}(
                layer.x_ph => buf_a,  # normed input (simplified: skip rmsnorm for first layer)
                layer.residual_ph => residual,
                layer.w_qkv_ph => w_qkv,
                layer.w_o_ph => w_o,
                layer.cos_ph => cos_t,
                layer.sin_ph => sin_t,
                layer.post_norm_w_ph => norm_w,
                layer.w_gate_up_ph => w_gu,
                layer.w_down_ph => w_down,
            )
            outputs = Dict{GraphTensor, MtlArray}(layer.out_residual => buf_b)
            execute_gpu!(layer.compiled, feeds, outputs)
            buf_a, buf_b = buf_b, buf_a
        end
        Metal.synchronize()
    end

    println("Benchmarking ($n_iter iterations)...")
    times = Float64[]
    for _ in 1:n_iter
        t0 = time_ns()
        for l in 1:n_layers
            feeds = Dict{GraphTensor, MtlArray}(
                layer.x_ph => buf_a,
                layer.residual_ph => buf_a,
                layer.w_qkv_ph => w_qkv,
                layer.w_o_ph => w_o,
                layer.cos_ph => cos_t,
                layer.sin_ph => sin_t,
                layer.post_norm_w_ph => norm_w,
                layer.w_gate_up_ph => w_gu,
                layer.w_down_ph => w_down,
            )
            outputs = Dict{GraphTensor, MtlArray}(layer.out_residual => buf_b)
            execute_gpu!(layer.compiled, feeds, outputs)
            buf_a, buf_b = buf_b, buf_a
        end
        Metal.synchronize()
        push!(times, (time_ns() - t0) / 1e6)
    end

    med = median(times)
    mn = minimum(times)
    tok_per_s = seq_len / (med / 1000)
    println("  Median: $(round(med, digits=2)) ms ($(round(Int, tok_per_s)) tok/s)")
    println("  Min:    $(round(mn, digits=2)) ms")
    return med
end

# ── Run benchmarks ──
println("MPSGraph Transformer Layer Benchmark")
println("Model: Llama-3.2-3B equivalent dims")
println("Hidden=$HIDDEN, Heads=$N_Q_HEADS/$N_KV_HEADS, Inter=$INTERMEDIATE")

# Single layer at various seq lengths
for seq in [8, 16, 32, 64]
    bench_graph_layer(seq)
end

# Full 28-layer forward pass
println("\n" * "="^60)
for seq in [8, 32, 64]
    bench_multi_layer(seq, N_LAYERS)
end
