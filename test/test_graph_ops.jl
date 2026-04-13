"""Test MPSGraph ops against CPU references."""

using Metal
using Test

# Load our module
include(joinpath(@__DIR__, "..", "src", "metal_graph.jl"))
using .MetalGraphModule

function test_rmsnorm()
    @testset "rmsnorm" begin
        M, N = 128, 16
        x_data = randn(Float16, M, N)
        gamma_data = rand(Float16, M) .+ Float16(0.5)

        # CPU reference
        x32 = Float32.(x_data)
        ms = sum(x32 .^ 2, dims=1) ./ M
        inv = 1.0f0 ./ sqrt.(ms .+ 1f-5)
        ref = Float16.(x32 .* inv .* Float32.(gamma_data))

        # MPSGraph
        g = MetalGraphBuilder()
        x = placeholder!(g, (M, N), Float16)
        gam = placeholder!(g, (M, 1), Float16)
        out = rmsnorm!(g, x, gam, 1f-5)
        cg = compile!(g, [out])

        x_gpu = MtlArray(x_data)
        gam_gpu = MtlArray(reshape(gamma_data, M, 1))
        result = execute!(cg, Dict(x => x_gpu, gam => gam_gpu))

        gpu_out = result[out]
        @test size(gpu_out) == (M, N)
        err = maximum(abs.(Float32.(gpu_out) .- Float32.(ref))) / maximum(abs.(Float32.(ref)))
        @test err < 0.02  # FP16 tolerance
        println("  rmsnorm: max relative error = $(round(err, digits=6))")
    end
end

function test_softmax()
    @testset "softmax" begin
        M, N = 64, 32
        x_data = randn(Float16, M, N)

        # CPU reference: softmax along dim 1
        x32 = Float32.(x_data)
        mx = maximum(x32, dims=1)
        ex = exp.(x32 .- mx)
        ref = Float16.(ex ./ sum(ex, dims=1))

        # MPSGraph
        g = MetalGraphBuilder()
        x = placeholder!(g, (M, N), Float16)
        out = softmax!(g, x, 1)
        cg = compile!(g, [out])

        result = execute!(cg, Dict(x => MtlArray(x_data)))
        gpu_out = result[out]
        err = maximum(abs.(Float32.(gpu_out) .- Float32.(ref)))
        @test err < 0.01
        println("  softmax dim=1: max abs error = $(round(err, digits=6))")

        # softmax along dim 2
        mx2 = maximum(x32, dims=2)
        ex2 = exp.(x32 .- mx2)
        ref2 = Float16.(ex2 ./ sum(ex2, dims=2))

        g2 = MetalGraphBuilder()
        x2 = placeholder!(g2, (M, N), Float16)
        out2 = softmax!(g2, x2, 2)
        cg2 = compile!(g2, [out2])
        result2 = execute!(cg2, Dict(x2 => MtlArray(x_data)))
        gpu_out2 = result2[out2]
        err2 = maximum(abs.(Float32.(gpu_out2) .- Float32.(ref2)))
        @test err2 < 0.01
        println("  softmax dim=2: max abs error = $(round(err2, digits=6))")
    end
end

function test_slice_concat()
    @testset "slice + concat" begin
        M, N = 128, 16
        x_data = randn(Float16, M, N)

        # Slice along dim 1: first half and second half
        half = M ÷ 2

        g = MetalGraphBuilder()
        x = placeholder!(g, (M, N), Float16)
        lo = slice!(g, x, 1, 0, half)    # MPSGraph uses 0-based start
        hi = slice!(g, x, 1, half, half)
        # Concat back
        rejoined = concat!(g, [lo, hi], 1)
        cg = compile!(g, [lo, hi, rejoined])

        result = execute!(cg, Dict(x => MtlArray(x_data)))
        @test result[lo] ≈ x_data[1:half, :]
        @test result[hi] ≈ x_data[half+1:end, :]
        @test result[rejoined] ≈ x_data
        println("  slice+concat dim=1: passed")

        # Slice along dim 2
        g2 = MetalGraphBuilder()
        x2 = placeholder!(g2, (M, N), Float16)
        s1 = slice!(g2, x2, 2, 0, 8)
        s2 = slice!(g2, x2, 2, 8, 8)
        r2 = concat!(g2, [s1, s2], 2)
        cg2 = compile!(g2, [s1, s2, r2])
        result2 = execute!(cg2, Dict(x2 => MtlArray(x_data)))
        @test result2[s1] ≈ x_data[:, 1:8]
        @test result2[s2] ≈ x_data[:, 9:16]
        @test result2[r2] ≈ x_data
        println("  slice+concat dim=2: passed")
    end
end

function test_transpose()
    @testset "transpose" begin
        D, H, S = 16, 8, 4
        x_data = randn(Float16, D, H, S)

        # Transpose dims 1 and 2: (D,H,S) → (H,D,S)
        g = MetalGraphBuilder()
        x = placeholder!(g, (D, H, S), Float16)
        out = graph_transpose!(g, x, 1, 2)
        cg = compile!(g, [out])
        result = execute!(cg, Dict(x => MtlArray(x_data)))
        @test size(result[out]) == (H, D, S)
        @test result[out] ≈ permutedims(x_data, (2, 1, 3))
        println("  transpose (1,2) on 3D: passed")

        # Transpose dims 2 and 3: (D,H,S) → (D,S,H)
        g2 = MetalGraphBuilder()
        x2 = placeholder!(g2, (D, H, S), Float16)
        out2 = graph_transpose!(g2, x2, 2, 3)
        cg2 = compile!(g2, [out2])
        result2 = execute!(cg2, Dict(x2 => MtlArray(x_data)))
        @test size(result2[out2]) == (D, S, H)
        @test result2[out2] ≈ permutedims(x_data, (1, 3, 2))
        println("  transpose (2,3) on 3D: passed")
    end
end

function test_rope()
    @testset "rope (graph ops)" begin
        hd, seq = 128, 16
        half = hd ÷ 2

        x_data = randn(Float16, hd, seq)
        cos_data = rand(Float16, half, seq)
        sin_data = rand(Float16, half, seq)

        # CPU reference: interleaved rope
        # x_lo = x[1:half, :], x_hi = x[half+1:end, :]
        # out_lo = x_lo * cos - x_hi * sin
        # out_hi = x_hi * cos + x_lo * sin
        x32 = Float32.(x_data)
        c32 = Float32.(cos_data); s32 = Float32.(sin_data)
        ref_lo = x32[1:half, :] .* c32 .- x32[half+1:end, :] .* s32
        ref_hi = x32[half+1:end, :] .* c32 .+ x32[1:half, :] .* s32
        ref = Float16.(vcat(ref_lo, ref_hi))

        # MPSGraph
        g = MetalGraphBuilder()
        x = placeholder!(g, (hd, seq), Float16)
        cos_t = placeholder!(g, (half, seq), Float16)
        sin_t = placeholder!(g, (half, seq), Float16)

        x_lo = slice!(g, x, 1, 0, half)
        x_hi = slice!(g, x, 1, half, half)

        # out_lo = x_lo * cos - x_hi * sin
        out_lo = sub!(g, mul!(g, x_lo, cos_t), mul!(g, x_hi, sin_t))
        # out_hi = x_hi * cos + x_lo * sin
        out_hi = add!(g, mul!(g, x_hi, cos_t), mul!(g, x_lo, sin_t))
        out = concat!(g, [out_lo, out_hi], 1)

        cg = compile!(g, [out])
        result = execute!(cg, Dict(
            x => MtlArray(x_data),
            cos_t => MtlArray(cos_data),
            sin_t => MtlArray(sin_data),
        ))

        gpu_out = result[out]
        err = maximum(abs.(Float32.(gpu_out) .- Float32.(ref))) / maximum(abs.(Float32.(ref)))
        @test err < 0.01
        println("  rope: max relative error = $(round(err, digits=6))")
    end
end

function test_matmul_chain()
    @testset "matmul chain (MLP-like)" begin
        D, I, N = 128, 256, 16
        x_data = randn(Float16, D, N)
        w_gate = randn(Float16, I, D)
        w_up = randn(Float16, I, D)
        w_down = randn(Float16, D, I)

        # CPU reference: down(silu(gate(x)) * up(x))
        x32 = Float32.(x_data)
        gate_out = Float32.(w_gate) * x32
        up_out = Float32.(w_up) * x32
        silu_gate = gate_out .* (1 ./ (1 .+ exp.(-gate_out)))
        mlp_out = Float16.(Float32.(w_down) * Float32.(silu_gate .* up_out))

        # MPSGraph: fused gate+up, slice, silu, mul, down
        g = MetalGraphBuilder()
        x = placeholder!(g, (D, N), Float16)
        wg = placeholder!(g, (I, D), Float16)
        wu = placeholder!(g, (I, D), Float16)
        wd = placeholder!(g, (D, I), Float16)

        gate = matmul!(g, wg, x)
        up = matmul!(g, wu, x)
        swi = silu!(g, gate)
        fused = mul!(g, swi, up)
        out = matmul!(g, wd, fused)

        cg = compile!(g, [out])
        result = execute!(cg, Dict(
            x => MtlArray(x_data),
            wg => MtlArray(w_gate),
            wu => MtlArray(w_up),
            wd => MtlArray(w_down),
        ))

        gpu_out = result[out]
        err = maximum(abs.(Float32.(gpu_out) .- Float32.(mlp_out))) / (maximum(abs.(Float32.(mlp_out))) + 1f-8)
        @test err < 0.05  # longer chain accumulates FP16 error
        println("  MLP chain: max relative error = $(round(err, digits=4))")
    end
end

# Run all tests
println("Testing MPSGraph ops...")
test_rmsnorm()
test_softmax()
test_slice_concat()
test_transpose()
test_rope()
test_matmul_chain()
println("\nAll MPSGraph op tests passed!")
