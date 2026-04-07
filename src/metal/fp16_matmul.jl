"""
FP16 matrix multiply using simdgroup 8×8 matrix ops.

out[M, N] = W[M, K] @ x[K, N]

Single simdgroup per 8×8 output tile. Each thread loads 2 elements
per tile (64 elements / 32 threads). K processed in chunks of 8.
"""

function fp16_matmul_kernel!(out, W, x, M::Int32, N::Int32, K::Int32)
    tile_m = Int32(threadgroup_position_in_grid().x)
    tile_n = Int32(threadgroup_position_in_grid().y)
    tid = Int32(thread_index_in_simdgroup())
    m_base = (tile_m - Int32(1)) * Int32(8)
    n_base = (tile_n - Int32(1)) * Int32(8)

    w = MtlThreadGroupArray(Float32, (8, 8))
    xt = MtlThreadGroupArray(Float32, (8, 8))

    # Zero accumulator
    f1 = tid - Int32(1)
    r1 = (f1 % Int32(8)) + Int32(1); c1 = (f1 ÷ Int32(8)) + Int32(1)
    @inbounds w[r1, c1] = 0f0
    f2 = tid + Int32(31)
    r2 = (f2 % Int32(8)) + Int32(1); c2 = (f2 ÷ Int32(8)) + Int32(1)
    if c2 <= Int32(8); @inbounds w[r2, c2] = 0f0; end
    simdgroup_barrier(Metal.MemoryFlagThreadGroup)
    acc = simdgroup_load(w, (1, 1))

    k = Int32(0)
    while k < K
        # Load W and x tiles — 2 elements per thread, no loop
        gm1 = m_base + r1; gk1 = k + c1
        @inbounds w[r1, c1] = (gm1 <= M && gk1 <= K) ? Float32(W[gm1, gk1]) : 0f0
        gkx1 = k + r1; gn1 = n_base + c1
        @inbounds xt[r1, c1] = (gkx1 <= K && gn1 <= N) ? Float32(x[gkx1, gn1]) : 0f0

        if c2 <= Int32(8)
            gm2 = m_base + r2; gk2 = k + c2
            @inbounds w[r2, c2] = (gm2 <= M && gk2 <= K) ? Float32(W[gm2, gk2]) : 0f0
            gkx2 = k + r2; gn2 = n_base + c2
            @inbounds xt[r2, c2] = (gkx2 <= K && gn2 <= N) ? Float32(x[gkx2, gn2]) : 0f0
        end

        simdgroup_barrier(Metal.MemoryFlagThreadGroup)
        acc = simdgroup_multiply_accumulate(simdgroup_load(w, (1, 1)), simdgroup_load(xt, (1, 1)), acc)
        simdgroup_barrier(Metal.MemoryFlagThreadGroup)
        k += Int32(8)
    end

    # Store
    simdgroup_store(acc, w, (1, 1))
    simdgroup_barrier(Metal.MemoryFlagThreadGroup)
    gm1 = m_base + r1; gn1 = n_base + c1
    if gm1 <= M && gn1 <= N; @inbounds out[gm1, gn1] = Float16(w[r1, c1]); end
    if c2 <= Int32(8)
        gm2 = m_base + r2; gn2 = n_base + c2
        if gm2 <= M && gn2 <= N; @inbounds out[gm2, gn2] = Float16(w[r2, c2]); end
    end
    return nothing
end

"""FP16 matmul using simdgroup 8×8 hardware matrix multiply."""
function metal_fp16_matmul!(out, W, x)
    M = Int32(size(W, 1)); K = Int32(size(W, 2)); N = Int32(size(x, 2))
    @metal threads=32 groups=(cld(Int(M), 8), cld(Int(N), 8)) fp16_matmul_kernel!(
        out, W, x, M, N, K)
    return out
end
