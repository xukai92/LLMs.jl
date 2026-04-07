"""
FP16 matrix multiply using simdgroup matrix ops.

For pre-dequantized weights or as a reference implementation.
Uses simdgroup_multiply_accumulate for hardware 8×8 matrix multiply.

out[M, N] = W[M, K] @ x[K, N]
Each simdgroup computes one 8×8 output tile.
Iterates over K in chunks of 8, loading tiles from W and x.
"""

function fp16_simd_matmul_kernel!(out, W, x, M::Int32, N::Int32, K::Int32)
    # Grid: (M/8, N/8) — one simdgroup per 8×8 output tile
    tile_m = Int32(threadgroup_position_in_grid().x)
    tile_n = Int32(threadgroup_position_in_grid().y)

    m_base = (tile_m - Int32(1)) * Int32(8)
    n_base = (tile_n - Int32(1)) * Int32(8)

    if m_base + Int32(8) > M || n_base + Int32(8) > N
        return nothing
    end

    # Shared memory for tiles
    w_tile = MtlThreadGroupArray(Float32, (8, 8))
    x_tile = MtlThreadGroupArray(Float32, (8, 8))
    zero_tile = MtlThreadGroupArray(Float32, (8, 8))
    result_tile = MtlThreadGroupArray(Float32, (8, 8))

    tid = Int32(thread_index_in_simdgroup())

    # Zero the accumulator tile
    if tid <= Int32(32)
        for elem in Int32(0):Int32(1)
            flat = (Int32(tid) - Int32(1)) * Int32(2) + elem
            r = (flat % Int32(8)) + Int32(1)
            c = (flat ÷ Int32(8)) + Int32(1)
            if c <= Int32(8)
                @inbounds zero_tile[r, c] = 0.0f0
            end
        end
    end
    simdgroup_barrier(Metal.MemoryFlagThreadGroup)
    acc = simdgroup_load(zero_tile, (1, 1))

    # Iterate over K in chunks of 8
    k = Int32(0)
    while k < K
        # Load W tile: W[m_base+1:m_base+8, k+1:k+8]
        if tid <= Int32(32)
            for elem in Int32(0):Int32(1)
                flat = (Int32(tid) - Int32(1)) * Int32(2) + elem
                r = (flat % Int32(8)) + Int32(1)
                c = (flat ÷ Int32(8)) + Int32(1)
                if c <= Int32(8)
                    gm = m_base + r
                    gk = k + c
                    @inbounds w_tile[r, c] = gm <= M && gk <= K ? Float32(W[gm, gk]) : 0.0f0
                    @inbounds x_tile[r, c] = (k + r) <= K && (n_base + c) <= N ? Float32(x[k + r, n_base + c]) : 0.0f0
                end
            end
        end
        simdgroup_barrier(Metal.MemoryFlagThreadGroup)

        w_mat = simdgroup_load(w_tile, (1, 1))
        x_mat = simdgroup_load(x_tile, (1, 1))
        acc = simdgroup_multiply_accumulate(w_mat, x_mat, acc)

        simdgroup_barrier(Metal.MemoryFlagThreadGroup)
        k += Int32(8)
    end

    # Store result
    simdgroup_store(acc, result_tile, (1, 1))
    simdgroup_barrier(Metal.MemoryFlagThreadGroup)

    if tid <= Int32(32)
        for elem in Int32(0):Int32(1)
            flat = (Int32(tid) - Int32(1)) * Int32(2) + elem
            r = (flat % Int32(8)) + Int32(1)
            c = (flat ÷ Int32(8)) + Int32(1)
            if c <= Int32(8)
                gm = m_base + r
                gn = n_base + c
                if gm <= M && gn <= N
                    @inbounds out[gm, gn] = Float16(result_tile[r, c])
                end
            end
        end
    end
    return nothing
end

"""
FP16 matmul using simdgroup 8×8 hardware matrix multiply.
out[M, N] = W[M, K] @ x[K, N]
"""
function metal_fp16_matmul!(out, W, x)
    M = Int32(size(W, 1))
    K = Int32(size(W, 2))
    N = Int32(size(x, 2))

    @metal threads=32 groups=(cld(M, 8), cld(N, 8)) fp16_simd_matmul_kernel!(
        out, W, x, M, N, K)
    return out
end
