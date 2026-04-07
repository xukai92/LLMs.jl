"""
FP16 tiled matrix multiply using simdgroup 8×8 matrix ops.

out[M, N] = W[M, K] @ x[K, N]

Three kernel variants, auto-selected by batch size:
- 1-SG (B<16): one simdgroup per 8×8 output tile, 32 threads
- 2×2 (B≥16): four simdgroups in 2×2 grid per 16×16 output tile, 128 threads
  Shares W across N and x across M via shared memory
- (4×4 available but slower than 2×2 at these sizes)
"""

# 1-SG: 8×8 output tile, one simdgroup (32 threads)
function fp16_matmul_1sg!(out, W, x, M::Int32, N::Int32, K::Int32)
    tile_m = Int32(threadgroup_position_in_grid().x)
    tile_n = Int32(threadgroup_position_in_grid().y)
    tid = Int32(thread_index_in_simdgroup())
    m_base = (tile_m - Int32(1)) * Int32(8)
    n_base = (tile_n - Int32(1)) * Int32(8)
    w = MtlThreadGroupArray(Float32, (8, 8))
    xt = MtlThreadGroupArray(Float32, (8, 8))
    f1 = tid - Int32(1); r1 = (f1 % Int32(8)) + Int32(1); c1 = (f1 ÷ Int32(8)) + Int32(1)
    @inbounds w[r1, c1] = 0f0
    f2 = tid + Int32(31); r2 = (f2 % Int32(8)) + Int32(1); c2 = (f2 ÷ Int32(8)) + Int32(1)
    if c2 <= Int32(8); @inbounds w[r2, c2] = 0f0; end
    simdgroup_barrier(Metal.MemoryFlagThreadGroup)
    acc = simdgroup_load(w, (1, 1))
    k = Int32(0)
    while k < K
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

# 2×2 SG: 16×16 output tile, four simdgroups in 2×2 grid (128 threads)
# W (16×8) shared across N, x (8×16) shared across M
function fp16_matmul_2x2!(out, W, x, M::Int32, N::Int32, K::Int32)
    tile_m_grp = Int32(threadgroup_position_in_grid().x)
    tile_n_grp = Int32(threadgroup_position_in_grid().y)
    sg = Int32(simdgroup_index_in_threadgroup())
    tid = Int32(thread_index_in_simdgroup())
    gtid = (sg - Int32(1)) * Int32(32) + tid
    sg_m = (sg - Int32(1)) % Int32(2)
    sg_n = (sg - Int32(1)) ÷ Int32(2)

    w_shared = MtlThreadGroupArray(Float32, (16, 8))
    x_shared = MtlThreadGroupArray(Float32, (8, 16))
    zero_tile = MtlThreadGroupArray(Float32, (8, 8))
    res = MtlThreadGroupArray(Float32, (16, 16))

    if sg == Int32(1) && tid <= Int32(32)
        for e in Int32(0):Int32(1)
            f = (tid - Int32(1)) * Int32(2) + e
            r = (f % Int32(8)) + Int32(1); c = (f ÷ Int32(8)) + Int32(1)
            if c <= Int32(8); @inbounds zero_tile[r, c] = 0f0; end
        end
    end
    threadgroup_barrier(Metal.MemoryFlagThreadGroup)
    acc = simdgroup_load(zero_tile, (1, 1))

    k = Int32(0)
    while k < K
        # Cooperative load: 128 threads, W=128 elems, x=128 elems
        if gtid <= Int32(128)
            f = gtid - Int32(1)
            r = (f % Int32(16)) + Int32(1); c = (f ÷ Int32(16)) + Int32(1)
            gm = (tile_m_grp - Int32(1)) * Int32(16) + r; gk = k + c
            @inbounds w_shared[r, c] = (gm <= M && gk <= K) ? Float32(W[gm, gk]) : 0f0
        end
        if gtid <= Int32(128)
            f = gtid - Int32(1)
            r = (f % Int32(8)) + Int32(1); c = (f ÷ Int32(8)) + Int32(1)
            gk = k + r; gn = (tile_n_grp - Int32(1)) * Int32(16) + c
            @inbounds x_shared[r, c] = (gk <= K && gn <= N) ? Float32(x[gk, gn]) : 0f0
        end
        threadgroup_barrier(Metal.MemoryFlagThreadGroup)

        w_mat = simdgroup_load(w_shared, (Int64(sg_m) * Int64(8) + Int64(1), Int64(1)))
        x_mat = simdgroup_load(x_shared, (Int64(1), Int64(sg_n) * Int64(8) + Int64(1)))
        acc = simdgroup_multiply_accumulate(w_mat, x_mat, acc)
        threadgroup_barrier(Metal.MemoryFlagThreadGroup)
        k += Int32(8)
    end

    simdgroup_store(acc, res, (Int64(sg_m) * Int64(8) + Int64(1), Int64(sg_n) * Int64(8) + Int64(1)))
    threadgroup_barrier(Metal.MemoryFlagThreadGroup)

    for p in Int32(0):Int32(1)
        idx = (gtid - Int32(1)) * Int32(2) + p + Int32(1)
        if idx <= Int32(256)
            r = ((idx - Int32(1)) % Int32(16)) + Int32(1)
            c = ((idx - Int32(1)) ÷ Int32(16)) + Int32(1)
            gm = (tile_m_grp - Int32(1)) * Int32(16) + r
            gn = (tile_n_grp - Int32(1)) * Int32(16) + c
            if gm <= M && gn <= N
                @inbounds out[gm, gn] = Float16(res[r, c])
            end
        end
    end
    return nothing
end

"""FP16 matmul with auto kernel selection."""
function metal_fp16_matmul!(out, W, x)
    M = Int32(size(W, 1)); K = Int32(size(W, 2)); N = Int32(size(x, 2))
    if N >= 16
        @metal threads=128 groups=(cld(Int(M), 16), cld(Int(N), 16)) fp16_matmul_2x2!(
            out, W, x, M, N, K)
    else
        @metal threads=32 groups=(cld(Int(M), 8), cld(Int(N), 8)) fp16_matmul_1sg!(
            out, W, x, M, N, K)
    end
    return out
end
