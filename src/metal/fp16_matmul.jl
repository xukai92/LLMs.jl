"""
FP16 tiled matmul using simdgroup matrix ops.

out[M, N] = W[M, K] @ x[K, N]

Strategy: each threadgroup has N_SG simdgroups sharing W rows.
- All simdgroups load the same W[8×8] tile
- Each simdgroup loads a different x[8×8] tile (different N columns)
- simdgroup_multiply_accumulate for the 8×8 inner matmul
- W reads reduced by N_SG factor

Grid: (M/8, N/(8*N_SG)) threadgroups, N_SG simdgroups per group.
"""

function fp16_matmul_kernel!(out, W, x, M::Int32, N::Int32, K::Int32)
    # Grid: (ceil(M/8), ceil(N/8))
    # One simdgroup per 8×8 output tile (simple, correct baseline)
    tile_m = Int32(threadgroup_position_in_grid().x)
    tile_n = Int32(threadgroup_position_in_grid().y)
    sg_lane = thread_index_in_simdgroup()

    m_base = (tile_m - Int32(1)) * Int32(8)
    n_base = (tile_n - Int32(1)) * Int32(8)

    if m_base >= M || n_base >= N
        return nothing
    end

    # Shared memory tiles
    w_tile = MtlThreadGroupArray(Float32, (8, 8))
    x_tile = MtlThreadGroupArray(Float32, (8, 8))
    result_tile = MtlThreadGroupArray(Float32, (8, 8))

    # Zero accumulator
    tid = Int32(sg_lane)
    if tid <= Int32(32)
        for elem in Int32(0):Int32(1)
            flat = (tid - Int32(1)) * Int32(2) + elem
            r = (flat % Int32(8)) + Int32(1)
            c = (flat ÷ Int32(8)) + Int32(1)
            if c <= Int32(8)
                @inbounds w_tile[r, c] = 0.0f0
            end
        end
    end
    simdgroup_barrier(Metal.MemoryFlagThreadGroup)
    acc = simdgroup_load(w_tile, (1, 1))

    # Process K in chunks of 8
    k = Int32(0)
    while k < K
        # Load W and x tiles (32 threads load 64 elements = 2 per thread)
        if tid <= Int32(32)
            for elem in Int32(0):Int32(1)
                flat = (tid - Int32(1)) * Int32(2) + elem
                r = (flat % Int32(8)) + Int32(1)
                c = (flat ÷ Int32(8)) + Int32(1)
                if c <= Int32(8)
                    gm = m_base + r; gk = k + c
                    @inbounds w_tile[r, c] = (gm <= M && gk <= K) ? Float32(W[gm, gk]) : 0.0f0
                    gk2 = k + r; gn = n_base + c
                    @inbounds x_tile[r, c] = (gk2 <= K && gn <= N) ? Float32(x[gk2, gn]) : 0.0f0
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
            flat = (tid - Int32(1)) * Int32(2) + elem
            r = (flat % Int32(8)) + Int32(1)
            c = (flat ÷ Int32(8)) + Int32(1)
            if c <= Int32(8)
                gm = m_base + r; gn = n_base + c
                if gm <= M && gn <= N
                    @inbounds out[gm, gn] = Float16(result_tile[r, c])
                end
            end
        end
    end
    return nothing
end

# Multi-simdgroup version: N_SG simdgroups share W rows
function fp16_matmul_nsg_kernel!(out, W, x, M::Int32, N::Int32, K::Int32)
    # Grid: (ceil(M/8), ceil(N/(8*N_SG)))
    # Each threadgroup has N_SG=4 simdgroups
    tile_m = Int32(threadgroup_position_in_grid().x)
    tile_n_group = Int32(threadgroup_position_in_grid().y)  # group of 4 N-tiles
    sg_idx = simdgroup_index_in_threadgroup()  # 1..4
    sg_lane = thread_index_in_simdgroup()
    tid_local = Int32(thread_index_in_threadgroup())

    NSG = Int32(4)
    m_base = (tile_m - Int32(1)) * Int32(8)
    n_base = (tile_n_group - Int32(1)) * NSG * Int32(8) + (Int32(sg_idx) - Int32(1)) * Int32(8)

    if m_base >= M || n_base >= N
        return nothing
    end

    # Shared memory: W tile shared by all simdgroups, x tiles per simdgroup
    w_shared = MtlThreadGroupArray(Float32, (8, 8))     # shared W tile
    x_sg1 = MtlThreadGroupArray(Float32, (8, 8))        # x tile for sg 1
    x_sg2 = MtlThreadGroupArray(Float32, (8, 8))        # x tile for sg 2
    x_sg3 = MtlThreadGroupArray(Float32, (8, 8))        # x tile for sg 3
    x_sg4 = MtlThreadGroupArray(Float32, (8, 8))        # x tile for sg 4
    res_sg1 = MtlThreadGroupArray(Float32, (8, 8))
    res_sg2 = MtlThreadGroupArray(Float32, (8, 8))
    res_sg3 = MtlThreadGroupArray(Float32, (8, 8))
    res_sg4 = MtlThreadGroupArray(Float32, (8, 8))

    # Get this simdgroup's x and result tiles
    my_x = sg_idx == UInt32(1) ? x_sg1 : sg_idx == UInt32(2) ? x_sg2 : sg_idx == UInt32(3) ? x_sg3 : x_sg4
    my_res = sg_idx == UInt32(1) ? res_sg1 : sg_idx == UInt32(2) ? res_sg2 : sg_idx == UInt32(3) ? res_sg3 : res_sg4

    # Zero accumulator
    tid = Int32(sg_lane)
    if tid <= Int32(32)
        for elem in Int32(0):Int32(1)
            flat = (tid - Int32(1)) * Int32(2) + elem
            r = (flat % Int32(8)) + Int32(1)
            c = (flat ÷ Int32(8)) + Int32(1)
            if c <= Int32(8); @inbounds my_res[r, c] = 0.0f0; end
        end
    end
    threadgroup_barrier(Metal.MemoryFlagThreadGroup)
    acc = simdgroup_load(my_res, (1, 1))

    k = Int32(0)
    while k < K
        # First simdgroup loads W tile (shared)
        if sg_idx == UInt32(1) && tid <= Int32(32)
            for elem in Int32(0):Int32(1)
                flat = (tid - Int32(1)) * Int32(2) + elem
                r = (flat % Int32(8)) + Int32(1)
                c = (flat ÷ Int32(8)) + Int32(1)
                if c <= Int32(8)
                    gm = m_base + r; gk = k + c
                    @inbounds w_shared[r, c] = (gm <= M && gk <= K) ? Float32(W[gm, gk]) : 0.0f0
                end
            end
        end

        # Each simdgroup loads its own x tile
        if tid <= Int32(32)
            for elem in Int32(0):Int32(1)
                flat = (tid - Int32(1)) * Int32(2) + elem
                r = (flat % Int32(8)) + Int32(1)
                c = (flat ÷ Int32(8)) + Int32(1)
                if c <= Int32(8)
                    gk = k + r; gn = n_base + c
                    @inbounds my_x[r, c] = (gk <= K && gn <= N) ? Float32(x[gk, gn]) : 0.0f0
                end
            end
        end
        threadgroup_barrier(Metal.MemoryFlagThreadGroup)

        w_mat = simdgroup_load(w_shared, (1, 1))
        x_mat = simdgroup_load(my_x, (1, 1))
        acc = simdgroup_multiply_accumulate(w_mat, x_mat, acc)
        threadgroup_barrier(Metal.MemoryFlagThreadGroup)

        k += Int32(8)
    end

    # Store
    simdgroup_store(acc, my_res, (1, 1))
    threadgroup_barrier(Metal.MemoryFlagThreadGroup)

    if tid <= Int32(32)
        for elem in Int32(0):Int32(1)
            flat = (tid - Int32(1)) * Int32(2) + elem
            r = (flat % Int32(8)) + Int32(1)
            c = (flat ÷ Int32(8)) + Int32(1)
            if c <= Int32(8)
                gm = m_base + r; gn = n_base + c
                if gm <= M && gn <= N
                    @inbounds out[gm, gn] = Float16(my_res[r, c])
                end
            end
        end
    end
    return nothing
end

"""FP16 matmul with automatic kernel selection."""
function metal_fp16_matmul!(out, W, x)
    M = Int32(size(W, 1)); K = Int32(size(W, 2)); N = Int32(size(x, 2))
    if N >= 32
        # Multi-simdgroup: 4 simdgroups share W, 128 threads
        NSG = 4
        @metal threads=(32*NSG) groups=(cld(Int(M), 8), cld(Int(N), 8*NSG)) fp16_matmul_nsg_kernel!(
            out, W, x, M, N, K)
    else
        # Single simdgroup per tile
        @metal threads=32 groups=(cld(Int(M), 8), cld(Int(N), 8)) fp16_matmul_kernel!(
            out, W, x, M, N, K)
    end
    return out
end
