"""
FP16 tiled matrix multiply using simdgroup 8×8 matrix ops.

out[M, N] = W[M, K] @ x[K, N]

Auto-selects from three kernel variants by batch size:
- 1-SG: B<16, one simdgroup per 8×8 tile (32 threads)
- 2×2 K×2: B=16, four SGs in 2×2 grid, 16×16 tile, K-unrolled ×2 (128 threads)
- 2×2 K×4: B≥32, same tiling, K-unrolled ×4 for fewer barriers (128 threads)
"""

# ── 1-SG kernel (8×8 tile, 32 threads) ──
function fp16_matmul_1sg!(out, W, x, M::Int32, N::Int32, K::Int32)
    tile_m = Int32(threadgroup_position_in_grid().x)
    tile_n = Int32(threadgroup_position_in_grid().y)
    tid = Int32(thread_index_in_simdgroup())
    m_base = (tile_m - Int32(1)) * Int32(8); n_base = (tile_n - Int32(1)) * Int32(8)
    w = MtlThreadGroupArray(Float32, (8, 8)); xt = MtlThreadGroupArray(Float32, (8, 8))
    f1 = tid - Int32(1); r1 = (f1 % Int32(8)) + Int32(1); c1 = (f1 ÷ Int32(8)) + Int32(1)
    @inbounds w[r1, c1] = 0f0
    f2 = tid + Int32(31); r2 = (f2 % Int32(8)) + Int32(1); c2 = (f2 ÷ Int32(8)) + Int32(1)
    if c2 <= Int32(8); @inbounds w[r2, c2] = 0f0; end
    simdgroup_barrier(Metal.MemoryFlagThreadGroup)
    acc = simdgroup_load(w, (1, 1)); k = Int32(0)
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
        simdgroup_barrier(Metal.MemoryFlagThreadGroup); k += Int32(8)
    end
    simdgroup_store(acc, w, (1, 1)); simdgroup_barrier(Metal.MemoryFlagThreadGroup)
    gm1 = m_base + r1; gn1 = n_base + c1
    if gm1 <= M && gn1 <= N; @inbounds out[gm1, gn1] = Float16(w[r1, c1]); end
    if c2 <= Int32(8); gm2 = m_base + r2; gn2 = n_base + c2
        if gm2 <= M && gn2 <= N; @inbounds out[gm2, gn2] = Float16(w[r2, c2]); end
    end; return nothing
end

# ── Shared load helpers for 2×2 kernels ──
# (defined as inner @inline functions in each kernel)

# ── 2×2 K×2 kernel (16×16 tile, 128 threads, 2 K-chunks per barrier) ──
function fp16_matmul_2x2_k2!(out, W, x, M::Int32, N::Int32, K::Int32)
    tile_m_grp = Int32(threadgroup_position_in_grid().x)
    tile_n_grp = Int32(threadgroup_position_in_grid().y)
    sg = Int32(simdgroup_index_in_threadgroup()); tid = Int32(thread_index_in_simdgroup())
    gtid = (sg - Int32(1)) * Int32(32) + tid
    sg_m = (sg - Int32(1)) % Int32(2); sg_n = (sg - Int32(1)) ÷ Int32(2)
    w1 = MtlThreadGroupArray(Float32, (16, 8)); x1 = MtlThreadGroupArray(Float32, (8, 16))
    w2 = MtlThreadGroupArray(Float32, (16, 8)); x2 = MtlThreadGroupArray(Float32, (8, 16))
    zero_tile = MtlThreadGroupArray(Float32, (8, 8)); res = MtlThreadGroupArray(Float32, (16, 16))
    if sg == Int32(1) && tid <= Int32(32)
        for e in Int32(0):Int32(1)
            f = (tid - Int32(1)) * Int32(2) + e; r = (f % Int32(8)) + Int32(1); c = (f ÷ Int32(8)) + Int32(1)
            if c <= Int32(8); @inbounds zero_tile[r, c] = 0f0; end
        end
    end
    threadgroup_barrier(Metal.MemoryFlagThreadGroup); acc = simdgroup_load(zero_tile, (1, 1))
    @inline function _lw(dst, ko)
        if gtid <= Int32(128)
            f = gtid - Int32(1); r = (f % Int32(16)) + Int32(1); c = (f ÷ Int32(16)) + Int32(1)
            gm = (tile_m_grp - Int32(1)) * Int32(16) + r; gk = ko + c
            @inbounds dst[r, c] = (gm <= M && gk <= K) ? Float32(W[gm, gk]) : 0f0
        end
    end
    @inline function _lx(dst, ko)
        if gtid <= Int32(128)
            f = gtid - Int32(1); r = (f % Int32(8)) + Int32(1); c = (f ÷ Int32(8)) + Int32(1)
            gk = ko + r; gn = (tile_n_grp - Int32(1)) * Int32(16) + c
            @inbounds dst[r, c] = (gk <= K && gn <= N) ? Float32(x[gk, gn]) : 0f0
        end
    end
    w_or = (Int64(sg_m) * Int64(8) + Int64(1), Int64(1))
    x_or = (Int64(1), Int64(sg_n) * Int64(8) + Int64(1))
    k = Int32(0)
    while k + Int32(16) <= K
        _lw(w1, k); _lx(x1, k); _lw(w2, k + Int32(8)); _lx(x2, k + Int32(8))
        threadgroup_barrier(Metal.MemoryFlagThreadGroup)
        acc = simdgroup_multiply_accumulate(simdgroup_load(w1, w_or), simdgroup_load(x1, x_or), acc)
        acc = simdgroup_multiply_accumulate(simdgroup_load(w2, w_or), simdgroup_load(x2, x_or), acc)
        threadgroup_barrier(Metal.MemoryFlagThreadGroup); k += Int32(16)
    end
    while k < K
        _lw(w1, k); _lx(x1, k); threadgroup_barrier(Metal.MemoryFlagThreadGroup)
        acc = simdgroup_multiply_accumulate(simdgroup_load(w1, w_or), simdgroup_load(x1, x_or), acc)
        threadgroup_barrier(Metal.MemoryFlagThreadGroup); k += Int32(8)
    end
    simdgroup_store(acc, res, (Int64(sg_m) * Int64(8) + Int64(1), Int64(sg_n) * Int64(8) + Int64(1)))
    threadgroup_barrier(Metal.MemoryFlagThreadGroup)
    for p in Int32(0):Int32(1)
        idx = (gtid - Int32(1)) * Int32(2) + p + Int32(1)
        if idx <= Int32(256)
            r = ((idx - Int32(1)) % Int32(16)) + Int32(1); c = ((idx - Int32(1)) ÷ Int32(16)) + Int32(1)
            gm = (tile_m_grp - Int32(1)) * Int32(16) + r; gn = (tile_n_grp - Int32(1)) * Int32(16) + c
            if gm <= M && gn <= N; @inbounds out[gm, gn] = Float16(res[r, c]); end
        end
    end; return nothing
end

# ── 2×2 K×4 kernel (16×16 tile, 128 threads, 4 K-chunks per barrier) ──
function fp16_matmul_2x2_k4!(out, W, x, M::Int32, N::Int32, K::Int32)
    tile_m_grp = Int32(threadgroup_position_in_grid().x)
    tile_n_grp = Int32(threadgroup_position_in_grid().y)
    sg = Int32(simdgroup_index_in_threadgroup()); tid = Int32(thread_index_in_simdgroup())
    gtid = (sg - Int32(1)) * Int32(32) + tid
    sg_m = (sg - Int32(1)) % Int32(2); sg_n = (sg - Int32(1)) ÷ Int32(2)
    w1 = MtlThreadGroupArray(Float32, (16, 8)); x1 = MtlThreadGroupArray(Float32, (8, 16))
    w2 = MtlThreadGroupArray(Float32, (16, 8)); x2 = MtlThreadGroupArray(Float32, (8, 16))
    w3 = MtlThreadGroupArray(Float32, (16, 8)); x3 = MtlThreadGroupArray(Float32, (8, 16))
    w4 = MtlThreadGroupArray(Float32, (16, 8)); x4 = MtlThreadGroupArray(Float32, (8, 16))
    zero_tile = MtlThreadGroupArray(Float32, (8, 8)); res = MtlThreadGroupArray(Float32, (16, 16))
    if sg == Int32(1) && tid <= Int32(32)
        for e in Int32(0):Int32(1)
            f = (tid - Int32(1)) * Int32(2) + e; r = (f % Int32(8)) + Int32(1); c = (f ÷ Int32(8)) + Int32(1)
            if c <= Int32(8); @inbounds zero_tile[r, c] = 0f0; end
        end
    end
    threadgroup_barrier(Metal.MemoryFlagThreadGroup); acc = simdgroup_load(zero_tile, (1, 1))
    @inline function _lw(dst, ko)
        if gtid <= Int32(128)
            f = gtid - Int32(1); r = (f % Int32(16)) + Int32(1); c = (f ÷ Int32(16)) + Int32(1)
            gm = (tile_m_grp - Int32(1)) * Int32(16) + r; gk = ko + c
            @inbounds dst[r, c] = (gm <= M && gk <= K) ? Float32(W[gm, gk]) : 0f0
        end
    end
    @inline function _lx(dst, ko)
        if gtid <= Int32(128)
            f = gtid - Int32(1); r = (f % Int32(8)) + Int32(1); c = (f ÷ Int32(8)) + Int32(1)
            gk = ko + r; gn = (tile_n_grp - Int32(1)) * Int32(16) + c
            @inbounds dst[r, c] = (gk <= K && gn <= N) ? Float32(x[gk, gn]) : 0f0
        end
    end
    w_or = (Int64(sg_m) * Int64(8) + Int64(1), Int64(1))
    x_or = (Int64(1), Int64(sg_n) * Int64(8) + Int64(1))
    k = Int32(0)
    while k + Int32(32) <= K
        _lw(w1, k); _lx(x1, k); _lw(w2, k + Int32(8)); _lx(x2, k + Int32(8))
        _lw(w3, k + Int32(16)); _lx(x3, k + Int32(16)); _lw(w4, k + Int32(24)); _lx(x4, k + Int32(24))
        threadgroup_barrier(Metal.MemoryFlagThreadGroup)
        acc = simdgroup_multiply_accumulate(simdgroup_load(w1, w_or), simdgroup_load(x1, x_or), acc)
        acc = simdgroup_multiply_accumulate(simdgroup_load(w2, w_or), simdgroup_load(x2, x_or), acc)
        acc = simdgroup_multiply_accumulate(simdgroup_load(w3, w_or), simdgroup_load(x3, x_or), acc)
        acc = simdgroup_multiply_accumulate(simdgroup_load(w4, w_or), simdgroup_load(x4, x_or), acc)
        threadgroup_barrier(Metal.MemoryFlagThreadGroup); k += Int32(32)
    end
    while k < K
        _lw(w1, k); _lx(x1, k); threadgroup_barrier(Metal.MemoryFlagThreadGroup)
        acc = simdgroup_multiply_accumulate(simdgroup_load(w1, w_or), simdgroup_load(x1, x_or), acc)
        threadgroup_barrier(Metal.MemoryFlagThreadGroup); k += Int32(8)
    end
    simdgroup_store(acc, res, (Int64(sg_m) * Int64(8) + Int64(1), Int64(sg_n) * Int64(8) + Int64(1)))
    threadgroup_barrier(Metal.MemoryFlagThreadGroup)
    for p in Int32(0):Int32(1)
        idx = (gtid - Int32(1)) * Int32(2) + p + Int32(1)
        if idx <= Int32(256)
            r = ((idx - Int32(1)) % Int32(16)) + Int32(1); c = ((idx - Int32(1)) ÷ Int32(16)) + Int32(1)
            gm = (tile_m_grp - Int32(1)) * Int32(16) + r; gn = (tile_n_grp - Int32(1)) * Int32(16) + c
            if gm <= M && gn <= N; @inbounds out[gm, gn] = Float16(res[r, c]); end
        end
    end; return nothing
end

# ── 4×4 K×8 kernel (32×32 tile, 16 SGs, 512 threads, 8 K-chunks per barrier) ──
function fp16_matmul_4x4_k8!(out, W, x, M::Int32, N::Int32, K::Int32)
    tile_m_grp = Int32(threadgroup_position_in_grid().x)
    tile_n_grp = Int32(threadgroup_position_in_grid().y)
    sg = Int32(simdgroup_index_in_threadgroup()); tid = Int32(thread_index_in_simdgroup())
    gtid = (sg - Int32(1)) * Int32(32) + tid
    sg_m = (sg - Int32(1)) % Int32(4); sg_n = (sg - Int32(1)) ÷ Int32(4)
    w1 = MtlThreadGroupArray(Float32, (32, 8)); x1 = MtlThreadGroupArray(Float32, (8, 32))
    w2 = MtlThreadGroupArray(Float32, (32, 8)); x2 = MtlThreadGroupArray(Float32, (8, 32))
    w3 = MtlThreadGroupArray(Float32, (32, 8)); x3 = MtlThreadGroupArray(Float32, (8, 32))
    w4 = MtlThreadGroupArray(Float32, (32, 8)); x4 = MtlThreadGroupArray(Float32, (8, 32))
    w5 = MtlThreadGroupArray(Float32, (32, 8)); x5 = MtlThreadGroupArray(Float32, (8, 32))
    w6 = MtlThreadGroupArray(Float32, (32, 8)); x6 = MtlThreadGroupArray(Float32, (8, 32))
    w7 = MtlThreadGroupArray(Float32, (32, 8)); x7 = MtlThreadGroupArray(Float32, (8, 32))
    w8 = MtlThreadGroupArray(Float32, (32, 8)); x8 = MtlThreadGroupArray(Float32, (8, 32))
    zt = MtlThreadGroupArray(Float32, (8, 8)); res = MtlThreadGroupArray(Float32, (32, 32))
    if sg == Int32(1) && tid <= Int32(32)
        for e in Int32(0):Int32(1)
            f = (tid - Int32(1)) * Int32(2) + e; r = (f % Int32(8)) + Int32(1); c = (f ÷ Int32(8)) + Int32(1)
            if c <= Int32(8); @inbounds zt[r, c] = 0f0; end
        end
    end
    threadgroup_barrier(Metal.MemoryFlagThreadGroup); acc = simdgroup_load(zt, (1, 1))
    @inline function _lw(dst, ko)
        if gtid <= Int32(256)
            f = gtid - Int32(1); r = (f % Int32(32)) + Int32(1); c = (f ÷ Int32(32)) + Int32(1)
            gm = (tile_m_grp - Int32(1)) * Int32(32) + r; gk = ko + c
            @inbounds dst[r, c] = (gm <= M && gk <= K) ? Float32(W[gm, gk]) : 0f0
        end
    end
    @inline function _lx(dst, ko)
        if gtid > Int32(256) && gtid <= Int32(512)
            f = gtid - Int32(257); r = (f % Int32(8)) + Int32(1); c = (f ÷ Int32(8)) + Int32(1)
            gk = ko + r; gn = (tile_n_grp - Int32(1)) * Int32(32) + c
            @inbounds dst[r, c] = (gk <= K && gn <= N) ? Float32(x[gk, gn]) : 0f0
        end
    end
    wo = (Int64(sg_m) * Int64(8) + Int64(1), Int64(1))
    xo = (Int64(1), Int64(sg_n) * Int64(8) + Int64(1))
    k = Int32(0)
    while k + Int32(64) <= K
        _lw(w1, k); _lx(x1, k); _lw(w2, k + Int32(8)); _lx(x2, k + Int32(8))
        _lw(w3, k + Int32(16)); _lx(x3, k + Int32(16)); _lw(w4, k + Int32(24)); _lx(x4, k + Int32(24))
        _lw(w5, k + Int32(32)); _lx(x5, k + Int32(32)); _lw(w6, k + Int32(40)); _lx(x6, k + Int32(40))
        _lw(w7, k + Int32(48)); _lx(x7, k + Int32(48)); _lw(w8, k + Int32(56)); _lx(x8, k + Int32(56))
        threadgroup_barrier(Metal.MemoryFlagThreadGroup)
        acc = simdgroup_multiply_accumulate(simdgroup_load(w1, wo), simdgroup_load(x1, xo), acc)
        acc = simdgroup_multiply_accumulate(simdgroup_load(w2, wo), simdgroup_load(x2, xo), acc)
        acc = simdgroup_multiply_accumulate(simdgroup_load(w3, wo), simdgroup_load(x3, xo), acc)
        acc = simdgroup_multiply_accumulate(simdgroup_load(w4, wo), simdgroup_load(x4, xo), acc)
        acc = simdgroup_multiply_accumulate(simdgroup_load(w5, wo), simdgroup_load(x5, xo), acc)
        acc = simdgroup_multiply_accumulate(simdgroup_load(w6, wo), simdgroup_load(x6, xo), acc)
        acc = simdgroup_multiply_accumulate(simdgroup_load(w7, wo), simdgroup_load(x7, xo), acc)
        acc = simdgroup_multiply_accumulate(simdgroup_load(w8, wo), simdgroup_load(x8, xo), acc)
        threadgroup_barrier(Metal.MemoryFlagThreadGroup); k += Int32(64)
    end
    while k < K
        _lw(w1, k); _lx(x1, k); threadgroup_barrier(Metal.MemoryFlagThreadGroup)
        acc = simdgroup_multiply_accumulate(simdgroup_load(w1, wo), simdgroup_load(x1, xo), acc)
        threadgroup_barrier(Metal.MemoryFlagThreadGroup); k += Int32(8)
    end
    simdgroup_store(acc, res, (Int64(sg_m) * Int64(8) + Int64(1), Int64(sg_n) * Int64(8) + Int64(1)))
    threadgroup_barrier(Metal.MemoryFlagThreadGroup)
    for p in Int32(0):Int32(1)
        idx = (gtid - Int32(1)) * Int32(2) + p + Int32(1)
        if idx <= Int32(1024)
            r = ((idx - Int32(1)) % Int32(32)) + Int32(1); c = ((idx - Int32(1)) ÷ Int32(32)) + Int32(1)
            gm = (tile_m_grp - Int32(1)) * Int32(32) + r; gn = (tile_n_grp - Int32(1)) * Int32(32) + c
            if gm <= M && gn <= N; @inbounds out[gm, gn] = Float16(res[r, c]); end
        end
    end; return nothing
end

# ── Unsafe 4×4 K×8 (no bounds checks — requires padded matrices) ──
function fp16_matmul_4x4_k8_unsafe!(out, W, x, M::Int32, N::Int32, K::Int32)
    tile_m_grp = Int32(threadgroup_position_in_grid().x)
    tile_n_grp = Int32(threadgroup_position_in_grid().y)
    sg = Int32(simdgroup_index_in_threadgroup()); tid = Int32(thread_index_in_simdgroup())
    gtid = (sg - Int32(1)) * Int32(32) + tid
    sg_m = (sg - Int32(1)) % Int32(4); sg_n = (sg - Int32(1)) ÷ Int32(4)
    w1 = MtlThreadGroupArray(Float32, (32, 8)); x1 = MtlThreadGroupArray(Float32, (8, 32))
    w2 = MtlThreadGroupArray(Float32, (32, 8)); x2 = MtlThreadGroupArray(Float32, (8, 32))
    w3 = MtlThreadGroupArray(Float32, (32, 8)); x3 = MtlThreadGroupArray(Float32, (8, 32))
    w4 = MtlThreadGroupArray(Float32, (32, 8)); x4 = MtlThreadGroupArray(Float32, (8, 32))
    w5 = MtlThreadGroupArray(Float32, (32, 8)); x5 = MtlThreadGroupArray(Float32, (8, 32))
    w6 = MtlThreadGroupArray(Float32, (32, 8)); x6 = MtlThreadGroupArray(Float32, (8, 32))
    w7 = MtlThreadGroupArray(Float32, (32, 8)); x7 = MtlThreadGroupArray(Float32, (8, 32))
    w8 = MtlThreadGroupArray(Float32, (32, 8)); x8 = MtlThreadGroupArray(Float32, (8, 32))
    zt = MtlThreadGroupArray(Float32, (8, 8)); res = MtlThreadGroupArray(Float32, (32, 32))
    if sg == Int32(1) && tid <= Int32(32)
        for e in Int32(0):Int32(1)
            f = (tid - Int32(1)) * Int32(2) + e; r = (f % Int32(8)) + Int32(1); c = (f ÷ Int32(8)) + Int32(1)
            if c <= Int32(8); @inbounds zt[r, c] = 0f0; end
        end
    end
    threadgroup_barrier(Metal.MemoryFlagThreadGroup); acc = simdgroup_load(zt, (1, 1))
    @inline function _lw(dst, ko)
        if gtid <= Int32(256)
            f = gtid - Int32(1); r = (f % Int32(32)) + Int32(1); c = (f ÷ Int32(32)) + Int32(1)
            gm = (tile_m_grp - Int32(1)) * Int32(32) + r; gk = ko + c
            @inbounds dst[r, c] = Float32(@inbounds W[gm, gk])
        end
    end
    @inline function _lx(dst, ko)
        if gtid > Int32(256) && gtid <= Int32(512)
            f = gtid - Int32(257); r = (f % Int32(8)) + Int32(1); c = (f ÷ Int32(8)) + Int32(1)
            gk = ko + r; gn = (tile_n_grp - Int32(1)) * Int32(32) + c
            @inbounds dst[r, c] = Float32(@inbounds x[gk, gn])
        end
    end
    wo = (Int64(sg_m) * Int64(8) + Int64(1), Int64(1))
    xo = (Int64(1), Int64(sg_n) * Int64(8) + Int64(1))
    k = Int32(0)
    while k + Int32(64) <= K
        _lw(w1, k); _lx(x1, k); _lw(w2, k + Int32(8)); _lx(x2, k + Int32(8))
        _lw(w3, k + Int32(16)); _lx(x3, k + Int32(16)); _lw(w4, k + Int32(24)); _lx(x4, k + Int32(24))
        _lw(w5, k + Int32(32)); _lx(x5, k + Int32(32)); _lw(w6, k + Int32(40)); _lx(x6, k + Int32(40))
        _lw(w7, k + Int32(48)); _lx(x7, k + Int32(48)); _lw(w8, k + Int32(56)); _lx(x8, k + Int32(56))
        threadgroup_barrier(Metal.MemoryFlagThreadGroup)
        acc = simdgroup_multiply_accumulate(simdgroup_load(w1, wo), simdgroup_load(x1, xo), acc)
        acc = simdgroup_multiply_accumulate(simdgroup_load(w2, wo), simdgroup_load(x2, xo), acc)
        acc = simdgroup_multiply_accumulate(simdgroup_load(w3, wo), simdgroup_load(x3, xo), acc)
        acc = simdgroup_multiply_accumulate(simdgroup_load(w4, wo), simdgroup_load(x4, xo), acc)
        acc = simdgroup_multiply_accumulate(simdgroup_load(w5, wo), simdgroup_load(x5, xo), acc)
        acc = simdgroup_multiply_accumulate(simdgroup_load(w6, wo), simdgroup_load(x6, xo), acc)
        acc = simdgroup_multiply_accumulate(simdgroup_load(w7, wo), simdgroup_load(x7, xo), acc)
        acc = simdgroup_multiply_accumulate(simdgroup_load(w8, wo), simdgroup_load(x8, xo), acc)
        threadgroup_barrier(Metal.MemoryFlagThreadGroup); k += Int32(64)
    end
    while k < K
        _lw(w1, k); _lx(x1, k); threadgroup_barrier(Metal.MemoryFlagThreadGroup)
        acc = simdgroup_multiply_accumulate(simdgroup_load(w1, wo), simdgroup_load(x1, xo), acc)
        threadgroup_barrier(Metal.MemoryFlagThreadGroup); k += Int32(8)
    end
    simdgroup_store(acc, res, (Int64(sg_m) * Int64(8) + Int64(1), Int64(sg_n) * Int64(8) + Int64(1)))
    threadgroup_barrier(Metal.MemoryFlagThreadGroup)
    for p in Int32(0):Int32(1)
        idx = (gtid - Int32(1)) * Int32(2) + p + Int32(1)
        if idx <= Int32(1024)
            r = ((idx - Int32(1)) % Int32(32)) + Int32(1); c = ((idx - Int32(1)) ÷ Int32(32)) + Int32(1)
            gm = (tile_m_grp - Int32(1)) * Int32(32) + r; gn = (tile_n_grp - Int32(1)) * Int32(32) + c
            @inbounds out[gm, gn] = Float16(res[r, c])
        end
    end; return nothing
end

"""FP16 matmul with auto kernel selection. Assumes padded inputs for B≥32."""
function metal_fp16_matmul!(out, W, x)
    M = Int32(size(W, 1)); K = Int32(size(W, 2)); N = Int32(size(x, 2))
    if N >= 32
        # Unsafe 4×4 K×8: fastest for B≥32 (requires M,N,K multiples of 32/8)
        @metal threads=512 groups=(cld(Int(M), 32), cld(Int(N), 32)) fp16_matmul_4x4_k8_unsafe!(
            out, W, x, M, N, K)
    elseif N >= 16
        # K×2 unrolled 2×2
        @metal threads=128 groups=(cld(Int(M), 16), cld(Int(N), 16)) fp16_matmul_2x2_k2!(
            out, W, x, M, N, K)
    else
        # 1-SG
        @metal threads=32 groups=(cld(Int(M), 8), cld(Int(N), 8)) fp16_matmul_1sg!(
            out, W, x, M, N, K)
    end
    return out
end
