"""
FP16 matmul via native MSL kernel, compiled at runtime.

Uses Metal Shading Language struct pointer casting for vectorized tile loading
(the same technique MLX uses), bypassing GPUCompiler's instruction overhead.
"""
module FP16MatmulMSL

using Metal
using Metal.MTL

const MSL_KERNEL = """
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// Vectorized load: 4 halfs (8 bytes) via float2 cast
inline void load_float2(threadgroup half* dst, device const half* src) {
    *((threadgroup float2*)dst) = *((device const float2*)src);
}

kernel void fp16_matmul_msl(
    device const half* W [[buffer(0)]],      // (M, K) column-major
    device const half* x [[buffer(1)]],      // (K, N) column-major
    device half* out [[buffer(2)]],          // (M, N) column-major
    constant int& M [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    uint2 group_id [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint tid [[thread_index_in_simdgroup]],
    uint gtid_raw [[thread_position_in_threadgroup]]
) {
    // 4x4 simdgroup grid, 32x32 output tile, K×8 unrolling
    const int TILE = 32;
    const int sg_m = sg_idx % 4;
    const int sg_n = sg_idx / 4;
    const int gtid = gtid_raw + 1; // 1-indexed to match Julia convention

    // Shared memory tiles (8 pairs for K×8 unrolling)
    threadgroup float w_shmem[8][TILE * 8];  // 8 tiles × (32 rows × 8 cols)
    threadgroup float x_shmem[8][8 * TILE];  // 8 tiles × (8 rows × 32 cols)

    // Zero accumulator
    simdgroup_float8x8 acc;
    {
        threadgroup float zt[64];
        if (sg_idx == 0 && tid < 32) {
            zt[tid * 2] = 0.0f;
            zt[tid * 2 + 1] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 zero;
        simdgroup_load(zero, (threadgroup float*)zt, 8);
        acc = zero;
    }

    const int tile_m = group_id.x * TILE;
    const int tile_n = group_id.y * TILE;

    for (int k = 0; k + 64 <= K; k += 64) {
        // Load 8 W tiles: vectorized 4-half loads via float2 cast
        // 128 threads load 32×8=256 elements per tile, 2 at a time = 128 threads
        for (int t = 0; t < 8; t++) {
            int local_id = gtid_raw;
            if (local_id < 128) {
                int pair = local_id % 16;
                int col = local_id / 16;
                int row = pair * 2;
                int gm = tile_m + row;
                int gk = k + t * 8 + col;
                device const half* src_ptr = W + (long)gk * M + gm;
                threadgroup float* dst_base = w_shmem[t] + col * TILE + row;
                // Load 2 halfs, convert to float, store to shmem
                half2 v = *((device const half2*)src_ptr);
                dst_base[0] = (float)v.x;
                dst_base[1] = (float)v.y;
            }
        }
        // Load 8 x tiles
        for (int t = 0; t < 8; t++) {
            int local_id = gtid_raw;
            if (local_id >= 128 && local_id < 256) {
                int f = local_id - 128;
                int pair = f % 4;
                int col = f / 4;
                int row = pair * 2;
                int gk = k + t * 8 + row;
                int gn = tile_n + col;
                device const half* src_ptr = x + (long)gn * K + gk;
                threadgroup float* dst_base = x_shmem[t] + col * 8 + row;
                half2 v = *((device const half2*)src_ptr);
                dst_base[0] = (float)v.x;
                dst_base[1] = (float)v.y;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 8 MACs
        for (int t = 0; t < 8; t++) {
            simdgroup_float8x8 w_mat, x_mat;
            simdgroup_load(w_mat, w_shmem[t] + sg_m * 8, TILE);
            simdgroup_load(x_mat, x_shmem[t] + sg_n * 8 * 8, 8);
            simdgroup_multiply_accumulate(acc, w_mat, x_mat, acc);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store result
    threadgroup float res[TILE * TILE];
    simdgroup_store(acc, res + sg_n * 8 * TILE + sg_m * 8, TILE);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int p = 0; p < 2; p++) {
        int idx = gtid_raw * 2 + p;
        if (idx < TILE * TILE) {
            int r = idx % TILE;
            int c = idx / TILE;
            int gm = tile_m + r;
            int gn = tile_n + c;
            if (gm < M && gn < N) {
                out[(long)gn * M + gm] = (half)res[c * TILE + r];
            }
        }
    }
}
"""

mutable struct MSLMatmul
    pipeline::Any
    compiled::Bool
    MSLMatmul() = new(nothing, false)
end

const _instance = MSLMatmul()

function ensure_compiled!()
    _instance.compiled && return
    dev = Metal.device()
    lib = MTL.MTLLibrary(dev, MSL_KERNEL)
    fn = MTL.MTLFunction(lib, "fp16_matmul_msl")
    _instance.pipeline = MTL.MTLComputePipelineState(dev, fn)
    _instance.compiled = true
end

"""FP16 matmul using native MSL kernel. out = W @ x."""
function metal_fp16_matmul_msl!(out, W, x)
    ensure_compiled!()
    M = Int32(size(W, 1)); K = Int32(size(W, 2)); N = Int32(size(x, 2))

    queue = Metal.global_queue(Metal.device())
    cmdbuf = MTL.MTLCommandBuffer(queue)
    cce = MTL.MTLComputeCommandEncoder(cmdbuf)
    MTL.set_function!(cce, _instance.pipeline)
    MTL.set_buffer!(cce, W.data[], 0, 1)    # buffer 0
    MTL.set_buffer!(cce, x.data[], 0, 2)    # buffer 1
    MTL.set_buffer!(cce, out.data[], 0, 3)  # buffer 2

    # Set M, N, K as constant buffers
    m_buf = Metal.alloc(Metal.device(), sizeof(Int32); storage=Metal.SharedStorage)
    n_buf = Metal.alloc(Metal.device(), sizeof(Int32); storage=Metal.SharedStorage)
    k_buf = Metal.alloc(Metal.device(), sizeof(Int32); storage=Metal.SharedStorage)
    unsafe_store!(convert(Ptr{Int32}, m_buf), M)
    unsafe_store!(convert(Ptr{Int32}, n_buf), N)
    unsafe_store!(convert(Ptr{Int32}, k_buf), K)
    MTL.set_buffer!(cce, m_buf, 0, 4)  # buffer 3
    MTL.set_buffer!(cce, n_buf, 0, 5)  # buffer 4
    MTL.set_buffer!(cce, k_buf, 0, 6)  # buffer 5

    groups = MTL.MTLSize(cld(Int(M), 32), cld(Int(N), 32), 1)
    threads = MTL.MTLSize(512, 1, 1)
    MTL.append_current_function!(cce, groups, threads)
    close(cce)
    commit!(cmdbuf)
end

end # module
