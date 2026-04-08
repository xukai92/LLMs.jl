"""
Research notes: Metal simdgroup_async_copy AIR intrinsics

## Status: BLOCKED — no wait mechanism available

The async copy intrinsic compiles and fires (verified: it doesn't crash, runs on GPU),
but we cannot wait for it to complete because `simdgroup_event_wait` is not in the
Metal backend compiler. The data never becomes visible in threadgroup memory.

## What works:
- `air.simdgroup_async_copy_2d.p3i8.p1i8` compiles with 12 parameters
- The DMA appears to fire (non-zero values observed in some tests)
- Metal.jl address spaces match AIR (Device=1, ThreadGroup=3)
- `threadgroup_barrier` does NOT wait for async copies

## What doesn't work:
- `air.simdgroup_event_wait(i32, i64)` — "native code failed"
- `air.simdgroup_event_wait(i32, ptr)` — "native code failed"
- `air.simdgroup_event.wait` — not found
- `air.wait_simdgroup_events` — not found
- `air.simdgroup_async_copy_fence` — not found
- `threadgroup_barrier` with various flags — doesn't wait for DMA

## 2D intrinsic signature (12 params):
```
air.simdgroup_async_copy_2d.p3i8.p1i8(
    i64,              # param 0: unknown (simd info?)
    i64,              # param 1: unknown (num threads?)
    p3 i8*,           # param 2: dst (threadgroup)
    i64,              # param 3: dst_bytes_per_row
    i64,              # param 4: dst_num_rows
    <2 x i64>,        # param 5: dst_origin (col_bytes, row)
    p1 i8*,           # param 6: src (device)
    i64,              # param 7: src_bytes_per_row
    i64,              # param 8: src_num_rows
    <2 x i64>,        # param 9: src_origin
    <2 x i64>,        # param 10: copy_size (width_bytes, height)
    i32               # param 11: flags
) -> i64 (event handle)
```

## Path forward:
1. Need Xcode (not just CommandLineTools) to compile MSL → AIR with async_copy
   and inspect the generated wait instruction name
2. Alternatively: write an MSL kernel file, compile to .metallib with Xcode's
   metal/metallib tools, and load it from Julia via Metal.jl's MTLLibrary
3. File upstream issue for Metal.jl to expose the async copy API
"""
