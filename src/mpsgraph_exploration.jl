"""
    mpsgraph_exploration.jl — MPSGraph accessibility from Julia

MPSGraph (MetalPerformanceShadersGraph) is Apple's graph-level computation
framework — the same engine MLX uses for fused GPU operations. It provides:
- Graph-level operation fusion (matmul + add + activation in one kernel)
- Optimized memory management (no intermediate buffers)
- Hardware-specific optimization (uses Metal's GPU scheduler)

## Discovery

All MPSGraph classes and methods are accessible from Julia via ObjectiveC.jl
(a dependency of Metal.jl):

```julia
OC = Base.loaded_modules[Base.PkgId(
    Base.UUID("e86c9b32-1129-44ac-8ea0-90d5bb39ded9"), "ObjectiveC")]
OC.load_framework("MetalPerformanceShadersGraph")

# Create graph
graph = ccall(:objc_msgSend, Ptr{Cvoid}, (Ptr{Cvoid}, Ptr{Cvoid}),
    ccall(:objc_msgSend, Ptr{Cvoid}, (Ptr{Cvoid}, Ptr{Cvoid}),
        ccall(:objc_getClass, Ptr{Cvoid}, (Cstring,), "MPSGraph"),
        ccall(:sel_registerName, Ptr{Cvoid}, (Cstring,), "alloc")),
    ccall(:sel_registerName, Ptr{Cvoid}, (Cstring,), "init"))
```

## Available methods (verified):
- `placeholderWithShape:dataType:name:` — create input tensors
- `matrixMultiplicationWithPrimaryTensor:secondaryTensor:name:` — matmul
- `rmsNormalizationWithTensor:axes:gamma:epsilon:name:` — rmsnorm
- `runWithMTLCommandQueue:feeds:targetTensors:targetOperations:` — execute
- `compileWithDevice:feeds:targetTensors:targetOperations:compilationDescriptor:` — pre-compile

## Approach for LLM inference

Build the entire transformer forward pass as an MPSGraph:
1. Create placeholder tensors for input tokens, KV cache, weights
2. Build the computation graph: embed → (rmsnorm → QKV → rope → attn → O → add →
   rmsnorm → gate_up → swiglu → down → add) × N_layers → rmsnorm → lm_head
3. Compile once with `compileWithDevice:`
4. Execute per-token with `runWithMTLCommandQueue:`

This would give us MLX-level performance since it's the same underlying framework.

## Complexity

High — requires wrapping all MPSGraph operations with proper ObjectiveC bindings.
The ObjectiveC API is verbose (many selector calls) but straightforward.
Could start with just the matmul + rmsnorm path as a proof of concept.
"""
