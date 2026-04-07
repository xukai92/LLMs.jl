"""
    gpucompiler_patch.jl — Patch for GPUCompiler.jl trap elimination on macOS 15+

## Problem
GPUCompiler.jl's `replace_unreachable!` pass removes trap instructions from GPU IR,
but is gated behind `macos < v"15"`. On macOS 15+, all `check_sign_bit` calls generate
`icmp → br → trap → unreachable` patterns that remain in the final Metal IR.

For our FP16 matmul kernel (3072×3072), this means 323 trap instructions and 36
conditional branches for sign-bit checking — adding 10-42% overhead at small batch sizes.

## Fix
Two changes to `~/.julia/packages/GPUCompiler/Yuvf5/src/metal.jl`:

### 1. Remove the macOS version gate (line ~180)
```julia
# Before:
if job.config.target.macos < v"15"
    for f in functions(mod)
        replace_unreachable!(job, f)
    end
end

# After:
any_replaced = false
for f in functions(mod)
    any_replaced |= replace_unreachable!(job, f)
end
```

### 2. Add SimplifyCFG cleanup after trap removal
```julia
# After replace_unreachable! loop:
if any_replaced
    @dispose pb=NewPMPassBuilder() begin
        add!(pb, NewPMFunctionPassManager()) do fpm
            add!(fpm, SimplifyCFGPass())
            add!(fpm, InstCombinePass())
        end
        run!(pb, mod)
    end
end
```

## Impact
- IR traps: 323 → 0 (1 remaining is just the declaration)
- B=2-8: 20-42% faster
- B=32: 10% faster (now beats MLX by 7%)
- B≥128: no change (memory-bound)

## How to apply
Edit `~/.julia/packages/GPUCompiler/Yuvf5/src/metal.jl` and clear caches:
```bash
rm -rf ~/.julia/compiled/v1.11/GPUCompiler/ ~/.julia/compiled/v1.11/Metal/
```

## Upstream PR
This should be filed as a bug fix for JuliaGPU/GPUCompiler.jl — the macOS 15 gate
appears to have been added as a workaround that inadvertently disabled trap elimination
for all macOS 15+ users.
"""
module GPUCompilerPatch
# This module exists only as documentation. The actual patch must be applied
# directly to GPUCompiler.jl's source at:
#   ~/.julia/packages/GPUCompiler/Yuvf5/src/metal.jl
end
