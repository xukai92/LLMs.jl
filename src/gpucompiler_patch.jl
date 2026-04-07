"""
    gpucompiler_patch.jl — Monkey-patch GPUCompiler.jl to enable trap elimination on macOS 15+

## Problem
GPUCompiler.jl's `finish_ir!` for Metal has a `replace_unreachable!` pass that removes trap
instructions from GPU IR, but it's gated behind `macos < v"15"`. On macOS 15+, all
`check_sign_bit` calls generate `icmp → br → trap → unreachable` patterns that remain in
the final Metal IR — 323 trap instructions per kernel for our FP16 matmul.

## Fix
Override `GPUCompiler.finish_ir!` for MetalCompilerTarget to always run `replace_unreachable!`
regardless of macOS version, plus a SimplifyCFG cleanup pass to fold dead branches.

## Impact
- IR traps: 323 → 0
- B=2-8:  20-42% faster (trap overhead was large fraction of runtime)
- B=32:   10% faster (now 0.90x MLX, was 1.04x)
- B≥128:  no change (memory-bound)

## How to apply upstream
File as GPUCompiler.jl PR: remove the `macos < v"15"` gate in `finish_ir!` (metal.jl ~line 180).
"""
module GPUCompilerPatch

using Metal

# Access GPUCompiler through Metal's dependency
const GC = let
    # GPUCompiler is loaded as a dependency of Metal
    Base.loaded_modules[Base.PkgId(Base.UUID("61eb1bfa-7361-4325-ad38-22787b887f55"), "GPUCompiler")]
end
const LLVM = GC.LLVM

# Override finish_ir! to remove the macOS 15 gate on replace_unreachable!
function GC.finish_ir!(@nospecialize(job::GC.CompilerJob{GC.MetalCompilerTarget}), mod::LLVM.Module,
                       entry::LLVM.Function)
    entry_fn = LLVM.name(entry)

    if job.config.kernel && GC.kernel_state_type(job) !== Nothing
        entry = GC.kernel_state_to_reference!(job, mod, entry)
    end

    if job.config.kernel
        entry = GC.add_parameter_address_spaces!(job, mod, entry)
        entry = GC.add_global_address_spaces!(job, mod, entry)
        GC.add_argument_metadata!(job, mod, entry)
        GC.add_module_metadata!(job, mod)
    end

    GC.hide_noreturn!(mod)

    # PATCHED: always run replace_unreachable! (removed macOS < 15 gate)
    any_replaced = false
    for f in LLVM.functions(mod)
        any_replaced |= GC.replace_unreachable!(job, f)
    end
    if any_replaced
        @LLVM.dispose pb=LLVM.NewPMPassBuilder() begin
            LLVM.add!(pb, LLVM.NewPMFunctionPassManager()) do fpm
                LLVM.add!(fpm, LLVM.SimplifyCFGPass())
                LLVM.add!(fpm, LLVM.InstCombinePass())
            end
            LLVM.run!(pb, mod)
        end
    end

    changed = false
    for f in LLVM.functions(mod)
        changed |= GC.lower_llvm_intrinsics!(job, f)
    end
    if changed
        @LLVM.dispose pb=LLVM.NewPMPassBuilder() begin
            LLVM.add!(pb, LLVM.AlwaysInlinerPass())
            LLVM.add!(pb, LLVM.NewPMFunctionPassManager()) do fpm
                LLVM.add!(fpm, LLVM.SimplifyCFGPass())
                LLVM.add!(fpm, LLVM.InstCombinePass())
            end
            LLVM.run!(pb, mod)
        end
    end

    if LLVM.has_oldpm()
        @LLVM.dispose pm=LLVM.ModulePassManager() begin
            LLVM.expand_reductions!(pm)
            LLVM.run!(pm, mod)
        end
    end

    return entry
end

end # module
