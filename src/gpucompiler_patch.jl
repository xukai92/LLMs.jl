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
const GC = Base.loaded_modules[Base.PkgId(
    Base.UUID("61eb1bfa-7361-4325-ad38-22787b887f55"), "GPUCompiler")]

using .GC: CompilerJob, MetalCompilerTarget, kernel_state_type,
    kernel_state_to_reference!, add_parameter_address_spaces!,
    add_global_address_spaces!, add_argument_metadata!, add_module_metadata!,
    hide_noreturn!, replace_unreachable!, lower_llvm_intrinsics!

using .GC.LLVM: Module, Function, functions, name,
    NewPMPassBuilder, NewPMFunctionPassManager, SimplifyCFGPass, InstCombinePass,
    AlwaysInlinerPass, ModulePassManager, expand_reductions!,
    has_oldpm

import .GC.LLVM  # for @dispose macro

# Override finish_ir! to remove the macOS 15 gate on replace_unreachable!
function GC.finish_ir!(@nospecialize(job::CompilerJob{MetalCompilerTarget}),
                       mod::Module, entry::Function)
    if job.config.kernel && kernel_state_type(job) !== Nothing
        entry = kernel_state_to_reference!(job, mod, entry)
    end

    if job.config.kernel
        entry = add_parameter_address_spaces!(job, mod, entry)
        entry = add_global_address_spaces!(job, mod, entry)
        add_argument_metadata!(job, mod, entry)
        add_module_metadata!(job, mod)
    end

    hide_noreturn!(mod)

    # PATCHED: always run replace_unreachable! (removed macOS < 15 gate)
    any_replaced = false
    for f in functions(mod)
        any_replaced |= replace_unreachable!(job, f)
    end
    if any_replaced
        @LLVM.dispose pb=NewPMPassBuilder() begin
            LLVM.add!(pb, NewPMFunctionPassManager()) do fpm
                LLVM.add!(fpm, SimplifyCFGPass())
                LLVM.add!(fpm, InstCombinePass())
            end
            LLVM.run!(pb, mod)
        end
    end

    changed = false
    for f in functions(mod)
        changed |= lower_llvm_intrinsics!(job, f)
    end
    if changed
        @LLVM.dispose pb=NewPMPassBuilder() begin
            LLVM.add!(pb, AlwaysInlinerPass())
            LLVM.add!(pb, NewPMFunctionPassManager()) do fpm
                LLVM.add!(fpm, SimplifyCFGPass())
                LLVM.add!(fpm, InstCombinePass())
            end
            LLVM.run!(pb, mod)
        end
    end

    if has_oldpm()
        @LLVM.dispose pm=ModulePassManager() begin
            expand_reductions!(pm)
            LLVM.run!(pm, mod)
        end
    end

    return entry
end

end # module
