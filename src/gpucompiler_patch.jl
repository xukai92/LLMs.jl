"""
    gpucompiler_patch.jl — Monkey-patch GPUCompiler.jl to enable trap elimination on macOS 15+

Must be applied at runtime via `__init__` (method overwriting not allowed during precompile).
"""
module GPUCompilerPatch

function apply!()
    GC = Base.loaded_modules[Base.PkgId(
        Base.UUID("61eb1bfa-7361-4325-ad38-22787b887f55"), "GPUCompiler")]
    LLVM = GC.LLVM

    @eval function $GC.finish_ir!(job::$GC.CompilerJob{$GC.MetalCompilerTarget},
                                   mod::$LLVM.Module, entry::$LLVM.Function)
        if job.config.kernel && $GC.kernel_state_type(job) !== Nothing
            entry = $GC.kernel_state_to_reference!(job, mod, entry)
        end
        if job.config.kernel
            entry = $GC.add_parameter_address_spaces!(job, mod, entry)
            entry = $GC.add_global_address_spaces!(job, mod, entry)
            $GC.add_argument_metadata!(job, mod, entry)
            $GC.add_module_metadata!(job, mod)
        end
        $GC.hide_noreturn!(mod)

        any_replaced = false
        for f in $LLVM.functions(mod)
            any_replaced |= $GC.replace_unreachable!(job, f)
        end
        if any_replaced
            pb = $LLVM.NewPMPassBuilder()
            try
                fpm = $LLVM.NewPMFunctionPassManager()
                $LLVM.add!(fpm, $LLVM.SimplifyCFGPass())
                $LLVM.add!(fpm, $LLVM.InstCombinePass())
                $LLVM.add!(pb, fpm)
                $LLVM.run!(pb, mod)
            finally
                $LLVM.dispose(pb)
            end
        end

        changed = false
        for f in $LLVM.functions(mod)
            changed |= $GC.lower_llvm_intrinsics!(job, f)
        end
        if changed
            pb = $LLVM.NewPMPassBuilder()
            try
                $LLVM.add!(pb, $LLVM.AlwaysInlinerPass())
                fpm = $LLVM.NewPMFunctionPassManager()
                $LLVM.add!(fpm, $LLVM.SimplifyCFGPass())
                $LLVM.add!(fpm, $LLVM.InstCombinePass())
                $LLVM.add!(pb, fpm)
                $LLVM.run!(pb, mod)
            finally
                $LLVM.dispose(pb)
            end
        end

        if $LLVM.has_oldpm()
            pm = $LLVM.ModulePassManager()
            try
                $LLVM.expand_reductions!(pm)
                $LLVM.run!(pm, mod)
            finally
                $LLVM.dispose(pm)
            end
        end

        return entry
    end
end

end # module
