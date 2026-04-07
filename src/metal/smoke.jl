"""
Metal smoke test: vector addition kernel.
Verifies that Metal.jl toolchain works on this machine.
"""

"""
Metal kernel: element-wise vector addition.
Each thread computes one element: C[i] = A[i] + B[i]
"""
function vector_add_kernel!(C, A, B)
    i = thread_position_in_grid_1d()
    if i <= length(C)
        @inbounds C[i] = A[i] + B[i]
    end
    return nothing
end

"""
    metal_vector_add!(C::MtlVector, A::MtlVector, B::MtlVector)

Launch the vector add kernel on Metal GPU.
"""
function metal_vector_add!(C::MtlVector, A::MtlVector, B::MtlVector)
    n = length(C)
    threads_per_group = 256
    ngroups = cld(n, threads_per_group)
    @metal threads=threads_per_group groups=ngroups vector_add_kernel!(C, A, B)
    return C
end

"""
    vector_add!(C::AbstractVector, A::AbstractVector, B::AbstractVector)

CPU reference implementation for testing.
"""
function vector_add!(C::AbstractVector, A::AbstractVector, B::AbstractVector)
    @assert length(C) == length(A) == length(B)
    @inbounds for i in eachindex(C)
        C[i] = A[i] + B[i]
    end
    return C
end
