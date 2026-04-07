"""
Elementwise utility kernels: add (residual connections).
"""

function add_kernel!(out, a, b, n::Int32)
    i = thread_position_in_grid_1d()
    if i <= n
        @inbounds out[i] = a[i] + b[i]
    end
    return nothing
end

"""
    metal_add!(out, a, b) — elementwise addition
"""
function metal_add!(out::MtlArray, a::MtlArray, b::MtlArray)
    n = length(out)
    tg = 256
    @metal threads=tg groups=cld(n, tg) add_kernel!(out, a, b, Int32(n))
    return out
end

"""
    metal_add!(a, b) — in-place addition: a .+= b
"""
function metal_add!(a::MtlArray, b::MtlArray)
    metal_add!(a, a, b)
end
