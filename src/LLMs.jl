module LLMs

using Metal
using KernelAbstractions
using JSON3
using Mmap

# Safetensors weight loading
include("safetensors.jl")

# Metal kernels
include("metal/smoke.jl")

export load_safetensors, load_safetensors_lazy, SafeTensorInfo
export vector_add!, metal_vector_add!

end # module LLMs
