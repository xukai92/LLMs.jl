module LLMs

using Metal
using KernelAbstractions
using JSON3
using Mmap

# Safetensors weight loading
include("safetensors.jl")

# Metal kernels
include("metal/smoke.jl")
include("metal/rmsnorm.jl")
include("metal/rope.jl")
include("metal/swiglu.jl")
include("metal/softmax.jl")
include("metal/attention.jl")
include("metal/quantized_matmul.jl")

export load_safetensors, load_safetensors_lazy, SafeTensorInfo
export vector_add!, metal_vector_add!
export rmsnorm_cpu!, metal_rmsnorm!
export compute_rope_freqs, rope_cpu!, metal_rope!
export swiglu_cpu!, metal_swiglu!
export softmax_cpu!, metal_softmax!
export attention_cpu!, metal_attention!
export dequantize_cpu, quantized_matmul_cpu!, metal_quantized_matmul!

end # module LLMs
