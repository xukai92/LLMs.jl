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
include("metal/elementwise.jl")

# Model definition and loading
include("model.jl")
include("kvcache.jl")
include("forward.jl")
include("tokenizer.jl")

# Prefix cache
include("prefix_cache.jl")

# Phase 0 exports
export load_safetensors, load_safetensors_lazy, SafeTensorInfo
export vector_add!, metal_vector_add!

# Phase 1 kernel exports
export rmsnorm_cpu!, metal_rmsnorm!
export compute_rope_freqs, rope_cpu!, metal_rope!
export swiglu_cpu!, metal_swiglu!
export softmax_cpu!, metal_softmax!
export attention_cpu!, metal_attention!
export dequantize_cpu, quantized_matmul_cpu!, metal_quantized_matmul!
export metal_add!

# Phase 2 model exports
export LlamaConfig, LlamaModel, load_llama_model
export KVCache, append_kv!, get_kv, reset!
export forward, generate
export Tokenizer, encode, decode, encode_chat

# Phase 3 prefix cache exports
export PrefixCache, prefix_match, insert_prefix!, restore_kv!
export generate_with_cache

end # module LLMs
