using Test

@testset "LLMs.jl" begin
    include("test_safetensors.jl")
    include("test_metal_smoke.jl")
    include("test_kernels.jl")
end
