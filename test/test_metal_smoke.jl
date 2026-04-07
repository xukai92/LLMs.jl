using Test
using Metal
using LLMs

@testset "Metal Smoke Test" begin
    @testset "Vector add Float32" begin
        n = 1024
        a = rand(Float32, n)
        b = rand(Float32, n)
        expected = a .+ b

        # CPU reference
        c_cpu = similar(a)
        vector_add!(c_cpu, a, b)
        @test c_cpu ≈ expected

        # Metal kernel
        a_gpu = MtlArray(a)
        b_gpu = MtlArray(b)
        c_gpu = MtlArray(zeros(Float32, n))
        metal_vector_add!(c_gpu, a_gpu, b_gpu)
        c_result = Array(c_gpu)
        @test c_result ≈ expected
    end

    @testset "Vector add Float16" begin
        n = 2048
        a = rand(Float16, n)
        b = rand(Float16, n)
        expected = a .+ b

        a_gpu = MtlArray(a)
        b_gpu = MtlArray(b)
        c_gpu = MtlArray(zeros(Float16, n))
        metal_vector_add!(c_gpu, a_gpu, b_gpu)
        c_result = Array(c_gpu)
        @test c_result ≈ expected
    end

    @testset "Vector add large" begin
        n = 1_000_000
        a = rand(Float32, n)
        b = rand(Float32, n)
        expected = a .+ b

        a_gpu = MtlArray(a)
        b_gpu = MtlArray(b)
        c_gpu = MtlArray(zeros(Float32, n))
        metal_vector_add!(c_gpu, a_gpu, b_gpu)
        c_result = Array(c_gpu)
        @test c_result ≈ expected
    end

    @testset "Metal device info" begin
        dev = Metal.device()
        println("  Metal device: ", dev.name)
        println("  Max threads per threadgroup: ", dev.maxThreadsPerThreadgroup)
        # dev.name is an NSString — just verify it's non-empty
        @test length(string(dev.name)) > 0
    end
end
