using Test
using LLMs

@testset "Safetensors" begin
    # Create a minimal safetensors file for testing
    # Format: 8-byte header length + JSON header + raw data

    function write_test_safetensors(path, tensors::Vector{Tuple{String, String, Vector{Int}, Vector{UInt8}}})
        # tensors: [(name, dtype_str, shape, raw_bytes), ...]
        # Order is preserved — data is written in the order given.
        header = Dict{String, Any}()
        offset = 0
        for (name, dtype, shape, data) in tensors
            header[name] = Dict(
                "dtype" => dtype,
                "shape" => shape,
                "data_offsets" => [offset, offset + length(data)]
            )
            offset += length(data)
        end

        header_json = JSON3.write(header)
        # Pad to 8-byte alignment
        pad = (8 - length(header_json) % 8) % 8
        header_json_padded = header_json * " "^pad

        open(path, "w") do io
            write(io, UInt64(length(header_json_padded)))
            write(io, header_json_padded)
            for (_, _, _, data) in tensors
                write(io, data)
            end
        end
    end

    using JSON3

    mktempdir() do tmpdir
        @testset "Float32 1D tensor" begin
            path = joinpath(tmpdir, "test_f32.safetensors")
            vals = Float32[1.0, 2.0, 3.0, 4.0]
            raw = reinterpret(UInt8, vals) |> collect
            write_test_safetensors(path, [("weights", "F32", [4], raw)])

            data = load_safetensors(path; mmap_data=false)
            @test haskey(data, "weights")
            @test data["weights"] ≈ vals
            @test eltype(data["weights"]) == Float32
        end

        @testset "Float16 2D tensor" begin
            path = joinpath(tmpdir, "test_f16.safetensors")
            vals = Float16[1.0 3.0; 2.0 4.0]  # 2x2 Julia (col-major)
            # safetensors is row-major: shape [2, 2] means rows=2, cols=2
            # Row-major data for [[1, 2], [3, 4]]:
            row_major_data = Float16[1.0, 2.0, 3.0, 4.0]
            raw = reinterpret(UInt8, row_major_data) |> collect
            write_test_safetensors(path, [("mat", "F16", [2, 2], raw)])

            data = load_safetensors(path; mmap_data=false)
            @test haskey(data, "mat")
            @test size(data["mat"]) == (2, 2)  # reversed from safetensors [2,2]
            @test eltype(data["mat"]) == Float16
            # Row-major [1,2;3,4] -> Julia col-major should give same memory layout
            @test vec(data["mat"]) == row_major_data
        end

        @testset "UInt32 tensor (for quantized weights)" begin
            path = joinpath(tmpdir, "test_u32.safetensors")
            vals = UInt32[0xDEADBEEF, 0xCAFEBABE, 0x12345678]
            raw = reinterpret(UInt8, vals) |> collect
            write_test_safetensors(path, [("packed", "U32", [3], raw)])

            data = load_safetensors(path; mmap_data=false)
            @test data["packed"] == vals
        end

        @testset "Multiple tensors" begin
            path = joinpath(tmpdir, "test_multi.safetensors")
            w = Float32[1.0, 2.0]
            b = Float32[0.5]
            write_test_safetensors(path, [
                ("weight", "F32", [2], reinterpret(UInt8, w) |> collect),
                ("bias",   "F32", [1], reinterpret(UInt8, b) |> collect),
            ])

            data = load_safetensors(path; mmap_data=false)
            @test data["weight"] ≈ w
            @test data["bias"] ≈ b
        end

        @testset "Mmap loading" begin
            path = joinpath(tmpdir, "test_mmap.safetensors")
            vals = Float32[10.0, 20.0, 30.0]
            raw = reinterpret(UInt8, vals) |> collect
            write_test_safetensors(path, [("data", "F32", [3], raw)])

            data = load_safetensors(path; mmap_data=true)
            @test data["data"] ≈ vals
            @test haskey(data, "__mmap_handle__")
        end

        @testset "Lazy loading" begin
            path = joinpath(tmpdir, "test_lazy.safetensors")
            vals = Float32[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
            raw = reinterpret(UInt8, vals) |> collect
            write_test_safetensors(path, [
                ("a", "F32", [3], reinterpret(UInt8, Float32[1.0, 2.0, 3.0]) |> collect),
                ("b", "F32", [3], reinterpret(UInt8, Float32[4.0, 5.0, 6.0]) |> collect),
            ])

            infos, loader = load_safetensors_lazy(path)
            @test length(infos) == 2

            # Load just one tensor
            for info in infos
                if info.name == "a"
                    t = loader(info)
                    @test t ≈ Float32[1.0, 2.0, 3.0]
                end
            end
        end
    end
end
