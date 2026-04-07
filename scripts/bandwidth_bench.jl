"""
Metal memory bandwidth benchmark.

Measures sequential read throughput on Metal GPU to compare
against theoretical bandwidth of the chip.

Theoretical bandwidth:
  M4 Max:  546 GB/s
  M3 Max:  400 GB/s
  M2 Max:  400 GB/s
  M1 Max:  400 GB/s
  M4 Pro:  273 GB/s
  M3 Pro:  150 GB/s

Usage:
    julia --project=. scripts/bandwidth_bench.jl
"""

using Metal
using Printf

function bandwidth_read_kernel!(output, input)
    i = thread_position_in_grid_1d()
    if i <= length(input)
        @inbounds output[i] = input[i]
    end
    return nothing
end

function benchmark_bandwidth(; dtype=Float32, size_mb=256, warmup=3, iters=10)
    elem_size = sizeof(dtype)
    n = (size_mb * 1024 * 1024) ÷ elem_size

    src = MtlArray(rand(dtype, n))
    dst = MtlArray(zeros(dtype, n))

    threads_per_group = 256
    ngroups = cld(n, threads_per_group)

    # Warmup
    for _ in 1:warmup
        @metal threads=threads_per_group groups=ngroups bandwidth_read_kernel!(dst, src)
        Metal.synchronize()
    end

    # Benchmark
    total_bytes = 2 * n * elem_size  # read + write
    times = Float64[]

    for _ in 1:iters
        t0 = time_ns()
        @metal threads=threads_per_group groups=ngroups bandwidth_read_kernel!(dst, src)
        Metal.synchronize()
        t1 = time_ns()
        push!(times, (t1 - t0) / 1e9)
    end

    median_time = sort(times)[length(times) ÷ 2 + 1]
    bw_gbs = total_bytes / median_time / 1e9

    return bw_gbs, median_time, times
end

function main()
    dev = Metal.current_device()
    println("Metal device: $(dev.name)")
    println("Max threads per threadgroup: $(dev.maxThreadsPerThreadgroup)")
    println()

    for (dtype, name) in [(Float32, "Float32"), (Float16, "Float16")]
        for size_mb in [64, 256, 512]
            bw, med_time, times = benchmark_bandwidth(; dtype=dtype, size_mb=size_mb)
            min_time = minimum(times)
            max_bw = 2 * (size_mb * 1024 * 1024) / min_time / 1e9

            @printf("  %-8s  %4d MB  →  median %.1f GB/s  peak %.1f GB/s  (%.2f ms)\n",
                    name, size_mb, bw, max_bw, med_time * 1000)
        end
    end

    println()
    println("Compare against theoretical bandwidth of your chip (see header comment).")
    println("Achieving 60-80% of theoretical is typical for a simple copy kernel.")
end

main()
