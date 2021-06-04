
# load script implementing all benchmark functions
include("scripts/run_benchmarks.jl")

# BENCHMARK CPU CODE
RESULTS = launch_benchmark(:HESSIAN_CPU)

# BENCHMARK CUDA CODE
if CUDA.has_cuda_gpu()
    RESULTS = launch_benchmark(:BATCH_AUTODIFF)
end

