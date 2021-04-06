using CUDA
using CUDA.CUSPARSE
using CUDA.CUSOLVER

using BlockPowerFlow
using BlockPowerFlow.CUSOLVERRF
using Statistics
using LinearAlgebra
using SuiteSparse
using MatrixMarket
using BenchmarkTools
using DelimitedFiles
using Test

OUTPUTDIR = joinpath(dirname(@__FILE__), "..", "..", "results", "batch_hessian", "cusolver")

function solve_rhs(J)
    n, m = size(J)
    gJ = CuSparseMatrixCSR(J)
    b = randn(n)
    gb = CuVector{Float64}(b)
    x0 = zeros(n)
    x = CUDA.zeros(Float64, m)

    batches =  [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]

    results = zeros(2 + length(batches), 4)
    f = lu(J)
    solution = J \ b
    # Compute solution with UMFPACK
    @info "UMFPACK (default)"
    SuiteSparse.UMFPACK.umf_ctrl[8] = 2
    @btime lu!($f, $J)
    trials = @benchmark ldiv!($x0, $f, $b)
    results[1, 1] = mean(trials.times)
    results[1, 2] = median(trials.times)
    results[1, 3] = std(trials.times)
    results[1, 4] = length(trials.times)

    @info "UMFPACK (without iterative refinement)"
    SuiteSparse.UMFPACK.umf_ctrl[8] = 0
    @btime lu!($f, $J)
    trials = @benchmark ldiv!($x0, $f, $b)
    results[2, 1] = mean(trials.times)
    results[2, 2] = median(trials.times)
    results[2, 3] = std(trials.times)
    results[2, 4] = length(trials.times)

    @info "CUSOLVERRF (batch)"
    # Test batch mode
    for (id, nbatch) in enumerate(batches)
        gB = CuMatrix{Float64}(undef, n, nbatch)
        gX = CuMatrix{Float64}(undef, n, nbatch)
        for i in 1:nbatch
            gB[:, i] .= gb
        end
        rflu = CUSOLVERRF.CusolverRfLUBatch(gJ, nbatch; fast_mode=true, ordering=:AMD)
        trials = @benchmark ldiv!($gX, $rflu, $gB)
        results[2+id, 1] = mean(trials.times)
        results[2+id, 2] = median(trials.times)
        results[2+id, 3] = std(trials.times)
        results[2+id, 4] = length(trials.times)
        CUSOLVERRF.cudestroy!(rflu)
    end

    return results
end

case = "case300.txt"
datafile = joinpath(dirname(@__FILE__), "..", "..", case)
J = mmread(datafile)
nrhs = 64
res = solve_rhs(J)
output = joinpath(OUTPUTDIR, case)
writedlm(output, res)
GC.gc(true)
CUDA.reclaim()
