using CUDA
using BenchmarkTools
using DelimitedFiles
using Statistics

using ExaPF
using BatchHessian
using KernelAbstractions
using Printf
using LinearAlgebra
using SuiteSparse
using BlockPowerFlow.CUSOLVERRF

const BH = BatchHessian
OUTPUTDIR = joinpath(dirname(@__FILE__), "..", "results")
SOURCE_DATA = joinpath(dirname(@__FILE__), "..", "..", "ExaPF.jl", "data")

SUBDIR = Dict(
    :BATCH_AUTODIFF=>"batch_autodiff",
    :BATCH_HESSIAN=>"batch_hessian",
    :HESSIAN_CPU=>"hessian_cpu",
    :ONESHOT=>"oneshot",
)


function bench_batch_autodiff(nlp::ExaPF.ReducedSpaceEvaluator)
    nu = ExaPF.n_variables(nlp)
    nx = ExaPF.get(nlp.model, ExaPF.NumberOfState())
    J = nlp.state_jacobian.x.J

    batches =  [4, 8, 16, 32, 64, 128]
    results = zeros(length(batches), 4)

    for (id, nbatch) in enumerate(batches)

        batch_ad = BH.BatchHessianStorage(nlp.model, J, nbatch)
        MT = isa(nlp.model.device, CUDADevice) ? CuMatrix : Matrix

        hv = zeros(nx + nu, nbatch) |> MT
        tgt = rand(nx + nu, nbatch) |> MT
        @time ExaPF.batch_adj_hessian_prod!(nlp.model, batch_ad.state, hv, nlp.buffer, nlp.λ, tgt)
        trials = @benchmark ExaPF.batch_adj_hessian_prod!(
            $nlp.model,
            $batch_ad.state,
            $hv,
            $nlp.buffer,
            $nlp.λ,
            $tgt
        )

        results[id, 1] = mean(trials.times)
        results[id, 2] = median(trials.times)
        results[id, 3] = std(trials.times)
        results[id, 4] = length(trials.times)

        CUSOLVERRF.cudestroy!(batch_ad.∇g)
        CUSOLVERRF.cudestroy!(batch_ad.∇gᵀ)
    end
    return results
end

function bench_one_shot_hessian(nlp; ntrials=50)
    nu = ExaPF.n_variables(nlp)
    nx = ExaPF.get(nlp.model, ExaPF.NumberOfState())
    J = nlp.state_jacobian.x.J

    nbatch = 16
    timings = zeros(ntrials)
    results = zeros(1, 4)

    batch_ad = BH.BatchHessianStorage(nlp.model, J, nbatch)
    hess = CUDA.zeros(Float64, nu, nu)

    for i in 1:ntrials
        t1, t2 = BH.batch_hessian!(nlp, hess, u, batch_ad, nbatch)
        timings[i] = t1
    end
    GC.gc(true)
    CUDA.reclaim()

    id = 1
    results[id, 1] = mean(timings)
    results[id, 2] = median(timings)
    results[id, 3] = std(timings)
    results[id, 4] = length(timings)

    CUSOLVERRF.cudestroy!(batch_ad.∇g)
    CUSOLVERRF.cudestroy!(batch_ad.∇gᵀ)
    return results
end

function bench_batched_hessian(nlp; ntrials=50)
    nu = ExaPF.n_variables(nlp)
    nx = ExaPF.get(nlp.model, ExaPF.NumberOfState())
    J = nlp.state_jacobian.x.J
    u = ExaPF.initial(nlp)

    batches = [4, 8, 16, 32, 64, 128, 256, 512]
    timings = zeros(ntrials)
    results = zeros(length(batches), 5)

    hess = CUDA.zeros(Float64, nu, nu)

    for (id, nbatch) in enumerate(batches)
        (nbatch > nu) && break
        batch_ad = BH.BatchHessianStorage(nlp.model, J, nbatch)
        for i in 1:ntrials
            t1, t2 = BH.batch_hessian!(nlp, hess, u, batch_ad, nbatch)
            timings[i] = t1
        end
        results[id, 1] = mean(timings)
        results[id, 2] = median(timings)
        results[id, 3] = std(timings)
        results[id, 4] = length(timings)
        results[id, 5] = nbatch

        CUSOLVERRF.cudestroy!(batch_ad.∇g)
        CUSOLVERRF.cudestroy!(batch_ad.∇gᵀ)

        GC.gc(true)
        CUDA.reclaim()
    end
    return results
end

function bench_hessian_cpu(nlp; ntrials=50)
    nu = ExaPF.n_variables(nlp)
    nx = ExaPF.get(nlp.model, ExaPF.NumberOfState())
    J = nlp.state_jacobian.x.J
    u = ExaPF.initial(nlp)

    nbatch = 16
    timings = zeros(ntrials)
    results = zeros(1, 4)
    SuiteSparse.UMFPACK.umf_ctrl[8]=0.0

    hess = zeros(Float64, nu, nu)

    for i in 1:ntrials
        t1, t2 = BH.cpu_hessian!(nlp, hess, u)
        timings[i] = t1
    end

    id = 1
    results[id, 1] = mean(timings)
    results[id, 2] = median(timings)
    results[id, 3] = std(timings)
    results[id, 4] = length(timings)

    return results
end

function launch_benchmark(bench; outputdir=OUTPUTDIR)
    RESULTS = Dict()

    outputdir = joinpath(OUTPUTDIR, SUBDIR[bench])
    for case in [
        # "case30.m",
        # "case118.m",
        # "case300.m",
        # "case1354.m",
        # "case2869.m",
        # "case9241pegase.m",
        # "case19402.m",
        "case30Kc.m",
    ]
        @info case
        datafile = joinpath(SOURCE_DATA, case)
        if bench == :ONESHOT
            nlp = ExaPF.ReducedSpaceEvaluator(datafile; device=CUDADevice())
            nlp.λ .= 1.0
            results = bench_one_shot_hessian(nlp; ntrials=50)
        elseif bench == :BATCH_AUTODIFF
            nlp = ExaPF.ReducedSpaceEvaluator(datafile; device=CUDADevice())
            nlp.λ .= 1.0
            results = bench_batch_autodiff(nlp)
        elseif bench == :BATCH_HESSIAN
            nlp = ExaPF.ReducedSpaceEvaluator(datafile; device=CUDADevice())
            nlp.λ .= 1.0
            results = bench_batched_hessian(nlp)
        elseif bench == :HESSIAN_CPU
            nlp = ExaPF.ReducedSpaceEvaluator(datafile; device=CPU())
            nlp.λ .= 1.0
            results = bench_hessian_cpu(nlp; ntrials=20)
        end
        RESULTS[case] = results

        output = joinpath(outputdir, case)
        writedlm(output, results)
        GC.gc(true)
        CUDA.reclaim()
    end

    return RESULTS
end

# RESULTS = launch_benchmark(:BATCH_HESSIAN)
RESULTS = launch_benchmark(:HESSIAN_CPU)
