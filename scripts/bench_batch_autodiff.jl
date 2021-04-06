using BenchmarkTools
using DelimitedFiles
using Statistics

OUTPUTDIR = joinpath(dirname(@__FILE__), "..", "..", "results", "batch_hessian", "autodiff")

function bench_batch_autodiff(nlp::ExaPF.ReducedSpaceEvaluator)
    nu = ExaPF.n_variables(nlp)
    nx = ExaPF.get(nlp.model, ExaPF.NumberOfState())
    J = nlp.state_jacobian.x.J

    batches =  [4, 8, 16, 32, 64, 128]
    results = zeros(length(batches), 4)

    for (id, nbatch) in enumerate(batches)

        batch_ad = BatchHessianStorage(nlp.model, J, nbatch)
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

function main_batch_autodiff()
    RESULTS = Dict()

    for case in [
        "case118.m",
        "case300.m",
        "case1354.m",
        "case2869.m",
        "case9241pegase.m",
    ]
        @info case
        datafile = joinpath(dirname(@__FILE__), "..", "..", "data", case)
        nlp = ExaPF.ReducedSpaceEvaluator(datafile; device=CUDADevice())
        nlp.λ .= 1.0
        results = bench_batch_autodiff(nlp)
        RESULTS[case] = results

        output = joinpath(OUTPUTDIR, case)
        writedlm(output, results)
        GC.gc(true)
        CUDA.reclaim()
    end

    return RESULTS
end

RESULTS = main_batch_autodiff()
