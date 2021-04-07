using CUDA
using ExaPF
using BatchHessian
using KernelAbstractions
using Printf
using LinearAlgebra
using SuiteSparse
using BlockPowerFlow.CUSOLVERRF

const BH = BatchHessian


SOURCE_DATA = joinpath(dirname(@__FILE__), "..", "..", "ExaPF.jl", "data")

function batch_hessian!(nlp::ExaPF.AbstractNLPEvaluator, hess, u, nbatch)
    n = ExaPF.n_variables(nlp)
    # Init AD
    J = nlp.state_jacobian.x.J
    batch_ad = BH.BatchHessianStorage(nlp.model, J, nbatch)
    # Allocate memory
    v_cpu = zeros(n, nbatch)

    v = similar(u, n, nbatch)
    hm = similar(u, n, nbatch)

    tic = time()

    ExaPF.update!(nlp.model, batch_ad.state, nlp.buffer)
    N = div(n, nbatch, RoundDown)
    for i in 1:N
        fill!(v_cpu, 0.0)
        @inbounds for j in 1:nbatch
            v_cpu[j+(i-1)*nbatch, j] = 1.0
        end
        copyto!(v, v_cpu)
        BH.batch_hessprod!(nlp, batch_ad, hm, u, v)

        copyto!(hess, (i-1)*nbatch*n + 1, hm, 1, n*nbatch)
    end

    # Last slice
    last_batch = n - N*nbatch
    if last_batch > 0
        fill!(v_cpu, 0.0)
        @inbounds for j in 1:nbatch
            v_cpu[n-j+1, j] = 1.0
        end
        copyto!(v, v_cpu)
        BH.batch_hessprod!(nlp, batch_ad, hm, u, v)
        # Keep only last columns in hm
        copyto!(hess, N*nbatch*n + 1, hm, (nbatch - last_batch)*n, n*last_batch)
    end

    println("Time elapsed: ", time() - tic)

    CUSOLVERRF.cudestroy!(batch_ad.∇g)
    CUSOLVERRF.cudestroy!(batch_ad.∇gᵀ)

    return
end

datafile = joinpath(SOURCE_DATA, "case118.m")
device = CUDADevice()
nbatch = 8
nlp = ExaPF.ReducedSpaceEvaluator(datafile; device=device)
u = ExaPF.initial(nlp)
n = length(u)

if isa(device, CUDADevice)
    nlp.λ .= 1.0
    hess = CUDA.zeros(Float64, n, n)
    CUDA.@time batch_hessian!(nlp, hess, u, nbatch)
    # @profile batch_hessian!(nlp, hess, u, nbatch)
    GC.gc(true)
    CUDA.reclaim()
elseif isa(device, CPU)
    nlp.λ .= 1.0
    hess = zeros(n, n)
    SuiteSparse.UMFPACK.umf_ctrl[8]=2.0
    @time BH.cpu_hessian!(nlp, hess, u)
    SuiteSparse.UMFPACK.umf_ctrl[8]=0.0
    @time BH.cpu_hessian!(nlp, hess, u)
end
