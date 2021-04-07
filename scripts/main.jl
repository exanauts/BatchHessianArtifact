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

datafile = joinpath(SOURCE_DATA, "case300.m")
device = CUDADevice()
nlp = ExaPF.ReducedSpaceEvaluator(datafile; device=device)
u = ExaPF.initial(nlp)
n = length(u)
nbatch = n

if isa(device, CUDADevice)
    nlp.λ .= 1.0
    hess = CUDA.zeros(Float64, n, n)
    t1, t2 = CUDA.@time BH.batch_hessian!(nlp, hess, u, nbatch)
    @printf("Total hess: %.4f \t Total hessprod: %.4f \n", t1, t2)
    # @profile batch_hessian!(nlp, hess, u, nbatch)
    GC.gc(true)
    CUDA.reclaim()
elseif isa(device, CPU)
    nlp.λ .= 1.0
    hess = zeros(n, n)
    SuiteSparse.UMFPACK.umf_ctrl[8]=2.0
    t1, t2 = @time BH.cpu_hessian!(nlp, hess, u)
    @printf("Total hess: %.4f \t Total hessprod: %.4f \n", t1, t2)
    SuiteSparse.UMFPACK.umf_ctrl[8]=0.0
    t1, t2 = @time BH.cpu_hessian!(nlp, hess, u)
    @printf("Total hess: %.4f \t Total hessprod: %.4f \n", t1, t2)
end
