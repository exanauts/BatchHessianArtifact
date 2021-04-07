module BatchHessian

using LinearAlgebra
using SparseArrays

# GPU library
using CUDA.CUSPARSE
using KernelAbstractions

# ExaPF
using ExaPF
using ExaPF.AutoDiff
using ExaPF.PowerSystem
using BlockPowerFlow.CUSOLVERRF

struct BatchHessianStorage{VT, MT, Hess, Fac1, Fac2}
    nbatch::Int
    state::Hess
    # Batch factorization
    ∇g::Fac1
    ∇gᵀ::Fac2
    # RHS
    ∂f::VT
    ∂²f::VT
    # Adjoints
    z::MT
    ψ::MT
    tmp_tgt::MT
    tmp_hv::MT
end

function BatchHessianStorage(polar::PolarForm{T, VI, VT, MT}, J, nbatch::Int) where {T, VI, VT, MT}
    nx = get(polar, NumberOfState())
    nu = get(polar, NumberOfControl())
    ngen = get(polar, PowerSystem.NumberOfGenerators())
    Hstate = ExaPF.batch_hessian(polar, ExaPF.power_balance, nbatch)
    ∂f = VT(undef, ngen)
    ∂²f = VT(undef, ngen)

    z = MT(undef, nx, nbatch)
    ψ = MT(undef, nx, nbatch)
    tgt = MT(undef, nx+nu, nbatch)
    hv =  MT(undef, nx+nu, nbatch)

    if isa(polar.device, CPU)
        ∇g = lu(J)
        ∇gᵀ = ∇g'
    else
        ∇g = CUSOLVERRF.CusolverRfLUBatch(J, nbatch; fast_mode=true)
        Jᵀ = CUSPARSE.CuSparseMatrixCSC(J)
        ∇gᵀ = CUSOLVERRF.CusolverRfLUBatch(Jᵀ, nbatch; fast_mode=true)
    end

    return BatchHessianStorage(nbatch, Hstate, ∇g, ∇gᵀ, ∂f, ∂²f, z, ψ, tgt, hv)
end

function batch_hessprod!(nlp::ExaPF.ReducedSpaceEvaluator, batch_ad, hessmat, u, w)
    @assert nlp.hessians != nothing

    nx = get(nlp.model, NumberOfState())
    nu = get(nlp.model, NumberOfControl())
    ∇gᵤ = nlp.state_jacobian.u.J

    nbatch = batch_ad.nbatch
    # Second order adjoints
    ψ = batch_ad.ψ
    z = batch_ad.z
    # Two vector products
    tgt = batch_ad.tmp_tgt
    hv = batch_ad.tmp_hv

    mul!(z, ∇gᵤ, w, -1.0, 0.0)
    LinearAlgebra.ldiv!(batch_ad.∇g, z)

    # Init tangent
    for i in 1:nbatch
        mt = 1 + (i-1)*(nx+nu)
        mz = 1 + (i-1)*nx
        copyto!(tgt, mt, z, mz, nx)
        mw = 1 + (i-1)*nu
        copyto!(tgt, mt+nx, w, mw, nu)
    end

    ## OBJECTIVE HESSIAN
    # TODO: not implemented yet
    ∂fₓ = hv[1:nx, :]
    ∂fᵤ = hv[nx+1:nx+nu, :]

    ExaPF.batch_adj_hessian_prod!(nlp.model, batch_ad.state, hv, nlp.buffer, nlp.λ, tgt)
    ∂fₓ .= @view hv[1:nx, :]
    ∂fᵤ .= @view hv[nx+1:nx+nu, :]

    # Second order adjoint
    LinearAlgebra.ldiv!(ψ, batch_ad.∇gᵀ, ∂fₓ)

    hessmat .= ∂fᵤ
    mul!(hessmat, transpose(∇gᵤ), ψ, -1.0, 1.0)

    return
end

function batch_hessian!(nlp::ExaPF.AbstractNLPEvaluator, hess, u, nbatch)
    J = nlp.state_jacobian.x.J
    # Init AD
    batch_ad = BatchHessianStorage(nlp.model, J, nbatch)
    res = batch_hessian!(nlp, hess, u, batch_ad, nbatch)
    CUSOLVERRF.cudestroy!(batch_ad.∇g)
    CUSOLVERRF.cudestroy!(batch_ad.∇gᵀ)
    return res
end

function batch_hessian!(nlp::ExaPF.AbstractNLPEvaluator, hess, u, batch_ad, nbatch)
    n = ExaPF.n_variables(nlp)
    # Allocate memory
    v_cpu = zeros(n, nbatch)

    v = similar(u, n, nbatch)
    hm = similar(u, n, nbatch)

    tic = time()
    time_hessian = 0.0

    ExaPF.update!(nlp.model, batch_ad.state, nlp.buffer)
    N = div(n, nbatch, RoundDown)
    for i in 1:N
        fill!(v_cpu, 0.0)
        @inbounds for j in 1:nbatch
            v_cpu[j+(i-1)*nbatch, j] = 1.0
        end
        copyto!(v, v_cpu)
        time_hessian += @elapsed batch_hessprod!(nlp, batch_ad, hm, u, v)

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
        time_hessian += @elapsed batch_hessprod!(nlp, batch_ad, hm, u, v)
        # Keep only last columns in hm
        copyto!(hess, N*nbatch*n + 1, hm, (nbatch - last_batch)*n, n*last_batch)
    end
    total_time = time()  - tic

    return (total_time, time_hessian)
end

function cpu_hessprod!(nlp::ExaPF.ReducedSpaceEvaluator, ∇g, hessvec, u::Array, w::Array)
    @assert nlp.hessians != nothing

    nx = get(nlp.model, NumberOfState())
    nu = get(nlp.model, NumberOfControl())
    ∇gᵤ = nlp.state_jacobian.u.J

    H = nlp.hessians
    # Second order adjoints
    ψ = H.ψ
    z = H.z
    # Two vector products
    tgt = H.tmp_tgt
    hv = H.tmp_hv

    mul!(z, ∇gᵤ, w, -1.0, 0.0)
    LinearAlgebra.ldiv!(∇g, z)

    # Init tangent
    copyto!(tgt, 1, z, 1, nx)
    copyto!(tgt, nx+1, w, 1, nu)

    ## OBJECTIVE HESSIAN
    # TODO: not implemented yet
    ∂fₓ = hv[1:nx]
    ∂fᵤ = hv[nx+1:nx+nu]

    AutoDiff.adj_hessian_prod!(nlp.model, H.state, hv, nlp.buffer, nlp.λ, tgt)
    ∂fₓ .= @view hv[1:nx]
    ∂fᵤ .= @view hv[nx+1:nx+nu]

    # Second order adjoint
    LinearAlgebra.ldiv!(ψ, ∇g', ∂fₓ)

    hessvec .= ∂fᵤ
    mul!(hessvec, transpose(∇gᵤ), ψ, -1.0, 1.0)

    return
end

function cpu_hessian!(nlp::ExaPF.AbstractNLPEvaluator, hess, x)
    n = ExaPF.n_variables(nlp)
    J = nlp.state_jacobian.x.J
    ∇g = lu(J)
    v = similar(x)
    tic = time()
    time_hessian = 0.0
    @inbounds for i in 1:n
        hv = @view hess[:, i]
        fill!(v, 0)
        v[i] = 1.0
        time_hessian += @elapsed cpu_hessprod!(nlp, ∇g, hv, x, v)
    end
    total_time = time() - tic
    return (total_time, time_hessian)
end

end

