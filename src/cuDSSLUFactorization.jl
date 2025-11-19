export cuDSSLUFactorization

# cuDSSLUFactorization
using CUDA, CUDA.CUSPARSE, CUDSS

struct cuDSSLUFactorization <: LinearSolve.SciMLLinearSolveAlgorithm
    refact_lim::Int
    settings::NamedTuple
end

cuDSSLUFactorization(refact_lim::Int = typemax(Int); kwargs...) = cuDSSLUFactorization(refact_lim, NamedTuple(kwargs))

function config_solver(solver, alg)
    for (n, v) in pairs(alg.settings)
        try
            cudss_set(solver, string(n), v)
        catch er
            println("Config $n to $v failed.")
        end
    end
    return nothing
end

function LinearSolve.init_cacheval(
    alg::cuDSSLUFactorization,
    A::CuSparseMatrixCSR,
    b::CuArray,
    u::CuArray,
    Pl,
    Pr,
    maxiters,
    abstol,
    reltol,
    verbose,
    assump,
)
    solver = CudssSolver(A, "G", 'F')
    config_solver(solver, alg)
    cudss("analysis", solver, u, b)

    if cudss_get(solver, "hybrid_memory_mode") == 1
        lim = cudss_get(solver, "hybrid_device_memory_min")
        cudss_set(solver, "hybrid_device_memory_limit", lim)
    end

    cudss("factorization", solver, u, b)

    dims = A.dims
    colVal = A.colVal |> collect
    rowPtr = A.rowPtr |> collect
    nnz = A.nnz

    return (; solver, nnz, dims, colVal, rowPtr, use = 0)
end

LinearSolve.init_cacheval(alg::cuDSSLUFactorization, A, b, u, Pl, Pr, maxiters, abstol, reltol, verbose, assump) =
    throw(
        ArgumentError(
            "cuDSSLUFactorization require the data types to be CUDA.CUSPARSE.CuSparseMatrixCSR and CUDA.CuArray",
        ),
    )

function SciMLBase.solve!(cache::LinearSolve.LinearCache, alg::cuDSSLUFactorization; kwargs...)
    solver, nnz, dims, colVal, rowPtr, use = cache.cacheval

    if cache.isfresh
        A = cache.A

        cudss_update(solver, A)

        _nnz = A.nnz
        _dims = A.dims
        _colVal = colVal
        _rowPtr = rowPtr
        _use = copy(use)

        if use == alg.refact_lim
            new_fact = true
            _use = 0
        elseif nnz != _nnz
            new_fact = true
        elseif dims != A.dims
            new_fact = true
        else
            _colVal = A.colVal |> collect
            _rowPtr = A.rowPtr |> collect
            if _colVal == colVal && rowPtr == _rowPtr
                new_fact = false
            else
                new_fact = true
            end
        end

        if new_fact
            cudss("factorization", solver, cache.u, cache.b)
        else
            cudss("refactorization", solver, cache.u, cache.b)
            _use += 1
        end
        cache.cacheval = (; solver, nnz = _nnz, dims = _dims, colVal = _colVal, rowPtr = _rowPtr, use = _use)
    end

    ldiv!(cache.u, solver, cache.b)

    resid = (cache.A * cache.u - cache.b) |> norm
    resid >= 1e-14 && @warn "residual norm = $resid"

    return SciMLBase.build_linear_solution(alg, cache.u, resid, cache)
end
