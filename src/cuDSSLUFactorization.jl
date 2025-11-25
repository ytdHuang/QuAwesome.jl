export cuDSSLUFactorization

# cuDSSLUFactorization
using CUDA, CUDA.CUSPARSE, CUDSS

@doc raw"""
    cuDSSLUFactorization(refact_lim::Int = typemax(Int); kwargs...)

`refact_lim`: reuse symbolic factorization for how many times.

The available keyword arguments are:
- `reordering_alg`: Algorithm for the reordering phase (`"default"`, `"algo1"`, `"algo2"`, `"algo3"`, `"algo4"`, or `"algo5"`);
- `factorization_alg`: Algorithm for the factorization phase (`"default"`, `"algo1"`, `"algo2"`, `"algo3"`, `"algo4"`, or `"algo5"`);
- `solve_alg`: Algorithm for the solving phase (`"default"`, `"algo1"`, `"algo2"`, `"algo3"`, `"algo4"`, or `"algo5"`);
- `use_matching`: A flag to enable (`1`) or disable (`0`) the matching;
- `matching_alg`: Algorithm for the matching;
- `solve_mode`: Potential modificator on the system matrix (transpose or adjoint);
- `ir_n_steps`: Number of steps during the iterative refinement;
- `ir_tol`: Iterative refinement tolerance;
- `pivot_type`: Type of pivoting (`'C'`, `'R'` or `'N'`);
- `pivot_threshold`: Pivoting threshold which is used to determine if digonal element is subject to pivoting;
- `pivot_epsilon`: Pivoting epsilon, absolute value to replace singular diagonal elements;
- `max_lu_nnz`: Upper limit on the number of nonzero entries in LU factors for non-symmetric matrices;
- `hybrid_memory_mode`: Hybrid memory mode -- `0` (default = device-only) or `1` (hybrid = host/device);
- `hybrid_device_memory_limit`: User-defined device memory limit (number of bytes) for the hybrid memory mode;
- `use_cuda_register_memory`: A flag to enable (`1`) or disable (`0`) usage of `cudaHostRegister()` by the hybrid memory mode;
- `host_nthreads`: Number of threads to be used by cuDSS in multi-threaded mode;
- `hybrid_execute_mode`: Hybrid execute mode -- `0` (default = device-only) or `1` (hybrid = host/device);
- `pivot_epsilon_alg`: Algorithm for the pivot epsilon calculation;
- `nd_nlevels`: Minimum number of levels for the nested dissection reordering;
- `ubatch_size`: The number of matrices in a uniform batch of systems to be processed by cuDSS;
- `ubatch_index`: Use `-1` (default) to process all matrices in the uniform batch, or a 0-based index to process a single matrix during the factorization or solve phase;
- `use_superpanels`: Use superpanel optimization -- `1` (default = enabled) or `0` (disabled);
- `device_count`: Device count in case of multiple device;
- `device_indices`: A list of device indices as an integer array;
- `schur_mode`: Schur complement mode -- `0` (default = disabled) or `1` (enabled);
- `deterministic_mode`: Enable deterministic mode -- `0` (default = disabled) or `1` (enabled).
"""
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
