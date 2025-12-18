export cuDSSLUFactorization, ResidueWarning

# cuDSSLUFactorization
using CUDA, CUDA.CUSPARSE, CUDSS

@doc raw"""
    cuDSSLUFactorization(
        ϵ::Real = 0,
        refine::Bool = false;
        reuse_symbolic::Bool = true,
        refact_lim::Int = typemax(Int),
        kwargs...,
    )
A GPU-accelerated sparse direct linear solver based on **NVIDIA cuDSS**, wrapped
as a `SciMLLinearSolveAlgorithm`. This solver supports optional diagonal
perturbation, iterative refinement, and controlled reuse of symbolic
factorizations for sequences of structurally identical sparse systems.

## Parameters

- `ϵ`:
    Diagonal perturbation strength.
    The system `A x = b` is replaced by `(A + ϵI)x = b`.
    Used to stabilize nearly singular or poorly pivoted systems.
    Set `ϵ = 0` to disable perturbation.

- `refine`:
    Whether to perform an additional refinement step
    `x ← x + ϵ⋅A⁻² b`.
    This one-step refine is only beneficial when the perturbed system is solved
    sufficiently accurate, use this after you tuned the solver by the `kwargs` in
    the next paragraph.

- `reuse_symbolic`:
    If `true`, symbolic factorization will be reused whenever the sparsity pattern
    (nonzero count and dimensions) is unchanged.
    If `false`, every call triggers full `analysis + factorization`.

- `refact_lim`:
    Maximum number of **numeric refactorizations** allowed before a full symbolic
    analysis is forced.
    Similar to behaviors found in UMFPACK and MKL Pardiso wrappers.

- `kwargs`:
    Any additional keyword arguments are passed directly to `cuDSS` via
    `cudss_set(solver, key, value)` during initialization.
    These control reordering, pivoting, hybrid memory mode, etc.

## Supported cuDSS keyword options

The following options correspond to cuDSS configuration parameters:

- `reordering_alg`: Reordering algorithm for symbolic analysis
    (`"default"`, `"algo1"`, `"algo2"`, `"algo3"`, `"algo4"`, `"algo5"`).
- `factorization_alg`: Algorithm for numeric factorization
    (same set of symbolic strings as above).
- `solve_alg`: Algorithm for the triangular solve phase
    (`"default"`, `"algo1"`, `"algo2"`, `"algo3"`, `"algo4"`, `"algo5"`).
- `use_matching`: Enable (`1`) or disable (`0`) diagonal matching.
- `matching_alg`: Matching algorithm used when `use_matching = 1`.
- `solve_mode`: Matrix modification mode (`"N"` = normal, `"T"` = transpose, `"C"` = adjoint).
- `ir_n_steps`: Number of steps in cuDSS internal iterative refinement.
- `ir_tol`: Tolerance for internal iterative refinement.
- `pivot_type`: Pivoting mode (`'C'` = column, `'R'` = row, `'N'` = none).
- `pivot_threshold`: Threshold used to decide whether to pivot on a given diagonal entry.
- `pivot_epsilon`: Value used to replace singular diagonal entries if pivoting is disabled.
- `max_lu_nnz`: Upper bound on LU nonzero count for unsymmetric matrices.
- `hybrid_memory_mode`:
    `0` = device-only (default),
    `1` = hybrid host/device memory.
- `hybrid_device_memory_limit`: Device memory limit in hybrid mode (bytes).
- `use_cuda_register_memory`: Enable (`1`) or disable (`0`) `cudaHostRegister()` in hybrid mode.
- `host_nthreads`: Number of CPU threads used by cuDSS in hybrid execution.
- `hybrid_execute_mode`:
    `0` = device-only (default),
    `1` = hybrid host/device compute.
- `pivot_epsilon_alg`: Algorithm for computing the pivot epsilon.
- `nd_nlevels`: Number of nested dissection levels for reordering.
- `ubatch_size`: Batch size for uniform batched factorization/solves.
- `ubatch_index`:
    `-1` (default) = process all matrices in the batch,
    otherwise process only a single 0-based matrix index.
- `use_superpanels`: Enable (`1`) or disable (`0`) superpanel optimization.
- `device_count`: Number of CUDA devices to be used.
- `device_indices`: Array of device indices.
- `schur_mode`:
    `0` = Schur complement disabled (default),
    `1` = Schur complement enabled.
- `deterministic_mode`:
    `0` = nondeterministic (faster),
    `1` = deterministic execution.

## Usage Example

```julia
using LinearSolve
using CUDA
using QuAwesome

A = CuSparseMatrixCSR(sprand(10_000, 10_000, 0.001))
b = CUDA.rand(10_000)

alg = cuDSSLUFactorization(1e-12, true; ir_n_steps = 7)

prob = LinearProblem(A, b)
sol  = solve(prob, alg)
```
"""
struct cuDSSLUFactorization <: LinearSolve.SciMLLinearSolveAlgorithm
    ϵ::Real      # diagonal perturbation A -> A + ϵI
    refine::Bool    # whether to perform the extra refinement step
    reuse_symbolic::Bool
    refact_lim::Int  # max number of "refactorization" calls before forcing full analysis+factorization
    settings::NamedTuple
end

LinearSolve.needs_concrete_A(alg::cuDSSLUFactorization) = true

cuDSSLUFactorization(
    ϵ::Real = 0,
    refine::Bool = false;
    reuse_symbolic::Bool = true,
    refact_lim::Int = typemax(Int),
    kwargs...,
) = cuDSSLUFactorization(ϵ, refine, reuse_symbolic, refact_lim, NamedTuple(kwargs))

struct cuDSSLUCache{S,TA}
    solver::S
    nnz::Union{Int64,Int32}
    dims::Tuple{Int,Int}
    use::Int
    work::Union{Nothing,TA} # workspace for refinement, if refine=true
end

function config_solver(solver, alg::cuDSSLUFactorization)
    for (n, v) in pairs(alg.settings)
        try
            cudss_set(solver, string(n), v)
        catch er
            # you might want @warn instead of println in real code
            println("Config $n = $v failed: ", er)
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
    A_ = alg.ϵ == 0 ? A : A + alg.ϵ * FillArrays.Eye{eltype(A)}(size(A, 1))

    solver = CudssSolver(A_, "G", 'F')
    config_solver(solver, alg)

    cudss("analysis", solver, u, b)

    if cudss_get(solver, "hybrid_memory_mode") == 1
        lim = cudss_get(solver, "hybrid_device_memory_min")
        cudss_set(solver, "hybrid_device_memory_limit", lim)
    end

    dims = A_.dims
    nnz = A_.nnz
    work = alg.refine ? similar(b) : nothing

    state = cuDSSLUCache{typeof(solver),typeof(work)}(
        solver,
        nnz,
        dims,
        0,     # use counter
        work,
    )

    return state
end

LinearSolve.init_cacheval(alg::cuDSSLUFactorization, A, b, u, Pl, Pr, maxiters, abstol, reltol, verbose, assump) =
    throw(ArgumentError("cuDSSLUFactorization only supports CuSparseMatrixCSR and CuArray types."))

function SciMLBase.solve!(cache::LinearSolve.LinearCache, alg::cuDSSLUFactorization; kwargs...)
    state = cache.cacheval::cuDSSLUCache
    solver = state.solver
    work = state.work

    if cache.isfresh
        A = cache.A::CuSparseMatrixCSR

        # Rebuild (possibly) shifted matrix with current A
        A_ = alg.ϵ == 0 ? A : A + alg.ϵ * FillArrays.Eye{eltype(A)}(size(A, 1))

        # Update matrix in the solver
        state.use != 0 && cudss_update(solver, A_)

        new_nnz = A_.nnz
        new_dims = A_.dims

        # Decide whether to reuse symbolic factorization or redo analysis
        if state.use == 0
            cudss("factorization", solver, cache.u, cache.b)
            new_use = state.use + 1
        elseif alg.reuse_symbolic && (state.use < alg.refact_lim) && (new_nnz == state.nnz) && (new_dims == state.dims)
            # Reuse pattern: numeric refactorization only
            cudss("refactorization", solver, cache.u, cache.b)
            new_use = state.use + 1
        else
            # New or changed pattern (or reuse disabled / refact_lim reached): full analysis + factorization
            cudss("analysis", solver, cache.u, cache.b)
            cudss("factorization", solver, cache.u, cache.b)
            new_use = 1
        end

        # Update cache state
        cache.cacheval = cuDSSLUCache{typeof(solver),typeof(work)}(solver, new_nnz, new_dims, new_use, work)
    end

    if alg.refine && (alg.ϵ != 0)
        ldiv!(cache.u, solver, cache.b)
        ldiv!(work, solver, cache.u)
        axpy!(alg.ϵ, work, cache.u)
    else
        ldiv!(cache.u, solver, cache.b)
    end

    return SciMLBase.build_linear_solution(alg, cache.u, nothing, cache)
end

@doc raw"""
    ResidueWarning(alg::LinearSolve.SciMLLinearSolveAlgorithm, tol::Real = 1e-14)

residue = norm(A * u - b)

Add a warning layer for a arbitrary `SciMLLinearSolveAlgorithm`.
Every time `solve!` with `alg` gets a bad solution which `residue > tol`, a `@warn` showing `residue` is thrown.
"""
struct ResidueWarning <: LinearSolve.SciMLLinearSolveAlgorithm
    alg::LinearSolve.SciMLLinearSolveAlgorithm
    tol::Real

    ResidueWarning(alg::LinearSolve.SciMLLinearSolveAlgorithm, tol::Real = 1e-14) = new(alg, tol)
end
LinearSolve.needs_concrete_A(alg::ResidueWarning) = false

LinearSolve.init_cacheval(alg::ResidueWarning, A, b, u, Pl, Pr, maxiters, abstol, reltol, verbose, assump) =
    LinearSolve.init_cacheval(alg.alg, A, b, u, Pl, Pr, maxiters, abstol, reltol, verbose, assump)

function SciMLBase.solve!(cache::LinearSolve.LinearCache, alg::ResidueWarning; kwargs...)
    sol = SciMLBase.solve!(cache, alg.alg; kwargs...)
    residue = isnothing(sol.resid) ? norm(cache.A * sol.u - cache.b) : sol.resid
    (residue > alg.tol) && (@warn "residue = $residue"; flush(stderr))
    return SciMLBase.build_linear_solution(alg, sol.u, residue, cache)
end
