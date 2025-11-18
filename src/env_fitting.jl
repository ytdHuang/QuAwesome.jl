export Boson_Environment_CurveFit

import QuantumToolbox: _float_type
import HierarchicalEOM: _check_bosonic_coupling_operator
import CurveFit: CurveFitProblem, ExpSumFitAlgorithm

@doc raw"""
    _nrmse(predict::AbstractVector{<:Real}, actual::AbstractVector{<:Real})

Normalized (by range of `actual`) root-mean-square error
"""
function _nrmse(predict::AbstractVector{<:Real}, actual::AbstractVector{<:Real})
    # @assert length(predict) == length(actual) "Vectors must be the same length!"
    return sqrt(sum(abs2, predict .- actual) / length(predict)) / (maximum(actual) - minimum(actual))
end

function iterated_fit(
    model::AbstractVector{<:Real},
    xdata::AbstractVector{<:Real};
    target_rmse::Real = 2e-5,
    Nmin::Int = 1,
    Nmax::Int = 10,
    m::Int = 1,
)
    ftype = _float_type(model)
    Model = convert(Vector{ftype}, model)
    Xdata = convert(Vector{ftype}, xdata)

    N = Nmin

    nrmse, sol = Inf, nothing
    prob = CurveFitProblem(Xdata, Model)

    while (N<=Nmax) && (nrmse > target_rmse)
        sol = solve(prob, ExpSumFitAlgorithm(; n = N, m = m, withconst = false))
        predict = sol(Xdata)
        nrmse = _nrmse(predict, Model)
        N += 1
    end

    return nrmse, sol
end

@doc raw"""
    Boson_Environment_CurveFit(
        op::QuantumObject,
        cf_data::AbstractVector{<:Number},
        tlist::AbstractVector{<:Real};
        target_rmse::Real = 2e-5,
        Nr_min::Int = 1,
        Ni_min::Int = 1,
        Nr_max::Int = 10,
        Ni_max::Int = 10,
        m::Int = 1,
        combine::Bool = true,
    )

Constructing bosonic bath by curve fitting the two-time correlation function data.

# Parameters
- `op` : The system coupling operator, must be Hermitian and, for fermionic systems, even-parity to be compatible with charge conservation.
- `cf_data::AbstractVector{<:Number}`: The two-time correlation function data to be fitted.
- `tlist::AbstractVector{<:Real}`: The time points corresponding to the correlation function data.
- `target_rmse::Real`: The target normalized root-mean-square error for the fitting. Default to `2e-5`.
- `Nr_min::Int`: The minimum number of exponentials for fitting the real part of the correlation function. Default to `1`.
- `Ni_min::Int`: The minimum number of exponentials for fitting the imaginary part of the correlation function. Default to `1`.
- `Nr_max::Int`: The maximum number of exponentials for fitting the real part of the correlation function. Default to `10`.
- `Ni_max::Int`: The maximum number of exponentials for fitting the imaginary part of the correlation function. Default to `10`.
- `m::Int`: Interpolation order used in the fitting algorithm. Default to `1`.
- `combine::Bool`: Whether to combine the real and imaginary parts into a single bath representation. Default to `true`.
"""
function Boson_Environment_CurveFit(
    op::QuantumObject,
    cf_data::AbstractVector{<:Number},
    tlist::AbstractVector{<:Real};
    target_rmse::Real = 2e-5,
    Nr_min::Int = 1,
    Ni_min::Int = 1,
    Nr_max::Int = 10,
    Ni_max::Int = 10,
    m::Int = 1,
    combine::Bool = true,
)
    nrmse_real, sol_real =
        iterated_fit(real(cf_data), tlist; target_rmse = target_rmse, Nmin = Nr_min, Nmax = Nr_max, m = m)
    nrmse_imag, sol_imag =
        iterated_fit(imag(cf_data), tlist; target_rmse = target_rmse, Nmin = Ni_min, Nmax = Ni_max, m = m)

    return BosonBath(
        _check_bosonic_coupling_operator(op),
        sol_real.u.p,
        -sol_real.u.λ,
        sol_imag.u.p,
        -sol_imag.u.λ,
        combine = combine,
    ),
    (nrmse_real = nrmse_real, nrmse_imag = nrmse_imag, sol_real = sol_real, sol_imag = sol_imag)
end
