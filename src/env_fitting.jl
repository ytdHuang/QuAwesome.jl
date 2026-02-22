export Boson_Environment_CurveFit
export auto_Fermion_Lorentz_Matsubara, auto_Fermion_Lorentz_Pade

import QuantumToolbox: _float_type
import HierarchicalEOM: _check_bosonic_coupling_operator
import CurveFit: CurveFitProblem, ExpSumFitAlgorithm
import HierarchicalEOM: _fermion_lorentz_pade_param, _fermion_lorentz_matsubara_param, FermionBath

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
        target_rmse::Real = 2.0e-5,
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

    while (N <= Nmax) && (nrmse > target_rmse)
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
        target_rmse::Real = 2.0e-5,
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

expsum(η, γ, tlist) = begin
    c(t) =
        sum(zip(η, γ)) do (e, g)
        return e * exp(-g * t)
    end
    c.(tlist)
end

function _auto_Fermion_Lorentz_Pade(
        op::QuantumObject,
        λ::T,
        μ::U,
        W::V,
        kT::S;
        Nmin::Int = 1,
        Nmax::Int = 100,
        Nref::Int = 500,
        tlist::AbstractVector{<:Real} = 0:0.01:20,
        target_nrmse::Union{<:Real} = 1.0e-5,
    ) where {T <: Real, U <: Real, V <: Real, S <: Real}
    ηp, γp = _fermion_lorentz_pade_param(1, λ, μ, W, kT, Nref)
    ηm, γm = _fermion_lorentz_pade_param(-1, λ, μ, W, kT, Nref)
    cpref = expsum(ηp, γp, tlist)
    cmref = expsum(ηm, γm, tlist)

    nrmse = Inf
    N = Nmin - 1

    ηp_, γp_ = nothing, nothing
    while (N <= Nmax) && (nrmse > target_nrmse)
        N += 1
        ηp_, γp_ = _fermion_lorentz_pade_param(1, λ, μ, W, kT, N)
        cp = expsum(ηp_, γp_, tlist)
        nrmse = max(_nrmse(real(cp), real(cpref)), _nrmse(imag(cp), imag(cpref)))
    end

    ηm_, γm_ = _fermion_lorentz_pade_param(-1, λ, μ, W, kT, N)
    cm = expsum(ηm_, γm_, tlist)
    nrmse = max(_nrmse(real(cm), real(cmref)), _nrmse(imag(cm), imag(cmref)))
    fresh = true
    while (N <= Nmax) && (nrmse > target_nrmse)
        fresh = false
        N += 1
        ηm_, γm_ = _fermion_lorentz_pade_param(-1, λ, μ, W, kT, N)
        cm = expsum(ηm_, γm_, tlist)
        nrmse = max(_nrmse(real(cm), real(cmref)), _nrmse(imag(cm), imag(cmref)))
    end

    if !fresh
        ηp_, γp_ = _fermion_lorentz_pade_param(1, λ, μ, W, kT, N)
    end

    return FermionBath(op, ηp_, γp_, ηm_, γm_), (; N, nrmse)
end

auto_Fermion_Lorentz_Pade(
    op::QuantumObject,
    λ::T,
    μ::U,
    W::V,
    kT::S,
    info::Val{true};
    kwargs...,
) where {T <: Real, U <: Real, V <: Real, S <: Real} = _auto_Fermion_Lorentz_Pade(op, λ, μ, W, kT; kwargs...)

auto_Fermion_Lorentz_Pade(
    op::QuantumObject,
    λ::T,
    μ::U,
    W::V,
    kT::S,
    info::Val{false};
    kwargs...,
) where {T <: Real, U <: Real, V <: Real, S <: Real} = _auto_Fermion_Lorentz_Pade(op, λ, μ, W, kT; kwargs...)[1]

@doc raw"""
    auto_Fermion_Lorentz_Pade(
        op::QuantumObject, λ::T, μ::U, W::V, kT::S, info::Bool=false;
        Nmin::Int=1,
        Nmax::Int=100,
        Nref::Int=500,
        tlist::AbstractVector{<:Real}=0:0.01:20, 
        target_nrmse::=1e-5,
    ) where {T<:Real, U<:Real, V<:Real, S<:Real}

Automatically find the number `N` for `Fermion_Lorentz_Pade` by comparing 
the normalized rmse with the correlation functions with `Nref` term.

See the doc string for `Fermion_Lorentz_Pade` for more detail about other arguments.
"""
auto_Fermion_Lorentz_Pade(
    op::QuantumObject,
    λ::T,
    μ::U,
    W::V,
    kT::S,
    info::Bool = false;
    kwargs...,
) where {T <: Real, U <: Real, V <: Real, S <: Real} = auto_Fermion_Lorentz_Pade(op, λ, μ, W, kT, Val(info); kwargs...)

function _auto_Fermion_Lorentz_Matsubara(
        op::QuantumObject,
        λ::T,
        μ::U,
        W::V,
        kT::S;
        Nmin::Int = 1,
        Nmax::Int = 100,
        Nref::Int = 500,
        tlist::AbstractVector{<:Real} = 0:0.01:20,
        target_nrmse::Union{<:Real} = 1.0e-5,
    ) where {T <: Real, U <: Real, V <: Real, S <: Real}
    ηp, γp = _fermion_lorentz_matsubara_param(1, λ, μ, W, kT, Nref)
    ηm, γm = _fermion_lorentz_matsubara_param(-1, λ, μ, W, kT, Nref)
    cpref = expsum(ηp, γp, tlist)
    cmref = expsum(ηm, γm, tlist)

    nrmse = Inf
    N = Nmin - 1

    ηp_, γp_ = nothing, nothing
    while (N <= Nmax) && (nrmse > target_nrmse)
        N += 1
        ηp_, γp_ = _fermion_lorentz_matsubara_param(1, λ, μ, W, kT, N)
        cp = expsum(ηp_, γp_, tlist)
        nrmse = max(_nrmse(real(cp), real(cpref)), _nrmse(imag(cp), imag(cpref)))
    end

    ηm_, γm_ = _fermion_lorentz_matsubara_param(-1, λ, μ, W, kT, N)
    cm = expsum(ηm_, γm_, tlist)
    nrmse = max(_nrmse(real(cm), real(cmref)), _nrmse(imag(cm), imag(cmref)))
    fresh = true
    while (N <= Nmax) && (nrmse > target_nrmse)
        fresh = false
        N += 1
        ηm_, γm_ = _fermion_lorentz_matsubara_param(-1, λ, μ, W, kT, N)
        cm = expsum(ηm_, γm_, tlist)
        nrmse = max(_nrmse(real(cm), real(cmref)), _nrmse(imag(cm), imag(cmref)))
    end

    if !fresh
        ηp_, γp_ = _fermion_lorentz_matsubara_param(1, λ, μ, W, kT, N)
    end

    return FermionBath(op, ηp_, γp_, ηm_, γm_), (; N, nrmse)
end

auto_Fermion_Lorentz_Matsubara(
    op::QuantumObject,
    λ::T,
    μ::U,
    W::V,
    kT::S,
    info::Val{true};
    kwargs...,
) where {T <: Real, U <: Real, V <: Real, S <: Real} = _auto_Fermion_Lorentz_Matsubara(op, λ, μ, W, kT; kwargs...)

auto_Fermion_Lorentz_Matsubara(
    op::QuantumObject,
    λ::T,
    μ::U,
    W::V,
    kT::S,
    info::Val{false};
    kwargs...,
) where {T <: Real, U <: Real, V <: Real, S <: Real} = _auto_Fermion_Lorentz_Matsubara(op, λ, μ, W, kT; kwargs...)[1]

@doc raw"""
    auto_Fermion_Lorentz_Matsubara(
        op::QuantumObject, λ::T, μ::U, W::V, kT::S, info::Bool=false;
        Nmin::Int=1,
        Nmax::Int=100,
        Nref::Int=500,
        tlist::AbstractVector{<:Real}=0:0.01:20, 
        target_nrmse::=1e-5,
    ) where {T<:Real, U<:Real, V<:Real, S<:Real}

Automatically find the `N` for `Fermion_Lorentz_Matsubara` by comparing 
the normalized rmse with the correlation functions with `Nref` term.

if `info = true`, return `(; N, nrmse)` after the `FermionBath`

See the doc string for `Fermion_Lorentz_Matsubara` for more detail about other arguments.
"""
auto_Fermion_Lorentz_Matsubara(
    op::QuantumObject,
    λ::T,
    μ::U,
    W::V,
    kT::S,
    info::Bool = false;
    kwargs...,
) where {T <: Real, U <: Real, V <: Real, S <: Real} = auto_Fermion_Lorentz_Matsubara(op, λ, μ, W, kT, Val(info); kwargs...)
