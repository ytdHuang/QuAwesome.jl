export Boson_BarycentricAAA, Fermion_BarycentricAAA
export LogDiscretization

import BaryRational: aaa, prz

_nb(ω::Real, β::Real) = (exp(β * ω) - 1)^(-1)
_nf(ω::Real, μ::Real, β::Real) = (exp(β * (ω - μ)) + 1)^(-1)

@doc raw"""
    LogDiscretization(N, z0, D, Λ)

Generate logarithmically discretized points around `z0` within the range of `[z0 - D, z0 + D]`.

The discretization is controlled by the parameter `Λ` (`Λ > 1`). The points are given as

```math
\{z_0 - D, z_0 - D\Lambda^{-1}, z_0 - D\Lambda^{-2}, \cdots, z_0 - D\Lambda^{-N}, z_0, z_0 + D\Lambda^{-N}, \cdots, z_0 + D\Lambda^{-2}, z_0 + D\Lambda^{-1}, z_0 + D\}.
```
"""
function LogDiscretization(N::Real, z0::Real, D::Real, Λ::Real = 2)
    if Int(N) < 0
        error("N must be greater or equal to 0")
    end
    if Λ <= 1
        error("Λ must be greater than 1")
    end

    return Float64[
        z0 .- D .* float(Λ) .^ [0:-1.0:(-N);];
        z0;
        z0 .+ D .* float(Λ) .^ [(-N):1.0:0;];
    ]
end

@doc raw"""
    Boson_BarycentricAAA(op, sample_pts, Jlist, kT; tol, mmax, verbose)
Constructing bosonic bath with the pole structure obtained from Barycentric AAA algorithm

The bosonic bath two-time correlation function can be given as
```math
C(t_1, t_2)=\frac{1}{2\pi}\int_{0}^{\infty} d\omega J(\omega)\left[n(\omega)e^{i\omega (t_1-t_2)}+(n(\omega)+1)e^{-i\omega (t_1-t_2)}\right],
```
where ``J(\omega)=2\pi\Sigma_k |g_k|^2 \delta(\omega-\omega_k)`` is the spectral density of the bath and ``n(\omega)=\{\exp(\omega/k_B T)-1\}^{-1}`` represents the Bose-Einstein distribution.
  
Due to the numerical fitting method (Barycentric AAA algorithm), it is better to consider an odd spectral density function ``\tilde{J}(\omega)``, such that, 
```math
\tilde{J}(-\omega)=-\tilde{J}(\omega)=-J(\omega) ~~\forall~~\omega \geq 0.
```

Because the integrand in ``C(t_1, t_2)`` becomes an even function, we can extend the integration interval to `(-\infty, \infty)` and divide the integration by ``2``, namely
```math
C(t_1, t_2)=\frac{1}{4\pi}\int_{-\infty}^{\infty} d\omega \tilde{J}(\omega)\left[n(\omega)e^{i\omega (t_1-t_2)}+(n(\omega)+1)e^{-i\omega (t_1-t_2)}\right],
```

# Parameters
- `op` : The system coupling operator, must be Hermitian and, for fermionic systems, even-parity to be compatible with charge conservation.
- `sample_pts::AbstractVector`: The sample points (list of ``\omega``) for Barycentric AAA algorithm.
- `Jlist::AbstractVector`: The list of the value of (anti-symmetric) spectral density ``\tilde{J}(\omega)`` corresponding to the sample points.
- `kT::Real`: The product of the Boltzmann constant ``k`` and the absolute temperature ``T`` of the bath.
- `tol`: relative tolerance for AAA algorithm. Default to `1e-13`.
- `mmax`: max type is `(mmax-1, mmax-1)` for AAA algorithm. Default to `100`.
- `verbose::Bool`: To display verbose output during the process or not. Defaults to `false`.

# Returns
- `bath::BosonBath` : a bosonic bath object with describes the interaction between system and bosonic bath
"""
function Boson_BarycentricAAA(
        op::QuantumObject,
        sample_pts::AbstractVector,
        Jlist::AbstractVector,
        kT::Real;
        tol::Real = 1.0e-13,
        mmax::Int = 100,
        verbose::Bool = false,
    )
    β = 1 / kT
    Z = Float64[sample_pts;]
    JZ = Float64[Jlist;]
    if length(Z) != length(JZ)
        error("The length of \`Jlist\` should be same as \`sample_pts\`.")
    end

    nZ = Float64[_nb.(Z, β);]

    # find Barycentric representation
    if verbose
        print("\nFinding barycentric representation for J(ω)")
        print("\n-------------------------------------------")
    end
    BJ = aaa(Z, JZ; tol = tol, mmax = mmax, verbose = verbose)
    if verbose
        print("[DONE]\n")
        print("\nFinding barycentric representation for n(ω)")
        print("\n-------------------------------------------")
    end
    Bn = aaa(Z, nZ; tol = tol, mmax = mmax, verbose = verbose)
    if verbose
        print("[DONE]\n")
    end

    # pole, residue, zero
    ωJ, RJ, ZJ = prz(BJ)
    ωn, Rn, Zn = prz(Bn)
    if verbose
        print("\nNumber of poles found for J(ω): $(length(ωJ))")
        print("\nNumber of poles found for n(ω): $(length(ωn))\n")
    end

    η_real = ComplexF64[]
    γ_real = ComplexF64[]
    η_imag = ComplexF64[]
    γ_imag = ComplexF64[]

    # poles from spectral density
    for j in 1:length(ωJ)
        ωj_im = imag(ωJ[j])

        # poles in upper plane
        if ωj_im > 0
            # exponents for real part of correlation func.
            push!(η_real, 0.25im * RJ[j] * (2 * Bn(ωJ[j]) + 1))
            push!(γ_real, -1im * ωJ[j])

            # exponents for imaginary part of correlation func.
            push!(η_imag, -0.25 * RJ[j])
            push!(γ_imag, -1im * ωJ[j])

            # poles in lower plane
        elseif ωj_im < 0
            # exponents for real part of correlation func.
            push!(η_real, -0.25im * RJ[j] * (2 * Bn(ωJ[j]) + 1))
            push!(γ_real, 1im * ωJ[j])

            # exponents for imaginary part of correlation func.
            push!(η_imag, -0.25 * RJ[j])
            push!(γ_imag, 1im * ωJ[j])
        end
    end

    # poles from Bose-Einstein Distrib.
    for k in 1:length(ωn)
        ωk_im = imag(ωn[k])

        # poles in upper plane
        if ωk_im > 0
            push!(η_real, 0.5im * BJ(ωn[k]) * Rn[k])
            push!(γ_real, -1im * ωn[k])

            # poles in lower plane
        elseif ωk_im < 0
            push!(η_real, -0.5im * BJ(ωn[k]) * Rn[k])
            push!(γ_real, 1im * ωn[k])
        end
    end

    return BosonBath(op, η_real, γ_real, η_imag, γ_imag)
end

@doc raw"""
    Fermion_BarycentricAAA(op, sample_pts, Jlist, μ, kT; tol, mmax, verbose)
Constructing fermionic bath with the pole structure obtained from Barycentric AAA algorithm

The fermionic bath two-time correlation function can be given as
```math
C^{\nu}(t_{1},t_{2})=\frac{1}{2\pi}\int_{-\infty}^{\infty} d\omega J(\omega)\left[\frac{1-\nu}{2}+\nu n(\omega)\right]e^{\nu i\omega (t_{1}-t_{2})}.
```
where ``J(\omega)=2\pi\Sigma_k |g_k|^2 \delta(\omega-\omega_k)`` is the spectral density of the bath and ``n(\omega)=\{\exp[(\omega-\mu)/k_B T]+1\}^{-1}`` represents the Fermi-Dirac distribution (with chemical potential ``\mu``). Here, ``\nu=+`` and ``\nu=-`` denotes the absorption and emission process of the fermionic system, respectively.

# Parameters
- `op` : The system annihilation operator according to the system-fermionic-bath interaction.
- `sample_pts::AbstractVector`: The sample points (list of ``\omega``) for Barycentric AAA algorithm.
- `Jlist::AbstractVector`: The list of the value of spectral density ``J(\omega)`` corresponding to the sample points.
- `μ::Real`: The chemical potential of the bath.
- `kT::Real`: The product of the Boltzmann constant ``k`` and the absolute temperature ``T`` of the bath.
- `tol`: relative tolerance for AAA algorithm. Default to `1e-13`.
- `mmax`: max type is `(mmax-1, mmax-1)` for AAA algorithm. Default to `100`.
- `verbose::Bool`: To display verbose output during the process or not. Defaults to `false`.

# Returns
- `bath::FermionBath` : a fermionic bath object with describes the interaction between system and fermionic bath
"""
function Fermion_BarycentricAAA(
        op::QuantumObject,
        sample_pts::AbstractVector,
        Jlist::AbstractVector,
        μ::Real,
        kT::Real;
        tol::Real = 1.0e-13,
        mmax::Int = 100,
        verbose::Bool = false,
    )
    β = 1 / kT
    Z = Float64[sample_pts;]
    JZ = Float64[Jlist;]
    if length(Z) != length(JZ)
        error("The length of \`Jlist\` should be same as \`sample_pts\`.")
    end

    nZ = Float64[_nf.(Z, μ, β);]

    # find Barycentric representation
    if verbose
        print("\nFinding barycentric representation for J(ω)")
        print("\n-------------------------------------------")
    end
    BJ = aaa(Z, JZ; tol = tol, mmax = mmax, verbose = verbose)
    if verbose
        print("[DONE]\n")
        print("\nFinding barycentric representation for n(ω)")
        print("\n-------------------------------------------")
    end
    Bn = aaa(Z, nZ; tol = tol, mmax = mmax, verbose = verbose)
    if verbose
        print("[DONE]\n")
    end

    # pole, residue, zero
    ωJ, RJ, ZJ = prz(BJ)
    ωn, Rn, Zn = prz(Bn)
    if verbose
        print("\nNumber of poles found for J(ω): $(length(ωJ))")
        print("\nNumber of poles found for n(ω): $(length(ωn))\n")
    end

    η_ab = ComplexF64[]
    γ_ab = ComplexF64[]
    η_em = ComplexF64[]
    γ_em = ComplexF64[]

    # poles from spectral density
    for j in 1:length(ωJ)
        ωj_im = imag(ωJ[j])

        # poles in upper plane
        if ωj_im > 0
            push!(η_ab, 1im * RJ[j] * Bn(ωJ[j]))
            push!(γ_ab, -1im * ωJ[j])

            # poles in lower plane
        elseif ωj_im < 0
            push!(η_em, -1im * RJ[j] * (1 - Bn(ωJ[j])))
            push!(γ_em, 1im * ωJ[j])
        end
    end

    # poles from Fermi-Dirac Distrib.
    for k in 1:length(ωn)
        ωk_im = imag(ωn[k])

        # poles in upper plane
        if ωk_im > 0
            push!(η_ab, 1im * BJ(ωn[k]) * Rn[k])
            push!(γ_ab, -1im * ωn[k])

            # poles in lower plane
        elseif ωk_im < 0
            push!(η_em, 1im * BJ(ωn[k]) * Rn[k])
            push!(γ_em, 1im * ωn[k])
        end
    end

    return FermionBath(op, η_ab, γ_ab, η_em, γ_em)
end
