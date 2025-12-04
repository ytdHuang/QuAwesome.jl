@testitem "auto_Fermion_Lorentz_Pade" begin
    using HierarchicalEOM

    Γ = 1.0
    W = 10
    μ = 1.0
    T = 0.01
    tlist = 0:0.01:10

    pade_auto, info = auto_Fermion_Lorentz_Pade(sigmax(), Γ, μ, W, T, true; target_nrmse = 1e-4, Nmax = 50)
    C1p, C1m = correlation_function(pade_auto, tlist)

    pade_std = Fermion_Lorentz_Pade(sigmax(), Γ, μ, W, T, info.N)
    C2p, C2m = correlation_function(pade_std, tlist)

    @test all(isapprox.(real.(C1p), real.(C2p); atol = 1e-3))
    @test all(isapprox.(imag.(C1p), imag.(C2p); atol = 1e-3))
    @test all(isapprox.(real.(C1m), real.(C2m); atol = 1e-3))
    @test all(isapprox.(imag.(C1m), imag.(C2m); atol = 1e-3))
end

@testitem "auto_Fermion_Lorentz_Matsubara" begin
    using HierarchicalEOM

    Γ = 1.0
    W = 0.1
    μ = 1.0
    T = 1
    tlist = 0:0.01:10

    mats_auto, info = auto_Fermion_Lorentz_Matsubara(sigmax(), Γ, μ, W, T, true; target_nrmse = 1e-4, Nmax = 50)
    C1p, C1m = correlation_function(mats_auto, tlist)

    mats_std = Fermion_Lorentz_Matsubara(sigmax(), Γ, μ, W, T, info.N)
    C2p, C2m = correlation_function(mats_std, tlist)

    @test all(isapprox.(real.(C1p), real.(C2p); atol = 1e-3))
    @test all(isapprox.(imag.(C1p), imag.(C2p); atol = 1e-3))
    @test all(isapprox.(real.(C1m), real.(C2m); atol = 1e-3))
    @test all(isapprox.(imag.(C1m), imag.(C2m); atol = 1e-3))
end
