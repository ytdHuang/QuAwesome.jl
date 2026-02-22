@testitem "BarycentricAAA" begin
    using HierarchicalEOM

    Γ = 1
    W = 10
    μ = 1
    T = 0.1
    J(ω) = Γ * W^2 / ((ω - μ)^2 + W^2)

    tlist = 0:0.01:10

    # Berycentric AAA fitting
    pts = LogDiscretization(Int(1.0e6), μ, 1000, 2)
    bAAA = Fermion_BarycentricAAA(sigmax(), pts, J.(pts), μ, T; tol = 1.0e-3, verbose = true)
    C1p, C1m = correlation_function(bAAA, tlist)

    # standard Pade
    pade = Fermion_Lorentz_Pade(sigmax(), Γ, μ, W, T, 24)
    C2p, C2m = correlation_function(pade, tlist)

    @test all(isapprox.(real.(C1p), real.(C2p); atol = 1.0e-2))
    @test all(isapprox.(imag.(C1p), imag.(C2p); atol = 1.0e-2))
    @test all(isapprox.(real.(C1m), real.(C2m); atol = 1.0e-2))
    @test all(isapprox.(imag.(C1m), imag.(C2m); atol = 1.0e-2))
end
