@testitem "cuDSSLUFactorization" begin
    using LinearSolve
    prob = LinearProblem(zeros(ComplexF64, 2, 2), zeros(ComplexF64, 2))
    @test_throws ArgumentError solve(prob, cuDSSLUFactorization())
end
