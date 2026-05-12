@testitem "bravyi_kitaev" begin
    using QuantumToolbox
    using LinearAlgebra

    @test fdestroy(1, 1, Val(:JW)) == fdestroy(1, 1, Val(:BK))

    N = 7
    vac = to_sparse(kron(fill(basis(2, 0), N)...))
    Cs = map(i -> fdestroy(N, i, Val(:BK)), 1:N)
    for i in 1:N
        Ci = Cs[i]
        @test isempty((Ci * vac).data.nzind)

        @test commutator(Ci, Ci', anti = true).data == I
        @test norm(commutator(Ci, Ci, anti = true).data) == 0

        for j in (i + 1):N
            Cj = Cs[j]
            @test norm(commutator(Ci, Cj, anti = true).data) == 0
            @test norm(commutator(Ci, Cj', anti = true).data) == 0
        end
    end

    X = sigmax()
    Y = sigmay()
    Z = sigmaz()
    i = qeye(2)
    d = [
        0.5 * tensor(X, X, i, X) + 0.5im * tensor(Y, X, i, X),
        0.5 * tensor(Z, X, i, X) + 0.5im * tensor(i, Y, i, X),
        0.5 * tensor(i, Z, X, X) + 0.5im * tensor(i, Z, Y, X),
        0.5 * tensor(i, Z, Z, X) + 0.5im * tensor(i, i, i, Y),
    ]
    @test all([fdestroy(4, i, Val(:BK)) == d[i] for i in 1:4])
end
