@testitem "bravyi_kitaev" begin
    using QuantumToolbox
    using LinearAlgebra

    @test fdestroy(1, 1, :JW) == fdestroy(1, 1, :BK)

    N = 7
    vac = to_sparse(kron(fill(basis(2, 0), N)...))
    Cs = map(i -> fdestroy(N, i, :BK), 1:N)
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
end
