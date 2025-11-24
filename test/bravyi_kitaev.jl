@testitem "bravyi_kitaev" begin
    @test fdestroy(1, 1, :JW) == fdestroy(1, 1, :BK)

    N = 7
    vac = kron(fill(basis(2, 0), N)...) |> sparse
    Cs = map(i -> fdestroy(N, i, :BK), 1:N)

    anticomm(A, B) = (A * B) + (B * A)

    for i in 1:N
        Ci = Cs[i]
        @test isempty((Ci * vac).data.nzind)

        @test anticomm(Ci, Ci').data == I
        @test norm(anticomm(Ci, Ci).data) == 0

        for j in (i+1):N
            Cj = Cs[j]
            @test norm(anticomm(Ci, Cj).data) == 0
            @test norm(anticomm(Ci, Cj').data) == 0
        end
    end
end
