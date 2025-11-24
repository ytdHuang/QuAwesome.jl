import QuantumToolbox: fdestroy

function U_set(N::Int, j::Int)
    p = j
    out = Int[]
    while p < N
        if p != j       # exclude qubit j itself
            push!(out, p)
        end
        p = p | (p + 1)
    end
    return out
end

function P_set(j::Int)
    p = j-1         # prefix parity = modes < i
    out = Int[]
    while p >= 0
        push!(out, p)
        p = (p & (p+1)) - 1    # drop the LSB block
    end
    return out
end

function F_set(j::Int)
    out = Int[]
    k = 0
    while ((j >> k) & 1) == 1     # as long as bit k of i is 1
        push!(out, j & ~(1 << k)) # zero out bit k
        k += 1
    end
    return out
end

function bkdestroy(N::Int, j::Int)
    U = U_set(N, j-1)
    P = P_set(j-1)
    F = F_set(j-1)
    R = setdiff(P, F)

    dims = Tuple(fill(2, N))
    Id = qeye(2^N, dims = dims)

    XU = isempty(U) ? Id : multisite_operator(dims, map(i -> i+1 => sigmax(), U)...)
    ZR = isempty(R) ? Id : multisite_operator(dims, map(i -> i+1 => sigmaz(), R)...)
    ZP = isempty(P) ? Id : multisite_operator(dims, map(i -> i+1 => sigmaz(), P)...)


    Xj = multisite_operator(dims, j => sigmax())
    Yj = multisite_operator(dims, j => sigmay())

    return 1/2 * XU * (Xj * ZP + 1im * Yj * ZR)
end

#################################################

@doc raw"""
    QuantumToolbox.fdestroy(N::Int, j::Int, mode::Symbol = :JW)

- `N`: Number of fermionic modes
- `j`: The `j`-th mode destroy operator to return
- `mode`: `:JW` for Jordan-Wigner and `:BK` for Bravyi-Kitaev
"""
QuantumToolbox.fdestroy(N::Int, j::Int, mode::Symbol = :JW) = QuantumToolbox.fdestroy(N, j, Val(mode))
QuantumToolbox.fdestroy(N::Int, j::Int, ::Val{:JW}) = QuantumToolbox.fdestroy(N, j)
QuantumToolbox.fdestroy(N::Int, j::Int, ::Val{:BK}) = bkdestroy(N, j)
