import QuantumToolbox: fdestroy

lsb(i::Int) = i & (-i)

function U_set(j::Int, N::Int)
    inds = Int[]
    k = j
    while k <= N
        push!(inds, k)
        k += lsb(k)
    end
    return inds
end

function P_set(j::Int)
    inds = Int[]
    k = j
    while k > 0
        push!(inds, k)
        k -= lsb(k)
    end
    return inds
end

function F_qubit(j::Int, N::Int)
    U = U_set(j, N)
    isempty(U) ? (return []) : (return [minimum(U)])
    # return minimum(U)  # or maximum(U), depending on convention; just be consistent
end

function bk_sets(j::Int, N::Int)
    U = U_set(j, N)
    P = P_set(j)           # you can experiment with P_set(j-1) vs P_set(j)
    F = F_qubit(j, N)         # occupation qubit choice; keep this consistent everywhere
    return P, U, F
end

function bkdestroy(N::Int, j::Int)
    P, U, F = bk_sets(j, N)

    X_sites = setdiff(U, F)
    Z_sites = setdiff(P, U)

    # Build the string S = ( ⊗_{k∈X_sites} X_k ) ( ⊗_{k∈Z_sites} Z_k )
    S = qeye(2^N, dims = Tuple(fill(2, N)))  # start with identity
    Sdim = S.dims
    if !isempty(X_sites)
        S *= multisite_operator(Sdim, map(i -> i => sigmax(), X_sites)...)
    end
    if !isempty(Z_sites)
        S *= multisite_operator(Sdim, map(i -> i => sigmaz(), Z_sites)...)
    end

    # Now attach the local X / Y on the F qubit
    return multisite_operator(Sdim, F => sigmam()) * S
end

#################################################

QuantumToolbox.fdestroy(N, j, mode::Symbol = :JW) = QuantumToolbox.fdestroy(N, j, Val(mode))
QuantumToolbox.fdestroy(N, j, ::Val{:JW}) = QuantumToolbox.fdestroy(N, j)
QuantumToolbox.fdestroy(N, j, ::Val{:BK}) = bkdestroy(N, j)'
