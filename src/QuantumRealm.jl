module QuantumRealm

# main packages
import Reexport: @reexport
@reexport using QuantumToolbox
@reexport using HierarchicalEOM

# for cuDSSLUFactorization
import CUDA: has_cuda_gpu
if CUDA.has_cuda_gpu()
    using CUDA, CUDA.CUSPARSE, CUDSS
    include("cuDSSLUFactorization.jl")
end

# standard libraries
using LinearAlgebra
using SparseArrays

# SciML packages
using LinearSolve

include("BarycentricAAA.jl")

end
