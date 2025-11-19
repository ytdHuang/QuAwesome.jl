module QuAwesome

# main packages
import Reexport: @reexport
@reexport using QuantumToolbox
@reexport using HierarchicalEOM

# standard libraries
@reexport using LinearAlgebra
@reexport using SparseArrays

# SciML packages
using LinearSolve

# HierarchicalEOM.jl
include("BarycentricAAA.jl")
include("env_fitting.jl")

# others
include("cuDSSLUFactorization.jl")

end
