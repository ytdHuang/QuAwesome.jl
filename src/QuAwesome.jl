module QuAwesome

# main packages
using QuantumToolbox
using HierarchicalEOM

# standard libraries
using LinearAlgebra
using SparseArrays

# SciML packages
using LinearSolve

# HierarchicalEOM.jl
include("BarycentricAAA.jl")
include("env_fitting.jl")

# others
include("linear_solver.jl")
include("bravyi_kitaev.jl")

end
