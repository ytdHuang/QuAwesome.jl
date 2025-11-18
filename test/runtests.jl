using Test
using TestItemRunner
using Pkg

using QuantumRealm

println("\nStart running tests...\n")
@run_package_tests verbose=true
