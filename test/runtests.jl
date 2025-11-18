using Test
using TestItemRunner
using Pkg

const GROUP_LIST = String["All", "Core", "Code-Quality"]

const GROUP = get(ENV, "GROUP", "All")
(GROUP in GROUP_LIST) || throw(ArgumentError("Unknown GROUP = $GROUP\nThe allowed groups are: $GROUP_LIST\n"))

# Core tests
if (GROUP == "All") || (GROUP == "Core")
    using QuantumRealm

    println("\nStart running Core tests...\n")
    @run_package_tests verbose=true
end

########################################################################
# Use traditional Test.jl instead of TestItemRunner.jl for other tests #
########################################################################

const testdir = dirname(@__FILE__)

if (GROUP == "All") || (GROUP == "Code-Quality")
    Pkg.activate("code-quality")
    Pkg.develop(PackageSpec(path = dirname(@__DIR__)))
    Pkg.update()

    using QuantumRealm
    using Aqua, JET

    include(joinpath(testdir, "code-quality", "code-quality.jl"))
end
