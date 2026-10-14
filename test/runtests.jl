using Test
using TestItemRunner
using Pkg

const GROUP_LIST = String["All", "Main", "Code-Quality"]

const GROUP = get(ENV, "GROUP", "All")
(GROUP in GROUP_LIST) || throw(ArgumentError("Unknown GROUP = $GROUP\nThe allowed groups are: $GROUP_LIST\n"))

# Main tests
if (GROUP == "All") || (GROUP == "Main")
    using QuAwesome

    println("\nStart running Main tests...\n")
    @run_package_tests verbose = true
end

########################################################################
# Use traditional Test.jl instead of TestItemRunner.jl for other tests #
########################################################################

const testdir = dirname(@__FILE__)

if (GROUP == "All") || (GROUP == "Code-Quality")
    path = joinpath(dirname(@__FILE__), "code-quality")
    Pkg.activate(path)

    # `cd` into `path` first; otherwise `Pkg.develop` writes an absolute (machine-specific) path to Project.toml
    cd(path) do
        Pkg.develop(PackageSpec(path = relpath(dirname(@__DIR__), path))) # must `develop` otherwise the code coverage for libraries (like CUDA) will not be collected
    end
    Pkg.update()

    using QuAwesome
    using Aqua, JET

    include(joinpath(testdir, "code-quality", "code-quality.jl"))
end
