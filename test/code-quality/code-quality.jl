@testset "Code quality" verbose = true begin
    @testset "Aqua.jl" begin
        Aqua.test_all(QuantumRealm; ambiguities = false, unbound_args = false)
    end

    @testset "JET.jl" begin
        JET.test_package(QuantumRealm; target_defined_modules = true, ignore_missing_comparison = true)
    end
end
