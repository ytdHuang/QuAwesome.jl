@testset "Code quality" verbose = true begin
    @testset "Aqua.jl" begin
        Aqua.test_all(QuAwesome; piracies = false, ambiguities = false, unbound_args = false)
    end

    @testset "JET.jl" begin
        JET.test_package(QuAwesome; target_modules = (QuAwesome,), ignore_missing_comparison = true)
    end
end
