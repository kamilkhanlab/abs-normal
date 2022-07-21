# module AbsNormal

import JuMP, PATHSolver

using LinearAlgebra

struct AnfCoeffs
    c::Vector{Float64}
    b::Vector{Float64}
    Z::Matrix{Float64}
    L::LowerTriangular{Float64, Matrix{Float64}}
    J::Matrix{Float64}
    Y::Matrix{Float64}
end

@enum EquationSolvingApproach begin
    BY_MLCP
    BY_LCP
end

function modify_anf_coeffs(a::AnfCoeffs)
    IMinusL = I - a.L
    (cm, Zm, Lm) = map(X -> IMinusL\X, (a.c, a.Z, I+a.L))
    bm = a.b + a.Y*cm
    Ym = a.Y*(I + Lm)
    Jm = a.J + a.Y*Zm
    return AnfCoeffs(cm, bm, Zm, Lm, Jm, Ym)
end

recover_anf_coeffs(a::AnfCoeffs) = a.c, a.b, a.Z, a.L, a.J, a.Y

function solve_pa_equation(a::AnfCoeffs; approach::EquationSolvingApproach = BY_MLCP)
    aMod = modify_anf_coeffs(a::AnfCoeffs)
    (cm, bm, Zm, Lm, Jm, Ym) = recover_anf_coeffs(aMod)

    anfModel = JuMP.Model(PATHSolver.Optimizer)
    JuMP.set_optimizer_attribute(anfModel, "output", "no")

    if approach == BY_MLCP
        JuMP.@variable(anfModel, x[1:size(Zm)[2]])
        JuMP.@variable(anfModel, u[1:length(cm)] >= 0.0)

        # JuMP.@constraint(anfModel, bm + Jm*x + Ym*u .== 0.0) # not supported by PATHSolver.jl
        JuMP.@constraint(anfModel, complements(bm + Jm*x + Ym*u, x))
        JuMP.@constraint(anfModel, complements(cm + Zm*x + Lm*u, u))

        JuMP.optimize!(anfModel)
        xStar = JuMP.value.(x)
        terminationStatus = JuMP.termination_status(anfModel)

    elseif approach == BY_LCP
        ZJmInv = Zm/Jm
        gamma = cm - ZJmInv*bm
        S = Lm - ZJmInv*Ym

        JuMP.@variable(anfModel, u[1:length(cm)] >= 0.0)

        JuMP.@constraint(anfModel, complements(gamma + S*u, u))

        JuMP.optimize!(anfModel)
        uStar = JuMP.value.(u)
        xStar = -Jm\(bm + Ym*uStar)
        terminationStatus = JuMP.termination_status(anfModel)
        
    else
        throw(DomainError(:approach, "unsupported equation-solving approach"))
    end

    return xStar, terminationStatus, aMod
end

# test PATHSolver.jl using its documented example
function test_PATH()
    M = [0 0 -1 -1;
         0 0 1 -2;
         1 -1 2 -2;
         1 2 -2 4]

    q = [2, 2, -2, -6]

    model = JuMP.Model(PATHSolver.Optimizer)
    JuMP.set_optimizer_attribute(model, "output", "no")
    
    JuMP.@variable(model, x[1:4] >= 0)
    
    JuMP.@constraint(model, complements(M*x + q, x))

    JuMP.optimize!(model)
    
    @show JuMP.termination_status(model)
    @show JuMP.value.(x)
    # @show JuMP.solution_summary(model) # apparently not yet implemented
end

# solve the equation system:
#   0 = 1 + x - abs(x) + abs(x + abs(x))
function example_1()
    c = [0.0, 0.0]
    b = [1.0]
    Z = [1.0; 1.0;;]
    L = LowerTriangular([0.0 0.0; 1.0 0.0])
    J = [1.0;;]
    Y = [-1.0 1.0]

    anf = AnfCoeffs(c, b, Z, L, J, Y)

    xStar, terminationStatus = solve_pa_equation(anf)

    @show xStar
    @show terminationStatus
end

# solve the equation system:
#   0 = 1 + x + 0.5*abs(x) - 0.5*abs(x)
function example_2()
    c = [0.0, 0.0]
    b = [1.0]
    Z = [1.0; 1.0;;]
    L = LowerTriangular(zeros(2, 2))
    J = [1.0;;]
    Y = [0.5 -0.5]

    anf = AnfCoeffs(c, b, Z, L, J, Y)

    xStar, terminationStatus = solve_pa_equation(anf)

    @show xStar
    @show terminationStatus
end

# solve the equation system:
#   0 = abs(x) - 1
function example_3()
    c = [0.0]
    b = [-1.0]
    Z = [1.0;;]
    L = LowerTriangular(zeros(1, 1))
    J = [0.0;;]
    Y = [1.0;;]

    anf = AnfCoeffs(c, b, Z, L, J, Y)

    xStar, terminationStatus = solve_pa_equation(anf)

    @show xStar
    @show terminationStatus
end

example_1()
example_2()
example_3()
