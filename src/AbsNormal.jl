# module AbsNormal

import JuMP, PATHSolver

using LinearAlgebra

## solve the test problem listed at https://github.com/chkwon/PATHSolver.jl

#=
const M = [0 0 -1 -1;
           0 0 1 -2;
           1 -1 2 -2;
           1 2 -2 4]

const q = [2, 2, -2, -6]

model = JuMP.Model(PATHSolver.Optimizer)
JuMP.set_optimizer_attribute(model, "output", "no")
JuMP.@variable(model, x[1:4] >= 0)
JuMP.@constraint(model, complements(M*x + q, x))

JuMP.optimize!(model)
@show JuMP.termination_status(model)
@show JuMP.value.(x)
# @show JuMP.solution_summary(model) # apparently not yet implemented
=#

## solve absolute value equations via CANF

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
    (cm, Lm, Zm) = map(X -> IMinusL\X, (a.c, I+a.L, a.Z))
    bm = a.b + a.Y*cm
    Ym = a.Y*(I + Lm)
    Jm = a.J + a.Y*Zm
    return AnfCoeffs(cm, bm, Zm, Lm, Jm, Ym)
end

function solve_pa_equation(a::AnfCoeffs, approach::EquationSolvingApproach = BY_MLCP)
    aMod = modify_anf_coeffs(a::AnfCoeffs)
    cm = aMod.c
    bm = aMod.b
    Zm = aMod.Z
    Lm = aMod.L
    Jm = aMod.J
    Ym = aMod.Y

    anfModel = JuMP.Model(PATHSolver.Optimizer)
    JuMP.set_optimizer_attribute(anfModel, "output", "no")

    if approach == BY_MLCP
        JuMP.@variable(anfModel, x[1:size(Z)[2]])
        JuMP.@variable(anfModel, u[1:length(c)] >= 0.0)

        # JuMP.@constraint(anfModel, bm + Jm*x + Ym*u .== 0.0) # not supported by PATHSolver.jl
        JuMP.@constraint(anfModel, complements(bm + Jm*x + Ym*u, x))
        JuMP.@constraint(anfModel, complements(cm + Zm*x + Lm*u, u))

        JuMP.optimize!(anfModel)
        xStar = JuMP.value.(x)
        terminationStatus = JuMP.termination_status(anfModel)

    elseif approach == BY_LCP
        ZJmInv = Zm/Jm
        gamma = cm - ZJmInv*bm
        S = Lm - ZJmInv*Y

        JuMP.@variable(anfModel, u[1:length(c)] >= 0.0)

        JuMP.@constraint(anfModel, complements(gamma + S*u, u))

        JuMP.optimize!(anfModel)
        uStar = JuMP.value.(u)
        xStar = -Jm\(bm + Ym*uStar)
        terminationStatus = JuMP.termination_status(anfModel)
        
    else
        throw(DomainError(:approach, "unsupported equation-solving approach"))
    end

    return xStar, terminationStatus
end

# solve the equation system:
# 0 = 1 + x - abs(x) + abs(x + abs(x))
function example_1()
    const c = [0.0, 0.0]
    const b = [1.0]
    const Z = [1.0; 1.0;;]
    const L = LowerTriangular([0.0 0.0; 1.0 0.0])
    const J = [1.0;;]
    const Y = [-1.0 1.0]

    anf = anfConstants(b, c, Z, L, Y, J)

    xStar, terminationStatus = solve_pa_equation(anf)

    @show xStar
    @show terminationStatus
end
    
#=
IMinusL = I - L
(cm, Lm, Zm) = map(X -> IMinusL\X, (c, I+L, Z))
bm = b + Y*cm
Ym = Y*(I + Lm)
Jm = J + Y*Zm

ZJmInv = Zm/Jm
gamma = cm - ZJmInv*bm
S = Lm - ZJmInv*Y

anfModel = JuMP.Model(PATHSolver.Optimizer)
JuMP.set_optimizer_attribute(anfModel, "output", "no")

JuMP.@variable(anfModel, x[1:size(Z)[2]])
JuMP.@variable(anfModel, u[1:length(c)] >= 0.0)
# JuMP.@variable(anfModel, dummyVar[1:length(b)])

# JuMP.@constraint(anfModel, bm + Jm*x + Ym*u .== 0.0) # not supported by PATHSolver.jl
JuMP.@constraint(anfModel, complements(bm + Jm*x + Ym*u, x))
JuMP.@constraint(anfModel, complements(cm + Zm*x + Lm*u, u))

JuMP.optimize!(anfModel)

@show JuMP.termination_status(anfModel)
@show JuMP.value.(x)
=#

#=
JuMP.@variable(anfModel, u[1:length(c)] >= 0.0)

JuMP.@constraint(anfModel, complements(gamma + S*u, u))

JuMP.optimize!(anfModel)

@show JuMP.termination_status(anfModel)
uStar = JuMP.value.(u)
xStar = -Jm\(bm + Ym*uStar)
@show xStar
=#

# end # module
