#=
module AbsNormal
=================
Written by Kamil Khan and Yulan Zhang
=#

module AbsNormal

import JuMP, PATHSolver, BARON

# A temporary PATH license that is valid for a year
# Which can be found in https://pages.cs.wisc.edu/~ferris/path/julia/LICENSE
PATHSolver.c_api_License_SetString("2830898829&Courtesy&&&USR&45321&5_1_2021&1000&PATH&GEN&31_12_2025&0_0_0&6000&0_0")

export AnfCoeffs,
    SolutionApproachEq, BY_MLCP, BY_LCP,
    SolutionApproachEqGriewank, BY_GriewankMLCP, BY_GriewankLCP,
    SolutionApproachOptim, BY_LPCC, BY_MILP,
    SolverEq, BY_PATHSolver, BY_BARON,
    solve_pa_equation,
    solve_pa_equation_Griewank,
    verify_minimum_existence,
    minimize_pa_function,
    modify_anf_coeffs,
    recover_anf_coeffs

using LinearAlgebra


struct AnfCoeffs
    c::Vector{Float64}
    b::Vector{Float64}
    Z::Matrix{Float64}
    L::LowerTriangular{Float64, Matrix{Float64}}
    J::Matrix{Float64}
    Y::Matrix{Float64}
end

@enum SolutionApproachEq begin
    BY_MLCP
    BY_LCP
end

@enum SolutionApproachOptim begin
    BY_LPCC
    BY_MILP
end

@enum SolutionApproachEqGriewank begin
    BY_GriewankMLCP
    BY_GriewankLCP
end

@enum SolverEq begin
    BY_PATHSolver
    BY_BARON
end

function modify_anf_coeffs(a::AnfCoeffs)
    IMinusL = UnitLowerTriangular(I - a.L)
    (cm, Zm, Lm) = map(X -> IMinusL\X, (a.c, a.Z, I+a.L))
    bm = a.b + a.Y*cm
    Ym = a.Y*(I + Lm)
    Jm = a.J + a.Y*Zm
    return AnfCoeffs(cm, bm, Zm, Lm, Jm, Ym)
end

function modify_horizf_coeffs(a::AnfCoeffs)
    IMinusL = UnitLowerTriangular(I - a.L)
    (Zm, Lm) = map(X -> IMinusL\X, (a.Z, I+a.L))
    cm = a.c * 0.0
    bm = [1.0]
    Ym = a.Y*(I + Lm)
    Jm = a.J + a.Y*Zm
    return AnfCoeffs(cm, bm, Zm, Lm, Jm, Ym)
end


recover_anf_coeffs(a::AnfCoeffs) = a.c, a.b, a.Z, a.L, a.J, a.Y


function solve_pa_equation(
    a::AnfCoeffs;
    approach::SolutionApproachEq = BY_LCP,
    solver::SolverEq = BY_PATHSolver,
)
    aMod = modify_anf_coeffs(a::AnfCoeffs)
    (cm, bm, Zm, Lm, Jm, Ym) = recover_anf_coeffs(aMod)

    if solver == BY_PATHSolver
        solverAttributes = ("output" => "no",)
        optimizer = PATHSolver.Optimizer
    elseif solver == BY_BARON
        solverAttributes = ("PrLevel" => 0,)
        optimizer = BARON.Optimizer
    end
    
    anfModel = JuMP.Model(JuMP.optimizer_with_attributes(optimizer, solverAttributes...))
        
    if approach == BY_MLCP 
        
        JuMP.@variable(anfModel, x[1:size(Zm)[2]])
        JuMP.@variable(anfModel, w[1:length(cm)] >= 0.0)
        
        if solver == BY_PATHSolver
            #JuMP.@constraint(anfModel, bm + Jm*x + Ym*u .== 0.0) # not supported by PATHSolver.jl
            JuMP.@constraint(anfModel, complements(bm + Jm*x + Ym*w, x)) 
            JuMP.@constraint(anfModel, complements(cm + Zm*x + Lm*w, w))
        elseif solver == BY_BARON
            JuMP.@constraint(anfModel, bm + Jm*x + Ym*w .== 0.0)
            for i in eachindex(cm)
                JuMP.@constraints(anfModel, begin
                            (cm[i] + (Zm*x)[i] + (Matrix(Lm)*w)[i]) * w[i] == 0.0
                             cm[i] + (Zm*x)[i] + (Matrix(Lm)*w)[i] >= 0.0
                             w[i] >= 0.0
                end)
            end
        end
    
        JuMP.optimize!(anfModel)
        terminationStatus = JuMP.termination_status(anfModel)
        
        if terminationStatus in [JuMP.OPTIMAL, JuMP.LOCALLY_SOLVED]
            xStar = JuMP.value.(x)
            wStar = JuMP.value.(w)
        else 
            wStar = Inf
            xStar = Inf
        end
        

    elseif approach == BY_LCP
        ZJmInv = Zm/Jm
        gamma = cm - ZJmInv*bm
        S = Lm - ZJmInv*Ym

        JuMP.@variable(anfModel, w[1:length(cm)] >= 0.0)

        if solver == BY_PATHSolver
            JuMP.@constraint(anfModel, complements(gamma + S*w, w))
        elseif solver == BY_BARON
            
            for i in eachindex(gamma)
            JuMP.@constraints(anfModel, begin
                        (gamma[i] + (S*w)[i]) * w[i] == 0.0
                         gamma[i] + (S*w)[i] >= 0.0
                         w[i] >= 0.0
                end)
            end
        end

            
        JuMP.optimize!(anfModel)
        terminationStatus = JuMP.termination_status(anfModel)
        if terminationStatus in [JuMP.OPTIMAL, JuMP.LOCALLY_SOLVED]
            wStar = JuMP.value.(w)
            xStar = -Jm\(bm + Ym*wStar)
        else 
            wStar = Inf
            xStar = Inf
        end
    
    else
        throw(DomainError(:approach, "unsupported equation-solving approach"))
    end

    return xStar, terminationStatus, aMod
end


function solve_pa_equation_Griewank(
    a::AnfCoeffs;
    approach::SolutionApproachEqGriewank = BY_GriewankMLCP,
    solver::SolverEq = BY_PATHSolver,
)

    
    (c, b, Z, L, J, Y) = recover_anf_coeffs(a::AnfCoeffs)
    
    ZJInv = Z/J
    gamma = c - ZJInv*b
    S = L - ZJInv*Y
        
    if solver == BY_PATHSolver
        solverAttributes = ("output" => "no",)
        optimizer = PATHSolver.Optimizer
    elseif solver == BY_BARON
        solverAttributes = ("PrLevel" => 0,)
        optimizer = BARON.Optimizer
    end
    
    anfModel = JuMP.Model(JuMP.optimizer_with_attributes(optimizer, solverAttributes...))
        
    if approach == BY_GriewankMLCP 
        
        JuMP.@variable(anfModel, u[1:length(c)] >= 0.0)
        JuMP.@variable(anfModel, w[1:length(c)] >= 0.0)
        
        if solver == BY_PATHSolver
            throw(DomainError(:solver, "unsupported solver, use BARON instead"))
            
        elseif solver == BY_BARON
            
            JuMP.@constraint(anfModel, u-w .== gamma + S*(u+w))
            
            for i in eachindex(c)
                JuMP.@constraints(anfModel, begin
                             u[i] * w[i] == 0.0
                             u[i] >= 0.0
                             w[i] >= 0.0
                end)
            end
        end

        JuMP.optimize!(anfModel)
        terminationStatus = JuMP.termination_status(anfModel)
        
        if terminationStatus in [JuMP.OPTIMAL, JuMP.LOCALLY_SOLVED]
            uStar = JuMP.value.(u)
            wStar = JuMP.value.(w)
            xStar = -J\(b + Y*(uStar+wStar))
        else 
            uStar = Inf
            wStar = Inf
            xStar = Inf
        end

    elseif approach == BY_GriewankLCP
        
        IMinusS = I - S

        JuMP.@variable(anfModel, w[1:length(c)] >= 0.0)

        if solver == BY_PATHSolver
            JuMP.@constraint(anfModel, complements(IMinusS\gamma + IMinusS\(I+S)*w, w))
        elseif solver == BY_BARON
            
            for i in eachindex(gamma)
            JuMP.@constraints(anfModel, begin
                        ((IMinusS\gamma)[i] + (IMinusS\(I+S)*w)[i]) * w[i] == 0.0
                         (IMinusS\gamma)[i] + (IMinusS\(I+S)*w)[i] >= 0.0
                         w[i] >= 0.0
                end)
            end
        end
            
        JuMP.optimize!(anfModel)
        terminationStatus = JuMP.termination_status(anfModel)
        
        if terminationStatus in [JuMP.OPTIMAL, JuMP.LOCALLY_SOLVED]
            wStar = JuMP.value.(w)
            uStar = IMinusS\gamma + IMinusS\(I+S)*wStar
            xStar = -J\(b + Y*(uStar+wStar))
        else 
            uStar = Inf
            wStar = Inf
            xStar = Inf
        end
    
    else
        throw(DomainError(:approach, "unsupported equation-solving approach"))
    end

    return xStar, terminationStatus
end


function verify_minimum_existence(
    a::AnfCoeffs;
    approach::SolutionApproachEq = BY_MLCP
)
    aMod = modify_horizf_coeffs(a::AnfCoeffs)
    (cm, bm, Zm, Lm, Jm, Ym) = recover_anf_coeffs(aMod)
    
    if approach == BY_MLCP
        
        optimizer = BARON.Optimizer
        solverAttributes = ("PrLevel" => 0,)
        
        horizfModel = JuMP.Model(JuMP.optimizer_with_attributes(optimizer, solverAttributes...))
        JuMP.@variable(horizfModel, xi[1:size(Zm)[2]])
        JuMP.@variable(horizfModel, omega[1:size(Zm)[1]] >= 0.0)

        
        JuMP.@constraint(horizfModel, [1.0] + Jm*xi + Ym*omega .== 0.0) 
        
        for i in eachindex(omega)
            JuMP.@constraints(horizfModel, begin
                        ((Zm*xi)[i] + (Matrix(Lm)*omega)[i]) * omega[i] == 0.0
                         (Zm*xi)[i] + (Matrix(Lm)*omega)[i] >= 0.0
                         omega[i] >= 0.0
            end)
            
        end
        
        JuMP.optimize!(horizfModel)
        terminationStatus_horizf = JuMP.termination_status(horizfModel)
        
    else
        throw(DomainError(:approach, "unsupported equation-solving approach"))
    end

    globalMinExists = true
    if terminationStatus_horizf in [JuMP.OPTIMAL, JuMP.LOCALLY_SOLVED]
        globalMinExists = false
    end

    return terminationStatus_horizf, globalMinExists
end


function minimize_pa_function(
    a::AnfCoeffs;
    approach::SolutionApproachOptim = BY_LPCC,
    optimizer = BARON.Optimizer,
    solverAttributes = ("PrLevel" => 0,),
    mu = 1e5
)
    
    aMod = modify_anf_coeffs(a::AnfCoeffs)
    (cm, bm, Zm, Lm, Jm, Ym) = recover_anf_coeffs(aMod)
    
    minfModel = JuMP.Model(JuMP.optimizer_with_attributes(optimizer, solverAttributes...))
    
    if approach == BY_LPCC
        
        JuMP.@variable(minfModel, x[1:size(Zm)[2]])
        JuMP.@variable(minfModel, w[1:length(cm)] >= 0.0)

        JuMP.@objective(minfModel, Min, (bm + Jm * x + Ym * w)[1])

        for i in eachindex(cm)
            JuMP.@constraints(minfModel, begin
                        (cm[i] + (Zm*x)[i] + (Matrix(Lm)*w)[i]) * w[i] == 0.0
                         cm[i] + (Zm*x)[i] + (Matrix(Lm)*w)[i] >= 0.0
                         w[i] >= 0.0
            end)
            
        end
    
        JuMP.optimize!(minfModel)
        
        terminationStatus_minif = JuMP.termination_status(minfModel) 
        
        wStar = JuMP.value.(w)
        xStar = JuMP.value.(x)
        fStar = bm + Jm * xStar + Ym * wStar
        
    
    elseif approach == BY_MILP
        
        JuMP.@variable(minfModel, x[1:size(Zm)[2]])
        JuMP.@variable(minfModel, 0.0 <= w[1:length(cm)] <= mu)
        JuMP.@variable(minfModel, y[1:length(cm)], Bin)
        
        JuMP.@objective(minfModel, Min, (bm + Jm * x + Ym * w)[1])
        
        JuMP.@constraint(minfModel,w <= mu * y)

        e = ones(length(cm))

        JuMP.@constraint(minfModel, zeros(length(cm)) <= cm + Zm*x + Matrix(Lm)*w)
        JuMP.@constraint(minfModel, cm + Zm*x + Matrix(Lm)*w <= mu*(e-y))
        
        
        JuMP.optimize!(minfModel)
        
        terminationStatus_minif = JuMP.termination_status(minfModel)
        
        wStar = JuMP.value.(w)
        xStar = JuMP.value.(x)
        fStar = bm + Jm * xStar + Ym * wStar


    else
        throw(DomainError(:approach, "unsupported equation-solving approach"))
    end
    
    
    if terminationStatus_minif in [JuMP.OPTIMAL, JuMP.LOCALLY_SOLVED]
        xStar = JuMP.value.(x)
    else
        xStar = Inf
        println("Minimum not found")
    end

    return xStar, fStar, terminationStatus_minif
    
end

end # module


