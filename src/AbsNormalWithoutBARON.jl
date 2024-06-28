#=
module AbsNormal
=================
Written by Kamil Khan and Yulan Zhang
=#

module AbsNormal

import JuMP, PATHSolver, GLPK

export AnfCoeffs,
    SolutionApproachEq, BY_MLCP, BY_LCP,
    SolutionApproachOptim, BY_LPCC, BY_MILP,
    solve_pa_equation,
    verify_minimum_existence,
    minimize_pa_function,
    modify_anf_coeffs


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
    optimizer = PATHSolver.Optimizer,
    solverAttributes = ("output" => "no",)
)
    aMod = modify_anf_coeffs(a::AnfCoeffs)
    (cm, bm, Zm, Lm, Jm, Ym) = recover_anf_coeffs(aMod)

    anfModel = JuMP.Model(JuMP.optimizer_with_attributes(optimizer, solverAttributes...))

    if approach == BY_MLCP
        JuMP.@variable(anfModel, x[1:size(Zm)[2]])
        JuMP.@variable(anfModel, w[1:length(cm)] >= 0.0)

        #JuMP.@constraint(anfModel, bm + Jm*x + Ym*u .== 0.0) # not supported by PATHSolver.jl
        JuMP.@constraint(anfModel, complements(bm + Jm*x + Ym*w, x)) 
        JuMP.@constraint(anfModel, complements(cm + Zm*x + Lm*w, w))

        JuMP.optimize!(anfModel)
        xStar = JuMP.value.(x)
        wStar = JuMP.value.(w)
        terminationStatus = JuMP.termination_status(anfModel)

    elseif approach == BY_LCP
        ZJmInv = Zm/Jm
        gamma = cm - ZJmInv*bm
        S = Lm - ZJmInv*Ym

        JuMP.@variable(anfModel, w[1:length(cm)] >= 0.0)

        JuMP.@constraint(anfModel, complements(gamma + S*w, w))

        JuMP.optimize!(anfModel)
        wStar = JuMP.value.(w)
        xStar = -Jm\(bm + Ym*wStar)
        terminationStatus = JuMP.termination_status(anfModel)
        
    else
        throw(DomainError(:approach, "unsupported equation-solving approach"))
    end

    return xStar, terminationStatus, aMod
end


function verify_minimum_existence(
    a::AnfCoeffs;
    approach::SolutionApproachEq = BY_LCP
)
    aMod = modify_horizf_coeffs(a::AnfCoeffs)
    (cm, bm, Zm, Lm, Jm, Ym) = recover_anf_coeffs(aMod)
    
    if approach == BY_MLCP
        
        error("this requires BARON; please install BARON and use AbsNormalWithBARON.jl instead.")

    elseif approach == BY_LCP

        optimizer = PATHSolver.Optimizer
        solverAttributes = ("output" => "no",)
        horizfModel = JuMP.Model(JuMP.optimizer_with_attributes(optimizer, solverAttributes...))
        
        ZJmInv = Zm/Jm
        gamma = -ZJmInv*bm
        S = Lm - ZJmInv*Ym

        JuMP.@variable(horizfModel, omega[1:size(Zm)[1]] >= 0.0)

        JuMP.@constraint(horizfModel, complements(gamma + S*omega, omega))

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
    optimizer = GLPK.Optimizer,
    solverAttributes = ("msg_lev" => GLPK.GLP_MSG_OFF,),
    mu = 1e5
)
    
    aMod = modify_anf_coeffs(a::AnfCoeffs)
    (cm, bm, Zm, Lm, Jm, Ym) = recover_anf_coeffs(aMod)
    
    minfModel = JuMP.Model(JuMP.optimizer_with_attributes(optimizer, solverAttributes...))
    
    if approach == BY_LPCC
        
        error("this requires BARON; please install BARON and use AbsNormalWithBARON.jl instead.")
        
    
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
        #global_minimum = "PA function f has a global minimum"
        xStar = JuMP.value.(x)
    else
        #global_minimum = "PA function f does not have a global minimum"
        xStar = Inf
        println("Minimum not found")
    end

    return xStar, fStar, terminationStatus_minif
    
end

end # module


