#=
module NNInput 
=================
Written by Maha Chaudhry
=#

module NNInput

include("../src/AbsNormalWithoutBARON.jl")

using .AbsNormal, LinearAlgebra, TimerOutputs, JuMP

export generate_RELU, calculate_root, solve_ODE

##Generate ζ and β structure:
function generate_RELU(
    THETA::Vector{Matrix{Float64}},      #vector of NN coefficients
    CCOEFF_BASE::Vector{Matrix{Float64}} #vector of NN intercepts
)
    #Calculate neuron array information
    n = [size.(THETA, 2); size(THETA[end], 1)]
    n_total = sum(n[2:end]); n_neurons = sum(n[2:end-1]); n_input = n[1]; n_output = n[end]; 

    #Safety measure
    if n_input != n_output
        throw("Number of input neurons must equal number of output neurons for a square system!")
    end #if 

    #Calculate META intermediate value 
    intermediate = cat(THETA..., dims=(1,2))
    intermediate[(n[2]+1):end, (n[1]+1):end] = 0.5 .* intermediate[(n[2]+1):end, (n[1]+1):end]
    META = zeros(n_total, n_total)
    META[(n[2]+1):end, 1:(end-n[end])] = intermediate[(n[2]+1):end, (n[1]+1):end]
    META = I(n_total) - META

    #Set up a coefficient matrix ζ (ZETA) 
    ZETA = META \ intermediate

    #Seperate ζ into ANF Coeffs
    Z_coeff = ZETA[1:n_neurons,        1:n_input]
    J_coeff = ZETA[end-n_input+1:end,  1:n_input]

    L_coeff = LowerTriangular(ZETA[1:n_neurons, n_input+1:(n_neurons+n_input)])
    Y_coeff = ZETA[end-n_input+1:end,           n_input+1:(n_neurons+n_input)]

    #Set up a coefficient matrix β (BETA)
    BETA_BASE = cat(CCOEFF_BASE..., dims=1)
    BETA_BASE = META \ BETA_BASE 

    c_coeff = BETA_BASE[1:n_neurons]
    b_coeff = BETA_BASE[end-n_input+1:end]

    return Z_coeff, L_coeff, J_coeff, Y_coeff, b_coeff, c_coeff 
end #function 

function generate_RELU(
    THETA::Vector{Matrix{Float64}},           #vector of NN coefficients
    CCOEFF_BASE::Vector{Matrix{Float64}},     #vector of NN intercepts
    CCOEFF_TIME::Vector{Float64}              #vector of NN time specific intercept        
)
    #Calculate neuron array information
    n = [size.(THETA, 2); size(THETA[end], 1)]
    n_total = sum(n[2:end]); n_neurons = sum(n[2:end-1]); n_input = n[1]; n_output = n[end]; 

    #Safety measure
    if n_input != n_output
        throw("Number of input neurons must equal number of output neurons for a square system!")
    end #if 

    #Calculate META intermediate value 
    intermediate = cat(THETA..., dims=(1,2))
    intermediate[(n[2]+1):end, (n[1]+1):end] = 0.5 .* intermediate[(n[2]+1):end, (n[1]+1):end]
    META = zeros(n_total, n_total)
    META[(n[2]+1):end, 1:(end-n[end])] = intermediate[(n[2]+1):end, (n[1]+1):end]
    META = I(n_total) - META

    #Set up a coefficient matrix ζ (ZETA) 
    ZETA = META \ intermediate

    #Seperate ζ into ANF Coeffs
    Z_coeff = ZETA[1:n_neurons,        1:n_input]
    J_coeff = ZETA[end-n_input+1:end,  1:n_input]

    L_coeff = LowerTriangular(ZETA[1:n_neurons, n_input+1:(n_neurons+n_input)])
    Y_coeff = ZETA[end-n_input+1:end,           n_input+1:(n_neurons+n_input)]

    #Set up a coefficient matrix β (BETA)
    BETA_BASE = cat(CCOEFF_BASE..., dims=1)
    BETA_BASE = META \ BETA_BASE 

    BETA_TIME = zeros(n_total)
    BETA_TIME[1:n[2]] = CCOEFF_TIME
    BETA_TIME = META \ BETA_TIME

    return Z_coeff, L_coeff, J_coeff, Y_coeff, BETA_BASE, BETA_TIME
end #function 

##Calculate Root:
function calculate_root(
    Z_coeff::Matrix{Float64}, 
    L_coeff::LowerTriangular{Float64, Matrix{Float64}}, 
    J_coeff::Matrix{Float64}, 
    Y_coeff::Matrix{Float64}, 
    b_coeff::Array{Float64}, 
    c_coeff::Array{Float64},
    P, Q, r;
    solve_mode::String = "LCP"
)
    #(a) Coefficient adjustement for P, Q, r: 
    Z_coeff_ = Z_coeff
    L_coeff_ = L_coeff
    c_coeff_ = c_coeff

    J_coeff_ = P + Q * J_coeff
    Y_coeff_ = Q * Y_coeff
    b_coeff_ = r + Q * b_coeff

    #(b) Solve root:
    anf = AbsNormal.AnfCoeffs(c_coeff_, b_coeff_, Z_coeff_, L_coeff_, J_coeff_, Y_coeff_)  
    if solve_mode == "LCP"
        try
            rootVal, terminationStatusLCP = AbsNormal.solve_pa_equation(anf, approach=AbsNormal.BY_LCP, solverAttributes = (MOI.Silent() => true,));
            @show terminationStatusLCP

            return rootVal
        catch err
            println("LCP solver failed to run!\n")
        end #try  

    elseif solve_mode == "MLCP"
        try
            rootVal, terminationStatusMLCP = AbsNormal.solve_pa_equation(anf, approach=AbsNormal.BY_MLCP, solverAttributes = (MOI.Silent() => true,));
            @show terminationStatusMLCP

            return rootVal
        catch err
            println("MLCP solver failed to run!\n")
        end #try 
    end #if 
end #function 

##Calculate ODE:
function solve_ODE(
    INIT_COND::Vector{Any}, 
    dt::Float64, 
    occurences::Int64,
    Z_coeff::Matrix{Float64}, 
    L_coeff::LowerTriangular{Float64, Matrix{Float64}}, 
    J_coeff::Matrix{Float64}, 
    Y_coeff::Matrix{Float64}, 
    BETA_BASE::Array{Float64}, 
    BETA_TIME::Array{Float64};
    solve_mode::String = "LCP"
)
    #Neuron array information:
    n_input = size(INIT_COND, 1); n_output = size(J_coeff, 1); n_neurons = size(L_coeff, 1);
    
    #Set up initial condition:
    xk_OUTPUT = zeros(n_input, occurences+1)
    xk_OUTPUT[:, 1] = INIT_COND
    timeI = [dt * i for i in 0:occurences]

    #Calculate un-changing coefficient P and Q:
    P = I(n_output)
    Q = - dt * I(n_output)

    #For each time-step:
    for i in 1:occurences
        #Calculate coefficient r:
        rI = -xk_OUTPUT[:, i]

        #Adjust coefficient β for time information:
        BETA_TOTAL = BETA_BASE + BETA_TIME*timeI[i]
        c_coeff_I = BETA_TOTAL[1:n_neurons]
        b_coeff_I = BETA_TOTAL[end-n_output+1:end]

        #Calculate x^{i+1} value:
        xk_OUTPUT[:, i+1] = calculate_root(
            Z_coeff, L_coeff, J_coeff, Y_coeff, b_coeff_I, c_coeff_I,
            P, Q, rI;
            solve_mode = "LCP"
        )
    end #for 

    return xk_OUTPUT
end #function

end #module