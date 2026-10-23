#=
module NNInput 
=================
Written by Maha Chaudhry
=#

module NNInput

include("../src/AbsNormalWithoutBARON.jl")

using .AbsNormal, LinearAlgebra, TimerOutputs, JuMP

export SolutionApproachEq, generate_RELU, generate_RELU_PI, calculate_root, solve_ODE, solve_BDF_ODE, solve_ODE_PI

#TODO: Set this up
    # struct BDFParams{T <: Tuple}
    #     q_BDF::NTuple{6, Float64}
    #     r_values::NTuple{6, Vector{Float64}}

    #     function BDFParams()
    #         q = (1.0, 2/3, 6/11, 12/25, 60/137, 60/147)
    #         r = (
    #             [-1.0], 
    #             [1/3, -4/3], 
    #             [-2/11, 9/11, -18/11], 
    #             [3/25, -16/25, 36/25, -48/25], 
    #             [-12/137, 75/137, -200/137, 300/137, -300/137], 
    #             [10/147, -72/147, 225/147, -400/147, 450/147, -360/147]
    #         )
    #         new(q, r)
    #     end #function 

    #     function (p::BDFParams)(order::Int, flag::Symbol)
    #         if flag === :q
    #             return p.q_values[order]
    #         elseif flag === :r
    #             return p.r_values[order]
    #         else
    #             error("Invalid flag. Use :q for constants or :r.")
    #         end #if 
    #     end #function 
    # end #struct
#

#TODO: Physics-Informed System: 
function generate_RELU_PI(
    THETA::Array{Matrix{Float64}},      #vector of NN coefficients
    CCOEFF_BASE::Array{Matrix{Float64}}; #vector of NN intercepts
    ALPHA::Float64 = 0.0                #parameter for Leaky ReLU
)
    n = [size.(THETA, 2); size(THETA[end], 1)]
    n_total = sum(n[2:end]); n_neurons = sum(n[2:end-1]); n_input = n[1]; n_output = n[end]; 

    #Calculate RHS coefficient matrix for |Z|:
    LETA = cat(THETA..., dims=(1,2))

    #Calculate LHS coefficient matrix for Z: 
    META = zeros(n_total, n_total)
    META[(n[2]+1):end, 1:(end-n[end])] = (1+ALPHA)/2 .* LETA[(n[2]+1):end, (n[1]+1):end]    
    META = I(n_total) - META

    LETA[(n[2]+1):end, (n[1]+1):end] = (1-ALPHA)/2 .* LETA[(n[2]+1):end, (n[1]+1):end]

    #Set up a coefficient matrix ζ (ZETA) 
    ZETA = META \ LETA

    #Seperate ζ into ANF Coeffs
    Z_coeff = ZETA[1:n_neurons,        1:n_input]
    J_coeff = ZETA[end-n_output+1:end, 1:n_input]

    L_coeff = LowerTriangular(ZETA[1:n_neurons,  n_input+1:end])
    Y_coeff = ZETA[end-n_output+1:end,           n_input+1:end]

    #Set up a coefficient matrix β (BETA)
    BETA_BASE = cat(CCOEFF_BASE..., dims=1)
    BETA_BASE = META \ BETA_BASE 

    c_coeff = BETA_BASE[1:n_neurons]
    b_coeff = BETA_BASE[end-n_output+1:end]

    return Z_coeff, L_coeff, J_coeff, Y_coeff, b_coeff, c_coeff 
end #function 

##Generate ζ and β structure:
function generate_RELU(
    THETA::Array{Matrix{Float64}},      #vector of NN coefficients
    CCOEFF_BASE::Array{Matrix{Float64}}; #vector of NN intercepts
    ALPHA::Float64 = 0.0                #parameter for Leaky ReLU
)   
    #Calculate neuron array information
    n = [size.(THETA, 2); size(THETA[end], 1)]
    n_total = sum(n[2:end]); n_neurons = sum(n[2:end-1]); n_input = n[1]; n_output = n[end]; 

    #Safety measure
    if n_input != n_output
        throw("Number of input neurons must equal number of output neurons for a square system!")
    end #if 

    #Calculate RHS coefficient matrix for |Z|:
    LETA = cat(THETA..., dims=(1,2))
   
    #Calculate LHS coefficient matrix for Z: 
    META = zeros(n_total, n_total)
    META[(n[2]+1):end, 1:(end-n[end])] = (1+ALPHA)/2 .* LETA[(n[2]+1):end, (n[1]+1):end]    
    META = I(n_total) - META

    LETA[(n[2]+1):end, (n[1]+1):end] = (1-ALPHA)/2 .* LETA[(n[2]+1):end, (n[1]+1):end]

    #Set up a coefficient matrix ζ (ZETA) 
    ZETA = META \ LETA

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
    THETA::Array{Matrix{Float64}},           #vector of NN coefficients
    CCOEFF_BASE::Array{Matrix{Float64}},     #vector of NN intercepts
    CCOEFF_TIME::Array{Float64};             #vector of NN time specific intercept       
    ALPHA::Float64 = 0.0                     #leaky RELU parameter 
)
    #Calculate neuron array information
    n = [size.(THETA, 2); size(THETA[end], 1)]
    n_total = sum(n[2:end]); n_neurons = sum(n[2:end-1]); n_input = n[1]; n_output = n[end]; 

    #Safety measure
    if n_input != n_output
        throw("Number of input neurons must equal number of output neurons for a square system!")
    end #if 

    #Calculate RHS coefficient matrix for |Z|:
    LETA = cat(THETA..., dims=(1,2))
   
    #Calculate LHS coefficient matrix for Z: 
    META = zeros(n_total, n_total)
    META[(n[2]+1):end, 1:(end-n[end])] = (1+ALPHA)/2 .* LETA[(n[2]+1):end, (n[1]+1):end]    
    META = I(n_total) - META

    LETA[(n[2]+1):end, (n[1]+1):end] = (1-ALPHA)/2 .* LETA[(n[2]+1):end, (n[1]+1):end]
    
    #Set up a coefficient matrix ζ (ZETA) 
    ZETA = META \ LETA

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
    solve_mode::SolutionApproachEq = AbsNormal.BY_LCP,
    solverAttributes = (MOI.Silent() => true,)
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
    try
        rootVal, terminationStatus = AbsNormal.solve_pa_equation(anf, approach=solve_mode, solverAttributes = solverAttributes);
        # @show terminationStatus

        return rootVal
    catch err
        println("Solver failed to run!\n")
    end #try  
end #function 

##Calculate ODE:
function solve_ODE(
    INIT_COND::Array{Float64}, 
    dt::Float64, 
    occurences::Int64,
    Z_coeff::Matrix{Float64}, 
    L_coeff::LowerTriangular{Float64, Matrix{Float64}}, 
    J_coeff::Matrix{Float64}, 
    Y_coeff::Matrix{Float64}, 
    BETA_BASE::Array{Float64}, 
    BETA_TIME::Array{Float64};
    solve_mode::SolutionApproachEq = AbsNormal.BY_LCP,
    solverAttributes = (MOI.Silent() => true,)
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
            solve_mode = solve_mode,
            solverAttributes = solverAttributes
        )
    end #for 

    return xk_OUTPUT
end #function

function solve_ODE_PI(
    INIT_COND::Array{Float64}, 
    dt::Float64, 
    occurences::Int64,
    Z_coeff::Matrix{Float64}, 
    L_coeff::LowerTriangular{Float64, Matrix{Float64}}, 
    J_coeff::Matrix{Float64}, 
    Y_coeff::Matrix{Float64}, 
    BETA_BASE::Array{Float64}, 
    BETA_TIME::Array{Float64};
    solve_mode::SolutionApproachEq = AbsNormal.BY_LCP,
    solverAttributes = (MOI.Silent() => true,)
)
    #Neuron array information:
    n_input = size(Z_coeff, 2); n_output = size(J_coeff, 1); n_neurons = size(L_coeff, 1);
    
    xk_OUTPUT = zeros(n_input, occurences+1)
    timeI = [dt * i for i in 0:occurences]

    #Set up initial condition:
    u_init = sin.(INIT_COND)
    v_init = cos.(INIT_COND)
    xk_OUTPUT[1:2:end, 1] .= u_init 
    xk_OUTPUT[2:2:end, 1] .= v_init 

    #Calculate un-changing coefficient P and Q:
    P = I(n_output)
    Q = - dt * I(n_output)

    println("Initial variables loaded. Now running Implicit Euler...")

    #For each time-step:
    for i in 1:occurences
        #Calculate coefficient r:
        rI = -xk_OUTPUT[:, i]

        #Adjust coefficient β for time information:
        BETA_TOTAL = BETA_BASE + BETA_TIME*timeI[i]
        c_coeff_I = BETA_TOTAL[1:n_neurons]
        b_coeff_I = BETA_TOTAL[end-n_output+1:end]

        #Calculate x^{i+1} value:
        xk_NEXT = calculate_root(
            Z_coeff, L_coeff, J_coeff, Y_coeff, b_coeff_I, c_coeff_I,
            P, Q, rI;
            solve_mode = solve_mode,
            solverAttributes = solverAttributes
        )

        #Normalize x^{i+1} value and store in ODE output trajectory:
        uk_NEXT = xk_NEXT[1:2:end]
        vk_NEXT = xk_NEXT[2:2:end]

        m = sqrt.(uk_NEXT.^2 +  vk_NEXT.^2)
        xk_OUTPUT[1:2:end, i+1] = uk_NEXT ./ m
        xk_OUTPUT[2:2:end, i+1] = vk_NEXT ./ m

        if i % 100 == 0
            println("Observing ", i, "th iteration.")
        end #if
    end #for 

    return xk_OUTPUT
end #function

function solve_BDF_ODE(
    INIT_COND::Array{Float64}, 
    dt::Float64, 
    occurences::Int64,
    Z_coeff::Matrix{Float64}, 
    L_coeff::LowerTriangular{Float64, Matrix{Float64}}, 
    J_coeff::Matrix{Float64}, 
    Y_coeff::Matrix{Float64}, 
    BETA_BASE::Array{Float64}, 
    BETA_TIME::Array{Float64};
    solve_mode::SolutionApproachEq = AbsNormal.BY_LCP,
    order::Int64 = 5,
    solverAttributes = (MOI.Silent() => true,)
)
    #Error checking:
    if order > 6
        throw("Method only defined until order 6.")
    end 

    #Neuron array information:
    n_input = size(INIT_COND, 1); n_output = size(J_coeff, 1); n_neurons = size(L_coeff, 1);
    
    #Set up initial condition:
    xk_OUTPUT = zeros(n_input, occurences+1)
    xk_OUTPUT[:, 1] = INIT_COND
    timeI = [dt * i for i in 0:occurences]

    #Calculate un-changing coefficient P and Q:
    P = I(n_output)
    Q = - dt * I(n_output)

    #Define BDF coefficients:
    Q_BDF = [1, 2/3, 6/11, 12/25, 60/137, 60/147]
    r_BDF = [
        [-1], #BDF1
        [1/3, -4/3], #BDF2 
        [-2/11, 9/11, -18/11], # BDF3
        [3/25, -16/25, 36/25, -48/25], # BDF4
        [-12/137, 75/137, -200/137, 300/137, -300/137], # BDF5
        [10/147, -72/147, 225/147, -400/147, 450/147, -360/147] # BDF6
    ]

    #For each time-step:
    for i in 1:occurences
        #Account for bootstrapping coefficient Q value:
        if i < order
            QI = Q * Q_BDF[i]
        else 
            QI = Q * Q_BDF[order]
        end #if 

        #Account for bootstrapping coefficient r value:
        if i < order 
            rI = xk_OUTPUT[:, 1:i] * r_BDF[i]
        else 
            rI = xk_OUTPUT[:, (i+1-order):i] * r_BDF[order]
        end 

        #Adjust coefficient β for time information:
        BETA_TOTAL = BETA_BASE + BETA_TIME*timeI[i]
        c_coeff_I = BETA_TOTAL[1:n_neurons]
        b_coeff_I = BETA_TOTAL[end-n_output+1:end]

        #Calculate x^{i+1} value:
        xk_OUTPUT[:, i+1] = calculate_root(
            Z_coeff, L_coeff, J_coeff, Y_coeff, b_coeff_I, c_coeff_I,
            P, QI, rI;
            solve_mode = solve_mode,
            solverAttributes = solverAttributes
        )
    end #for 

    return xk_OUTPUT
end #function 


end #module