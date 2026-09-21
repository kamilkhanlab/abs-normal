#Use testingNN2 [6] - Adapted Walther V4 
##(1) Set-up packages:
#Necessary Packages:
include("../src/AbsNormalWithoutBARON.jl")
using .AbsNormal, LinearAlgebra, TimerOutputs, JuMP
#Optional Packages:
using XLSX, PlotlyJS

##(2) External function: Pull coefficient information:
function example_set_up(
    Z::Int64;                           #example dataset number 
    account_for_time::Bool = false,     #accounting for ODE dynamics  
    pull_raw_data::Bool = true          #pull raw data used for training NN 
)
    #(a) Pull data from excel sheet:
    xf_CCOEFF = XLSX.readxlsx(string("data/DataSet", Z, "/FitRNet_Storage_CCOEFF.xlsx"))
    xf_THETA = XLSX.readxlsx(string("data/DataSet", Z,"/FitRNet_Storage_THETA.xlsx"))
    if pull_raw_data == true 
        xk_RAW_DATA = XLSX.readxlsx(string("data/DataSet", Z,"/FitRNet_Storage_RawData.xlsx"))[1][:]'
    else 
        xk_RAW_DATA = false 
    end #if 

    #(b) Set-up coefficients Θ and c:
    numLayers = size(XLSX.sheetnames(xf_CCOEFF), 1)
    THETA = Array{Matrix{Float64}}(undef, numLayers);	#matrix to store Θ values 
    CCOEFF_BASE = Array{Matrix{Float64}}(undef, numLayers);	#matrix to store c values

    for i in 1:numLayers #layer 1 to n
        THETA[i] = xf_THETA[i][:]
        CCOEFF_BASE[i] = xf_CCOEFF[i][:]
    end #for 

    #(c) For ODEs, calculate time specific coefficient:
    if account_for_time == true 
        CCOEFF_TIME = THETA[1][:, end] 
        THETA[1] = THETA[1][:, 1:end-1]
        
        return numLayers, xk_RAW_DATA, THETA, CCOEFF_BASE, CCOEFF_TIME
    else 
        return numLayers, xk_RAW_DATA, THETA, CCOEFF_BASE
    end #if 
end #function 

##(3) Internal function: Generate ζ and β structure:
function generate_RELU(
    numLayers::Int64,                    #number of NN layers
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
    numLayers::Int64,       #number of NN layers
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

##(4) Internal Function: Root Calculate:
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

##(5) Internal Function: Calculate ODE:
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
        rI = -xk_OUTPUT[:, i]

        BETA_TOTAL = BETA_BASE + BETA_TIME*timeI[i]
        c_coeff_I = BETA_TOTAL[1:n_neurons]
        b_coeff_I = BETA_TOTAL[end-n_output+1:end]

        xk_OUTPUT[:, i+1] = calculate_root(
            Z_coeff, L_coeff, J_coeff, Y_coeff, b_coeff_I, c_coeff_I,
            P, Q, rI
        )
    end #for 

    return xk_OUTPUT
end #function

##(6) External Function: Plot information:
function plot_it(
    xk_raw_data,
    xk_model1,
    xk_model2,
    num_outputs::Int64;
    names = ["Raw Data", "Model 1", "Model 2"]
)
    #Calculate number of rows and cols to fit in the closest perfect square pattern:
    cols = round(Int, sqrt(num_outputs))
    rows = cld(num_outputs, cols)

    #Initialize subplot grid
    p = make_subplots(rows=rows, cols=cols)

    #Add traces to subplot
    for i in 1:num_outputs
        rI = cld(i, cols)
        cI = mod1(i, cols)

        if i == 1
            add_trace!(p, scatter(y = xk_raw_data'[:, i], mode = "markers", name = names[1], marker_color = "black", legendgroup = "raw-data"), row = rI, col = cI)
            add_trace!(p, scatter(y = xk_model1'[:, i], mode = "lines", name = names[2], marker_color = "red", legendgroup = "model1"), row = rI, col = cI)
            add_trace!(p, scatter(y = xk_model2'[:, i], mode = "lines", name = names[3], marker_color = "blue", legendgroup = "model2"), row = rI, col = cI)
        else 
            add_trace!(p, scatter(y = xk_raw_data'[:, i], mode = "markers", name = names[1], marker_color = "black", legendgroup = "raw-data", showlegend=false), row = rI, col = cI)
            add_trace!(p, scatter(y = xk_model1'[:, i], mode = "lines", name = names[2], marker_color = "red", legendgroup = "model1", showlegend=false), row = rI, col = cI)
            add_trace!(p, scatter(y = xk_model2'[:, i], mode = "lines", name = names[3], marker_color = "blue", legendgroup = "model2", showlegend=false), row = rI, col = cI)
        end #if 
    end #for 
    
    return p
end #function 

function plot_it(
    xk_raw_data,
    xk_model1,
    num_outputs::Int64;
    names = ["Raw Data", "Model 1", "Model 2"]
)
    #Calculate number of rows and cols to fit in the closest perfect square pattern:
    cols = round(Int, sqrt(num_outputs))
    rows = cld(num_outputs, cols)

    #Initialize subplot grid
    p = make_subplots(rows=rows, cols=cols)

    #Add traces to subplot
    for i in 1:num_outputs
        rI = cld(i, cols)
        cI = mod1(i, cols)

        if i == 1
            add_trace!(p, scatter(y = xk_raw_data'[:, i], mode = "markers", name = names[1], marker_color = "black", legendgroup = "raw-data"), row = rI, col = cI)
            add_trace!(p, scatter(y = xk_model1'[:, i], mode = "lines", name = names[2], marker_color = "red", legendgroup = "model1"), row = rI, col = cI)
        else 
            add_trace!(p, scatter(y = xk_raw_data'[:, i], mode = "markers", name = names[1], marker_color = "black", legendgroup = "raw-data", showlegend=false), row = rI, col = cI)
            add_trace!(p, scatter(y = xk_model1'[:, i], mode = "lines", name = names[2], marker_color = "red", legendgroup = "model1", showlegend=false), row = rI, col = cI)
        end #if 
    end #for 
    
    return p
end #function 

##Example Datasets:
#Solve simple linear system:
#=
    Z=42; account_for_time = false; pull_raw_data = false;
    modelMu = [0.1881; 0.2126; 0.2495]; modelSigma = [5.6921; 5.7793; 5.7065];

    numLayers, xk_RAW_DATA, THETA, CCOEFF_BASE = example_set_up(Z, account_for_time = account_for_time, pull_raw_data = pull_raw_data)

    Z_coeff, L_coeff, J_coeff, Y_coeff, b_coeff, c_coeff = generate_RELU(numLayers, THETA, CCOEFF_BASE)
    n_output = size(J_coeff, 1)
    P = zeros(n_output, n_output); r = zeros(n_output); Q = I(n_output);
    ROOT_VAL = calculate_root(
        Z_coeff, L_coeff, J_coeff, Y_coeff, b_coeff, c_coeff,
        P, Q, r;
        solve_mode = "LCP"
    )
=#

#Solve ODE system:
Z = 31;   occurences=100;    dt= 0.0202; account_for_time = true; pull_raw_data = true;
# Z = 39;   occurences=1000;   dt= 0.0010; account_for_time = true; pull_raw_data = true;

numLayers, xk_RAW_DATA, THETA, CCOEFF_BASE, CCOEFF_TIME = example_set_up(Z, account_for_time = account_for_time, pull_raw_data = pull_raw_data)
xk_init_cond = xk_RAW_DATA[:, 1]

Z_coeff, L_coeff, J_coeff, Y_coeff, BETA_BASE, BETA_TIME = generate_RELU(numLayers, THETA, CCOEFF_BASE, CCOEFF_TIME)
xk_OUTPUT = solve_ODE(
    xk_init_cond, dt, occurences,
    Z_coeff, L_coeff, J_coeff, Y_coeff, BETA_BASE, BETA_TIME,
    solve_mode = "LCP"
)

plot_it(xk_RAW_DATA, xk_OUTPUT, 4)
