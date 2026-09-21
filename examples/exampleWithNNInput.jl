#Necessary Packages:
include("../src/AbsNormalNNInput.jl")
using .NNInput, LinearAlgebra, XLSX, PlotlyJS

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
Z = 42; account_for_time = false; pull_raw_data = false;
modelMu = [0.1881; 0.2126; 0.2495]; modelSigma = [5.6921; 5.7793; 5.7065];

numLayers, xk_RAW_DATA, THETA, CCOEFF_BASE = example_set_up(Z, account_for_time = account_for_time, pull_raw_data = pull_raw_data)

Z_coeff, L_coeff, J_coeff, Y_coeff, b_coeff, c_coeff = NNInput.generate_RELU(THETA, CCOEFF_BASE)
n_output = size(J_coeff, 1)
P = zeros(n_output, n_output); r = zeros(n_output); Q = I(n_output);
ROOT_VAL = NNInput.calculate_root(
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

Z_coeff, L_coeff, J_coeff, Y_coeff, BETA_BASE, BETA_TIME = NNInput.generate_RELU(THETA, CCOEFF_BASE, CCOEFF_TIME)
xk_OUTPUT = NNInput.solve_ODE(
    xk_init_cond, dt, occurences,
    Z_coeff, L_coeff, J_coeff, Y_coeff, BETA_BASE, BETA_TIME,
    solve_mode = "LCP"
)

plot_it(xk_RAW_DATA, xk_OUTPUT, 4)
