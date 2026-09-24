#Necessary Packages:
include("../src/AbsNormalNNInput.jl")
using .NNInput, LinearAlgebra, XLSX, PlotlyJS
using CSV

##(2) External function: Pull coefficient information:
function example_set_up(
    Z::Int64;                           #example dataset number 
    account_for_time::Bool = false,     #accounting for ODE dynamics  
    pull_raw_data::Bool = true          #pull raw data used for training NN for plotting purposes  
)
    #(a) Pull data from excel sheet:
    if pull_raw_data == true 
        xk_RAW_DATA = XLSX.readxlsx(string("examples/example data/DataSet", Z,"/FitRNet_Storage_RawData.xlsx"))[1][:]'
        # xk_RAW_DATA = CSV.read(string("data/DataSet", Z,"/FitRNet_Storage_RawData.csv"), CSV.Tables.matrix; header=false)'
    else 
        xk_RAW_DATA = false 
    end #if 

    #(a) Pull data from CSV files:
    files_THETA = filter(f -> endswith(f, ".csv"), readdir(string("examples/example data/DataSet",Z,"/THETA"), join=true))
    files_CCOEFF = filter(f -> endswith(f, ".csv"), readdir(string("examples/example data/DataSet",Z,"/CCOEFF"), join=true))

    #(b) Set-up coefficients Θ and c:
    numLayers = length(readdir(string("examples/example data/DataSet",Z,"/THETA")))
    THETA = Array{Matrix{Float64}}(undef, numLayers);	#matrix to store Θ values 
    CCOEFF_BASE = Array{Matrix{Float64}}(undef, numLayers);	#matrix to store c values

    for (i, (file_THETA, file_CCOEFF)) in enumerate(zip(files_THETA, files_CCOEFF))
        THETA[i] = CSV.read(file_THETA, CSV.Tables.matrix; header=false)
        CCOEFF_BASE[i] = CSV.read(file_CCOEFF, CSV.Tables.matrix; header=false) 
    end #for 

    #Safety measure
    if size(THETA, 1) != size(CCOEFF_BASE, 1)
        throw("Number of THETA files must be the same as the number of CCOEFF files!")
    end #if 

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
    names = ["Raw Data", "Model 1", "Model 2"],
    subplot_titles = false
)
    #(a) Initialize subplot grid    
    #Calculate number of rows and cols to fit in the closest perfect square pattern:
    cols = round(Int, sqrt(num_outputs))
    rows = cld(num_outputs, cols)
    if subplot_titles == false 
        p = make_subplots(rows=rows, cols=cols)
    else 
        p = make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles)
    end #if 

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
    names = ["Raw Data", "Model 1", "Model 2"],
    subplot_titles = false
)
    #(a) Initialize subplot grid    
    #Calculate number of rows and cols to fit in the closest perfect square pattern:
    cols = round(Int, sqrt(num_outputs))
    rows = cld(num_outputs, cols)
    if subplot_titles == false 
        p = make_subplots(rows=rows, cols=cols)
    else 
        p = make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles)
    end #if 

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

##Solve Linear System NN:
#=
#Indicate example #42:
Z = 42; account_for_time = false; pull_raw_data = false;

#Pull NN parameter data from CSV files: 
numLayers, xk_RAW_DATA, THETA, CCOEFF_BASE = example_set_up(Z, account_for_time = account_for_time, pull_raw_data = pull_raw_data)

#Generate abs normal coefficients:
Z_coeff, L_coeff, J_coeff, Y_coeff, b_coeff, c_coeff = NNInput.generate_RELU(THETA, CCOEFF_BASE)
#Calculate root value: 
n_output = size(J_coeff, 1)
P = zeros(n_output, n_output); r = zeros(n_output); Q = I(n_output);
ROOT_VAL = NNInput.calculate_root(
    Z_coeff, L_coeff, J_coeff, Y_coeff, b_coeff, c_coeff,
    P, Q, r;
    solve_mode = NNInput.BY_LCP
)
=#
##Solve ODE NN:
#=
#Indicate example #31:
Z = 31;   occurences=100;    dt= 0.0202; account_for_time = true; pull_raw_data = true;

#Pull NN parameter data from CSV files: 
numLayers, xk_RAW_DATA, THETA, CCOEFF_BASE, CCOEFF_TIME = example_set_up(Z, account_for_time = account_for_time, pull_raw_data = pull_raw_data)
xk_init_cond = Float64.(xk_RAW_DATA[:, 1])

#Generate base-value of abs normal coefficients:
Z_coeff, L_coeff, J_coeff, Y_coeff, BETA_BASE, BETA_TIME = NNInput.generate_RELU(THETA, CCOEFF_BASE, CCOEFF_TIME)
#Iterate through and calculate ODE x^{k} values  
xk_OUTPUT = NNInput.solve_ODE(
    xk_init_cond, dt, occurences,
    Z_coeff, L_coeff, J_coeff, Y_coeff, BETA_BASE, BETA_TIME,
    solve_mode = NNInput.BY_LCP
)
=#

#Indicate example #31:
Z = 37;   occurences=1000; dt= 0.001; account_for_time = true; pull_raw_data = true;

#Pull NN parameter data from CSV files: 
numLayers, xk_RAW_DATA, THETA, CCOEFF_BASE, CCOEFF_TIME = example_set_up(Z, account_for_time = account_for_time, pull_raw_data = pull_raw_data)
xk_init_cond = Float64.(xk_RAW_DATA[:, 1])

#Generate base-value of abs normal coefficients:
Z_coeff, L_coeff, J_coeff, Y_coeff, BETA_BASE, BETA_TIME = NNInput.generate_RELU(THETA, CCOEFF_BASE, CCOEFF_TIME)
#Iterate through and calculate ODE x^{k} values  
xk_OUTPUT = NNInput.solve_ODE(
    xk_init_cond, dt, occurences,
    Z_coeff, L_coeff, J_coeff, Y_coeff, BETA_BASE, BETA_TIME,
    solve_mode = NNInput.BY_LCP
)

plot_it(xk_RAW_DATA, xk_OUTPUT, 3, subplot_titles=["C1" "C2"; "C3" ""])