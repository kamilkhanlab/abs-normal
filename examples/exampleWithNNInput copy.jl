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
    #(a.1) Pull data from excel sheet:
    if pull_raw_data == true 
        xk_RAW_DATA = CSV.read(string("data/DataSet", Z,"/FitRNet_Storage_RawData.csv"), CSV.Tables.matrix; header=false)'
    else 
        xk_RAW_DATA = false 
    end #if 

    #(a.2) Pull data from CSV files:
    files_THETA = filter(f -> endswith(f, ".csv"), readdir(string("data/DataSet",Z,"/THETA"), join=true))
    files_CCOEFF = filter(f -> endswith(f, ".csv"), readdir(string("data/DataSet",Z,"/CCOEFF"), join=true))

    #(b) Set-up coefficients Θ and c:
    numLayers = length(readdir(string("data/DataSet",Z,"/THETA")))
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

##
#Indicate example #60: Kuramoto model: (60 is PI and 61 is non-PI)
# Z = 37;   occurrences=1000; dt= 0.001; account_for_time = true; pull_raw_data = true; ALPHA = 0.0; #functional
Z = 61; account_for_time = true; pull_raw_data = true; ALPHA = 0.0;
# dt= 0.002; occurrences=3000; 
dt = 0.0005714; occurrences=70_000;

xk_OMEGA = CSV.read(string("data/DataSet", Z,"/FitRNet_Storage_OMEGA.csv"), CSV.Tables.matrix; header=false);

#Pull NN parameter data from CSV files: 
numLayers, xk_RAW_DATA, THETA, CCOEFF_BASE, CCOEFF_TIME = example_set_up(Z, account_for_time = account_for_time, pull_raw_data = pull_raw_data)
xk_init_cond = Float64.(xk_RAW_DATA[:, 1])

Z_coeff, L_coeff, J_coeff, Y_coeff, BETA_BASE, BETA_TIME = NNInput.generate_RELU(THETA, CCOEFF_BASE, CCOEFF_TIME)

xk_OUTPUT = NNInput.solve_ODE_PI(
    xk_init_cond, dt, occurrences,
    Z_coeff, L_coeff, J_coeff, Y_coeff, BETA_BASE, BETA_TIME,
    solve_mode = NNInput.BY_LCP
)

uk_OUTPUT = xk_OUTPUT[1:2:end, :]
vk_OUTPUT = xk_OUTPUT[2:2:end, :]

fNN_OUTPUT = atan.(uk_OUTPUT, vk_OUTPUT)
fNN_OUTPUT = unwrap(fNN_OUTPUT, dims=2)

time_length = collect(0:dt:(occurrences*dt))

p=[plot(time_length, xk_RAW_DATA'); plot(time_length, fNN_OUTPUT')]

println("ALL DONE!")