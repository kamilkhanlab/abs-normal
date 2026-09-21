#THIS FILE WAS NOT TOUCHED AFTER JUL-16 2026
##(1) Call Packages
include("../src/AbsNormalWithoutBARON.jl")
using .AbsNormal, LinearAlgebra, TimerOutputs, JuMP
using XLSX, PlotlyJS

##(1) PLOT GRAPHS (Raw Data vs. FitRNet vs. abs-norm) for Dataset Z = 21
function plot_it(
    xk_raw_data, 
    xk_output,
    num_outputs
)
    if num_outputs == 4
        p = make_subplots(rows=2,cols=2, subplot_titles=["C1" "C3"; "C2" "C4"],
            vertical_spacing=0.15, horizontal_spacing=0.06)
        rowloc = [1,1,2,2]
        colloc = [1,2,1,2]
    elseif num_outputs == 3
        p = make_subplots(rows=2,cols=2, subplot_titles=["C1" "C3"; "C2" ""],
            vertical_spacing=0.23, horizontal_spacing=0.06)
        rowloc = [1,1,2]
        colloc = [1,2,1]
        relayout!(p, xaxis_title="Time", xaxis2_title="Time", xaxis3_title="Time")
    elseif num_outputs == 2
        p = make_subplots(rows=1,cols=2, subplot_titles=["C1" "C2"],
            vertical_spacing=0.15, horizontal_spacing=0.06)
        rowloc = [1,1]
        colloc = [1,2]
    elseif num_outputs == 1
        p = make_subplots(rows=1,cols=1, subplot_titles=["C1" "C2"],
            vertical_spacing=0.15, horizontal_spacing=0.06)
        rowloc = [1,1]
        colloc = [1,2]
    end #if 

    for i in 1:num_outputs
        if i == 1
            add_trace!(p, scatter(y=xk_raw_data'[:, i], mode="markers", name="raw-data", marker_color="black",legendgroup="raw-data"), row=rowloc[i],col=colloc[i])
            add_trace!(p, scatter(y=xk_output'[:, i], mode="lines", name="abs-norm", line_color="red",legendgroup="abs-norm"), row=rowloc[i],col=colloc[i])
        else
            add_trace!(p, scatter(y=xk_raw_data'[:, i], mode="markers", name="raw-data $i", marker_color="black",legendgroup="raw-data", showlegend=false), row=rowloc[i],col=colloc[i])
            add_trace!(p, scatter(y=xk_output'[:, i], mode="lines", name="abs-norm $i", line_color="red",legendgroup="abs-norm",showlegend=false), row=rowloc[i],col=colloc[i])
        end #if 
    end #for

    #=
    #Add xaxis titles to the subplots 
    relayout!(p, xaxis=attr(
                title_text="X Axis for Subplot 1",
                title_font_size=12,
                title_standoff=8),
                xaxis2=attr(
                title_text="X Axis for Subplot 2",
                title_font_size=12,
                title_standoff=8),
                xaxis3=attr(
                title_text="X Axis for Subplot 3",
                title_font_size=12,
                title_standoff=8),
                xaxis4=attr(
                title_text="X Axis for Subplot 4",
                title_font_size=12,
                title_standoff=8),
    )
    =#

    return p
end #function

##(2) TEST DATASETS:
#Pull NN coefficient data from excel files generated via FitRNet in MATLAB 
function pull_data(
    Z   #dataset number 
)
    xf_CCOEFF = XLSX.readxlsx(string("data/DataSet", Z, "/FitRNet_Storage_CCOEFF.xlsx"))
    xf_THETA = XLSX.readxlsx(string("data/DataSet", Z,"/FitRNet_Storage_THETA.xlsx"))
    xk_raw_data = XLSX.readxlsx(string("data/DataSet", Z,"/FitRNet_Storage_RawData.xlsx"))[1][:]'

    return xf_CCOEFF, xf_THETA, xk_raw_data
end #function 

function adjust_data(
    xf_CCOEFF,  #XLSX file of NN coefficients 
    xf_THETA    #XLSX file of NN intercepts 
)
    numLayers = size(XLSX.sheetnames(xf_CCOEFF), 1)

    THETA = Array{Any}(undef, numLayers);	#matrix to store Θ values 
    CCOEFF = Array{Any}(undef, numLayers);	#matrix to store c values
    for i in 1:numLayers #layer 1 to n
        THETA[i] = xf_THETA[i][:]
        CCOEFF[i] = xf_CCOEFF[i][:]
    end #for 

    #number of neurons per layer
    n = [size.(THETA, 2); size(THETA[end], 1)]

    return numLayers, n, THETA, CCOEFF
end #function 

#Generate abs-normal coefficients:
function generate_RELU(
    numLayers, #number of NN layers
    THETA,     #matrix of NN coefficients
    CCOEFF     #matrix of NN intercepts
)
    #Calculate neuron array information
    n = [size.(THETA, 2); size(THETA[end], 1)]
    n_input = n[1]; n_output = n[end]; n_array = n[2:end-1]
    n_neurons = sum(n_array)
    n_swap = n_input >= n_output ? n_input : n_output

    #Set up column and row indexing corresponding to each NN layer
    #Indexing = [1:n0, n0+1:n1, n1+:n2, n2+1:n3, ...]
    rowIdx = vcat(0, cumsum([n_array; n_output]))
    rowIdx = [rowIdx[i]+1:rowIdx[i+1] for i in 1:numLayers]
    colIdx = vcat(0, cumsum([n_input; n_array]))
    colIdx = [colIdx[i]+1:colIdx[i+1] for i in 1:numLayers]

    #Set up a coefficient matrix ζ (ZETA)  
    #Safety feature on matrix size implemented 
    intermediate = zeros(n_neurons+n_swap, n_neurons)
    for i in 1:numLayers - 1
        intermediate[rowIdx[i+1], rowIdx[i]] = 0.5 .* THETA[i+1]
    end #for 
    META = I(n_neurons+n_swap) - [intermediate zeros(n_neurons+n_swap, n_swap)] 
    intermediate = [zeros(n_neurons+n_swap, n_swap) intermediate]
    intermediate[rowIdx[1],colIdx[1]] = THETA[1]
    ZETA = META \ intermediate

    #Set up a coefficient matrix β (BETA)
    BETA = [reduce(vcat, CCOEFF); zeros(abs(n_input - n_output))]
    BETA = META \ BETA

    #Seperate ζ and β into ANF Coeffs
    Z_coeff = ZETA[1:n_neurons,           1:n_swap]
    J_coeff = ZETA[end-n_swap+1:end,    1:n_swap]

    L_coeff = LowerTriangular(ZETA[1:n_neurons,   n_swap+1:(n_neurons+n_swap)])
    Y_coeff = ZETA[end-n_swap+1:end,            n_swap+1:(n_neurons+n_swap)]

    c_coeff = BETA[1:n_neurons]
    b_coeff = BETA[end-n_swap+1:end]
    
    return Z_coeff, L_coeff, c_coeff, J_coeff, Y_coeff, b_coeff
end #function 

function adjust_RELU_ODE(
    Z_coeff,   #abs-normal coefficients  
    L_coeff, 
    c_coeff, 
    J_coeff,
    Y_coeff, 
    b_coeff,
    P, Q, r
)
    Z_coeff_ = Z_coeff
    L_coeff_ = L_coeff
    c_coeff_ = c_coeff

    J_coeff_ = P + Q * J_coeff
    Y_coeff_ = Q * Y_coeff
    b_coeff_ = r + Q * b_coeff

    return Z_coeff_, L_coeff_, c_coeff_, J_coeff_, Y_coeff_, b_coeff_
end #function 

#Calculate ODE points:
function calculate_ODE(
    n,              #number of neurons per NN layer
    xk_init_cond,   #initial condition
    initialize,     #size of initial condition  
    dt,             #time step 
    occurences,      #number of time steps
    Z_coeff, L_coeff, c_coeff, J_coeff, Y_coeff, b_coeff;
    account_for_time = false
)
    n_input = n[1]; n_output = n[end];
    n_swap = n_input >= n_output ? n_input : n_output

    #Set up initial condition:
    xk_output = zeros(n[end], occurences+1)
    time = [dt * i for i in 0:occurences]
    xk_output[:, 1:initialize] = xk_init_cond[:, 1:initialize]

    #Calculate un-changing coefficient P and Q:
    P = I(n_swap)
    Q = - dt * I(n_swap)

    #For each time-step:
    for i in initialize:occurences
        if account_for_time == true
            r = -[xk_output[:, i]; time[i]]
        else 
            r = -xk_output[:, i]
        end 
        #Calculate adjusted ANF Coefficients:

        Z_coeff_I, L_coeff_I, c_coeff_I, J_coeff_I, Y_coeff_I, b_coeff_I = adjust_RELU_ODE(
            Z_coeff, L_coeff, c_coeff, J_coeff, Y_coeff, b_coeff, P, Q, r
        )
        # anfI = AbsNormal.AnfCoeffs(c_coeff_I, b_coeff_I[1:n_output], Z_coeff_I, L_coeff_I, J_coeff_I[1:n_output,:], Y_coeff_I[1:n_output,:])
        anfI = AbsNormal.AnfCoeffs(c_coeff_I, b_coeff_I, Z_coeff_I, L_coeff_I, J_coeff_I, Y_coeff_I) 
        
        #Calculate root via LCP:
        try
            rootLCP, terminationStatusLCP = AbsNormal.solve_pa_equation(anfI, approach=AbsNormal.BY_LCP);
            xk_output[:, i+1] = rootLCP[1:n_output]
            if any(isnan.(rootLCP))
                println(terminationStatusLCP)
            end 
        catch err
            println("LCP solver failed to run!\n")
            break
        end #try 
    end #for 

    return xk_output
end #function 

##(3) Run ODE calculation:
#Dataset 21,22 :             dt=0.0202, occurences=100
#Dataset 23,24,25,26,27,31:  dt=0.0101, occurences=100
#Dataset 28,29:              dt=0.0010, occurences=1000
#Dataset 32:                 dt=0.0020, occurences=1000

# Z=27;   occurences=100;    dt= 0.0101;    initialize=1;
# Z=30;   occurences=1000;   dt= 0.0010;    initalize=1;

# Z=31;   occurences=100;    dt= 0.0202;    initalize=1;
# Z=32;   occurences=1000;   dt= 0.0020;    initalize=1;
# Z=33;   occurences=200;    dt= 0.0101;    initalize=1;
# Z=34;   occurences=500;    dt=0.0040;     initialize=1;

# Z=38;   occurences=1000;   dt=0.002;      initialize=1;
Z=39;   occurences=1000;   dt=0.0010;     initialize=1;


xf_CCOEFF, xf_THETA, xk_raw_data = pull_data(Z)
numLayers, n, THETA, CCOEFF = adjust_data(xf_CCOEFF, xf_THETA)
Z_coeff, L_coeff, c_coeff, J_coeff, Y_coeff, b_coeff = generate_RELU(numLayers, THETA, CCOEFF)
xk_output = calculate_ODE(
    n, xk_raw_data, initialize, dt, occurences, 
    Z_coeff, L_coeff, c_coeff, J_coeff, Y_coeff, b_coeff,
    account_for_time = true  
)

plot_it(xk_raw_data, xk_output, n[end])
