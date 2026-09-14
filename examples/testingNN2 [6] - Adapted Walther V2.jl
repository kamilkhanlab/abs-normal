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

    THETA = Array{Matrix{Float64}}(undef, numLayers);	#matrix to store Θ values 
    CCOEFF = Array{Matrix{Float64}}(undef, numLayers);	#matrix to store c values
    for i in 1:numLayers #layer 1 to n
        THETA[i] = xf_THETA[i][:]
        CCOEFF[i] = xf_CCOEFF[i][:]
    end #for 

    #number of neurons per layer
    n = [size.(THETA, 2); size(THETA[end], 1)]

    return numLayers, n, THETA, CCOEFF
end #function 

#Generate abs-normal coefficients:
function generate_coefficients(
    n,
    numLayers, #number of NN layers
    THETA,     #matrix of NN coefficients
    CCOEFF     #matrix of NN intercepts
)
    #GENERATE COEFFICIENTS:
    #Calculate neuron array information
    n_total = sum(n[2:end])

    #Calculate intermediate variables
    intermediate = cat(THETA..., dims=(1,2))
    META = zeros(n_total, n_total)
    META[(n[2]+1):end, 1:(end-n[end])] = intermediate[(n[2]+1):end, (n[1]+1):end]
    META = I(n_total) - META

    #Set up a coefficient matrix ζ (ZETA)  
    #TODO: Remove rounding. 
    ZETA = META \ intermediate

    #Set up a coefficient matrix β (BETA)
    #Note: Add time_intermediate*timeval to this at each iteration.
    BETA_base = cat(CCOEFF..., dims=1)
    BETA_base = META \ BETA_base
    BETA_time = ZETA[:, n[1]]
    ZETA = [ZETA[:, 1:n[end]] ZETA[:, (n[end]+1):end]]

    return ZETA, BETA_base, BETA_time
end #function 

#Calculate ODE points:
function calculate_ODE(
    n,              #number of neurons per NN layer
    xk_init_cond,   #initial condition
    dt,             #time step 
    occurences,      #number of time steps
    ZETA, BETA_base, BETA_time
)
    n[1] = n[end]
    #Set up outputs:
    xk_output = zeros(n[end], occurences+1)
    xk_output[:, 1] = xk_init_cond
    timeI = [dt * i for i in 0:occurences]

    #Seperate ζ into ANF Coeffs
    n_input = n[1]; n_neurons = sum(n[2:end-1]); n_total = n_input + n_neurons

    Z_coeff = ZETA[1:n_neurons,         1:n_input]
    J_coeff = ZETA[end-n_input+1:end,   1:n_input]

    L_coeff = LowerTriangular(ZETA[1:n_neurons, n_input+1:n_total])
    Y_coeff = ZETA[end-n_input+1:end,           n_input+1:n_total]

    #Account for Euler equation for J and Y:
    J_coeff = 1 .- dt * J_coeff
    Y_coeff =    - dt * Y_coeff 

    #For each time-step:
    for i in 1:occurences
        # BETA_base = cat(CCOEFF..., dims=1)
        # BETA_base[1:24] = BETA_base[1:24] + time_intermediate[1:24]*timeI[i]
        # BETA_base[25:36] = BETA_base[25:36] + 0.5*THETA[2]*BETA_base[1:24]
        # BETA_base[37:39] = BETA_base[37:39] + 0.5*THETA[3]*BETA_base[25:36]
        # BETA_base[40:43] = BETA_base[40:43] + 0.5*THETA[4]*BETA_base[37:39]

        # BETA_base = cat(CCOEFF..., dims=1)
        # BETA_base[1:24] = BETA_base[1:24] + time_intermediate[1:24]*timeI[2]
        # BETA_base = META\BETA_base

        #Account for the time value
        BETA_total = BETA_base + BETA_time * timeI[i]
        #Seperate β into ANF Coeffs
        c_coeff_I = BETA_total[1:n_neurons]
        b_coeff_I = BETA_total[end-n_input+1:end]
        #Add x-value into b_coeff
        b_coeff_I = - b_coeff_I * dt + xk_output[:, i]

        #Convert ANF coefficients:
        anfI = AbsNormal.AnfCoeffs(c_coeff_I, b_coeff_I, Z_coeff, L_coeff, J_coeff, Y_coeff) 
        
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
# Z=30;   occurences=1000;   dt= 0.0010;    initialize=1;

# Z=31;   occurences=100;    dt= 0.0202;    initialize=1;
# Z=32;   occurences=1000;   dt= 0.0020;    initialize=1;
# Z=33;   occurences=200;    dt= 0.0101;    initialize=1;
# Z=34;   occurences=500;    dt=0.0040;     initialize=1;

# Z=38;   occurences=1000;   dt=0.001;      initialize=1;

#DEBUG
Z = 39;   occurences=1000;   dt=0.001;      initialize=1;

xf_CCOEFF, xf_THETA, xk_raw_data = pull_data(Z)
numLayers, n, THETA, CCOEFF = adjust_data(xf_CCOEFF, xf_THETA)

#account_for_time in ODE:
if n[1] == n[end]
    throw("Wrong dataset!")
end #if 

# if n[1] != n[end]
#     time_intermediate = zeros(sum(n[2:end]))
#     time_intermediate[1:n[2]] = THETA[1][:, end]
#     THETA[1] = THETA[1][:, 1:end-1]
#     n[1] = n[end]
# end #if 

ZETA, BETA_base, BETA_time = generate_coefficients(n, numLayers, THETA, CCOEFF)

xk_init_cond = xk_raw_data[:, 1]
xk_output = calculate_ODE(
    n, xk_init_cond, dt, occurences, 
    ZETA, BETA_base, BETA_time
)

# plot_it(xk_raw_data, xk_output, n[end])
