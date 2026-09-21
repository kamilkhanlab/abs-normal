##(1) Call Packages
include("../src/AbsNormalWithoutBARON.jl")
using .AbsNormal, LinearAlgebra, TimerOutputs, JuMP
using XLSX, PlotlyJS

##(2) Pull NN coefficient data from excel files generated via FitRNet in MATLAB 
function pull_data(
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
    n = size.(THETA, 2) 
    n = vcat(n, n[1])

    return numLayers, n, THETA, CCOEFF
end #function 

##(3) Calculate ANF Coefficients
function generate_RELU(
    numLayers, #number of NN layers
    n,         #number of neurons per NN layer
    THETA,     #matrix of NN coefficients
    CCOEFF,    #matrix of NN intercepts
)
    #Set up column and row indexing corresponding to each NN layer
    #Indexing = [1:n0, n0+1:n1, n1+:n2, n2+1:n3, ...]
    colIdx = vcat(0, cumsum(n))
    colIdx = [colIdx[i]+1:colIdx[i+1] for i in 1:numLayers]
    rowIdx = vcat(0, cumsum(n[2:end]))
    rowIdx = [rowIdx[i]+1:rowIdx[i+1] for i in 1:numLayers]
    
    numRow = sum(n[2:end])
    numCol = sum(n[1:end-1])

    #Set up a coefficient matrix ζ (ZETA) 
    ZETA = zeros(numRow, numCol)
    ZETA[rowIdx[1], colIdx[1]] = THETA[1]
    for i in 2:numLayers
        ZETA[rowIdx[i], colIdx[i]] = 0.5 .* THETA[i] 
        for j in 1:(i-1)
            ZETA[rowIdx[i], colIdx[j]] = 0.5 .* THETA[i] * ZETA[rowIdx[i-1], colIdx[j]] 
        end #for 
    end #for 

    #Set up coefficient vector β (BETA)
    BETA = zeros(numRow)
    BETA[rowIdx[1]] = CCOEFF[1] 
    for i in 2:numLayers
        BETA[rowIdx[i]] = 0.5 .* THETA[i] * BETA[rowIdx[i-1]] + CCOEFF[i] 
    end #for 

    #Seperate ζ and β into ANF Coeffs
    println(n)
    Z_coeff = ZETA[1:numRow-n[end],  colIdx[1]]
    J_coeff = ZETA[rowIdx[end],     colIdx[1]]
    L_coeff = LowerTriangular(ZETA[1:numRow-n[end],  n[1]+1:numCol])
    Y_coeff = ZETA[rowIdx[end],     n[1]+1:numCol]
    c_coeff = BETA[1:numRow-n[end]]
    b_coeff = BETA[rowIdx[end]]

    return Z_coeff, J_coeff, L_coeff, Y_coeff, c_coeff, b_coeff, rowIdx, colIdx
end #function 

##(4) Adjust ANF matrix for ODE root calculation:
function adjust_RELU(
    Z_coeff, J_coeff, L_coeff, Y_coeff, c_coeff, b_coeff, #Default ANF coefficients 
    n,      #number of neurons per NN layer
    xk,     #Value of x(k)
    dt      #Value of ∆t 
)
    Z_coeff_ = Z_coeff
    L_coeff_ = L_coeff
    c_coeff_ = c_coeff

    J_coeff_ = I(n[end]) - dt .* J_coeff
    Y_coeff_ = -dt .* Y_coeff
    b_coeff_ = -xk - dt .* b_coeff

    return Z_coeff_, J_coeff_, L_coeff_, Y_coeff_, c_coeff_, b_coeff_
end #function 

##(4.5) Calculate ODE via implicit Euler using abs-norm solver
function ODESolve(
    Z_coeff, J_coeff, L_coeff, Y_coeff, c_coeff, b_coeff, #ANF Coefficients
    n,           #Number of neurons per NN layer 
    xf_initcond, #Transposed matrix of initial condition data 
    dt,          #Size of ∆t
    occurences,  #Number of iterations 
    num_outputs; #Number of outputs to f(x)
    initialize=3 #Initial condition (initialize first 3 values from raw data) 
)
    #Set up initial condition:
    xk = zeros(num_outputs, occurences+1)
    xk[:, 1:initialize] = xf_initcond[:, 1:initialize]

    #For each iteration:
    for i in initialize:occurences
        #Calculate adjusted ANF Coefficients:
        Z_coeff_I, J_coeff_I, L_coeff_I, Y_coeff_I, c_coeff_I, b_coeff_I = adjust_RELU(
            Z_coeff, J_coeff, L_coeff, Y_coeff, c_coeff, b_coeff, n, xk[:, i], dt)
        anfI = AbsNormal.AnfCoeffs(c_coeff_I, b_coeff_I, Z_coeff_I, L_coeff_I, J_coeff_I, Y_coeff_I)  
        
        #Calculate root via LCP:
        try
            rootLCP, terminationStatusLCP = AbsNormal.solve_pa_equation(anfI, approach=AbsNormal.BY_LCP);
            xk[:, i+1] = rootLCP 

            if any(isnan.(rootLCP))
                println(terminationStatusLCP)
            end 
        catch err
            println("LCP solver failed to run!\n")
            break
        end #try 
    end #for 

    return xk
end #function 

##(5) Validate Results: Calculate ODE via Explicit Euler via...
#DefaultNN: Original NN coefficients from FitRNet
#AbsNorm: ANF coefficients 
#Adjusted AbsNorm: Adjusted ANF coefficients (?)
function fwd_pass_input(
    xk,         #Value x(k)
    numLayers;  #Number of NN layers 
    method = "abs-norm",    #Forward pass method setting {abs-norm, default-NN}
    colIdx = false,         #Column indexing corresponding to each NN layer
    rowIdx = false,          #Row indexing corresponding to each NN layer
    Z_coeff=false, J_coeff=false, L_coeff=false, Y_coeff=false, c_coeff=false, b_coeff=false, #Default ANF Coefficients
    THETA=false, CCOEFF=false #NN Coefficients and intercepts
)
    if method == "abs-norm"
        zk = Z_coeff[rowIdx[1], :] * xk + c_coeff[rowIdx[1]]
        for i in 2:numLayers-1
            sum_rowIdx = vcat(rowIdx[1:i-1]...)
            zi = Z_coeff[rowIdx[i], :] * xk + L_coeff[rowIdx[i], sum_rowIdx] * abs.(zk) + c_coeff[rowIdx[i]]
            zk = vcat(zk, zi)
        end #for 
        zk = J_coeff * xk + Y_coeff * abs.(zk) + b_coeff

        return zk
    elseif method == "default-NN"
        zk = xk
        for i in 1:numLayers
            zk = THETA[i] * zk + CCOEFF[i]
            if i < numLayers 
                zk = max.(zk, 0)
            end #if 
        end #for 

        return zk
    end #if 
end #function

function fwd_pass_run(
    xf_initcond, #Transposed matrix of initial condition data 
    dt,          #Size of ∆t
    occurences,  #Number of iterations 
    num_outputs, #Number of outputs to f(x)
    numLayers;   #Number of NN layers
    initialize=3, #Initial condition (initialize first 3 values from raw data)
    method = "abs-norm",    #Method for f'(x) generation
    colIdx = false,         #Column indexing corresponding to each NN layer
    rowIdx = false,          #Row indexing corresponding to each NN layer
    Z_coeff=false, J_coeff=false, L_coeff=false, Y_coeff=false, c_coeff=false, b_coeff=false, #Default ANF Coefficients
    THETA=false, CCOEFF=false #NN Coefficients and intercepts
)
    #set up initial condition 
    xk_fwd_pass = zeros(num_outputs, occurences+1)
    xk_fwd_pass[:, 1:initialize] = xf_initcond[:, 1:initialize] 

    #iteratively calculate x(k) values
    for i in initialize:occurences    
        if method == "abs-norm"
            fNN = fwd_pass_input( 
                xk_fwd_pass[:, i], numLayers;
                method = method, colIdx = colIdx, rowIdx = rowIdx,
                Z_coeff=Z_coeff, J_coeff=J_coeff, L_coeff=L_coeff, Y_coeff=Y_coeff, c_coeff=c_coeff, b_coeff=b_coeff, 
            )
        elseif method == "default-NN"
            fNN = fwd_pass_input(
                xk_fwd_pass[:, i], numLayers;
                method = method, 
                THETA=THETA, CCOEFF=CCOEFF
            )
        end 
        xk_fwd_pass[:, i+1] = xk_fwd_pass[:, i] + dt .* fNN
    end #for 

    return xk_fwd_pass
end #function

##(6) Run Example Dataset: 
function calculate_it(Z, dt, occurences)
    xf_CCOEFF = XLSX.readxlsx(string("data/DataSet", Z, "/FitRNet_Storage_CCOEFF.xlsx"))
    xf_THETA = XLSX.readxlsx(string("data/DataSet", Z,"/FitRNet_Storage_THETA.xlsx"))
    xk_raw_data = XLSX.readxlsx(string("data/DataSet", Z,"/FitRNet_Storage_RawData.xlsx"))[1][:]'

    numLayers, n, THETA, CCOEFF = pull_data(xf_CCOEFF, xf_THETA)
    Z_coeff, J_coeff, L_coeff, Y_coeff, c_coeff, b_coeff, rowIdx, colIdx = generate_RELU(numLayers, n, THETA, CCOEFF)

    xk_absnorm = ODESolve(
        Z_coeff, J_coeff, L_coeff, Y_coeff, c_coeff, b_coeff,
        n,
        xk_raw_data, dt, occurences, n[1]
    )
    xk_fwd_pass = fwd_pass_run(
        xk_raw_data, dt, occurences, n[1], numLayers,
        method = "abs-norm", colIdx=colIdx, rowIdx=rowIdx,
        THETA=THETA, CCOEFF=CCOEFF,
        Z_coeff=Z_coeff, J_coeff=J_coeff, L_coeff=L_coeff, Y_coeff=Y_coeff, c_coeff=c_coeff, b_coeff=b_coeff, 
    )

    return xk_raw_data, xk_absnorm, xk_fwd_pass, n[1] 
end #function 

##(6.5) PLOT GRAPHS (Raw Data vs. FitRNet vs. abs-norm) for Dataset Z = 21
function plot_it(xk_raw_data, xk_absnorm, xk_fwd_pass, num_outputs)
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
            # add_trace!(p, scatter(y=xk_fwd_pass'[:, i], mode="lines", name="fwd-pass", line_color="blue",legendgroup="fwd-pass"), row=rowloc[i],col=colloc[i])
            add_trace!(p, scatter(y=xk_absnorm'[:, i], mode="lines", name="abs-norm", line_color="red",legendgroup="abs-norm"), row=rowloc[i],col=colloc[i])
        else
            add_trace!(p, scatter(y=xk_raw_data'[:, i], mode="markers", name="raw-data $i", marker_color="black",legendgroup="raw-data", showlegend=false), row=rowloc[i],col=colloc[i])
            # add_trace!(p, scatter(y=xk_fwd_pass'[:, i], mode="lines", name="fwd-pass $i", line_color="blue",legendgroup="fwd-pass",showlegend=false), row=rowloc[i],col=colloc[i])
            add_trace!(p, scatter(y=xk_absnorm'[:, i], mode="lines", name="abs-norm $i", line_color="red",legendgroup="abs-norm",showlegend=false), row=rowloc[i],col=colloc[i])
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

#TEST DATASETS:
#Dataset 21,22 :          dt=0.0202, occurences=100
#Dataset 23,24,25,26,27 : dt=0.0101, occurences=100
#Dataset 28,29 :          dt=0.0010, occurences=1000

xk_raw_data, xk_absnorm, xk_fwd_pass, num_outputs = calculate_it(27, 0.0101, 100)
plot_it(xk_raw_data, xk_absnorm, xk_fwd_pass, num_outputs)

##(7) Walther Formulation
Z=27; occurrences=100; dt=0.0101;
xf_CCOEFF = XLSX.readxlsx(string("data/DataSet", Z, "/FitRNet_Storage_CCOEFF.xlsx"))
xf_THETA = XLSX.readxlsx(string("data/DataSet", Z,"/FitRNet_Storage_THETA.xlsx"))
xk_raw_data = XLSX.readxlsx(string("data/DataSet", Z,"/FitRNet_Storage_RawData.xlsx"))[1][:]'

numLayers, n, THETA, CCOEFF = pull_data(xf_CCOEFF, xf_THETA)
Z_coeff, J_coeff, L_coeff, Y_coeff, c_coeff, b_coeff, rowIdx, colIdx = generate_RELU(numLayers, n, THETA, CCOEFF)
