include("../src/AbsNormalWithoutBARON.jl")
using .AbsNormal, LinearAlgebra, TimerOutputs, JuMP
using XLSX

function pull_data(xf_CCOEFF, xf_THETA)
    numLayers = size(XLSX.sheetnames(xf_CCOEFF), 1)

    THETA = Array{Any}(undef, numLayers);	#matrix to store Θ values 
    CCOEFF = Array{Any}(undef, numLayers);	#matrix to store c values
    for i in 1:numLayers #layer 1 to n
        THETA[i] = xf_THETA[i][:]
        CCOEFF[i] = xf_CCOEFF[i][:]
    end #for 

    #Let n be the number of neurons per layer +1
    #The first value is the number of initial variables x
    #The last value is the number of output variabls y
    n = size.(THETA, 2) 
    n = vcat(n, n[1])

    return numLayers, n, THETA, CCOEFF
end #function 

function generate_RELU(numLayers, n, THETA, CCOEFF)
    #Set up indexing for each layer
    #Indexing = [1:n0, n0+1:n1, n1+:n2, n2+1:n3, ...]
    colIdx = vcat(0, cumsum(n))
    colIdx = [colIdx[i]+1:colIdx[i+1] for i in 1:numLayers]
    rowIdx = vcat(0, cumsum(n[2:end]))
    rowIdx = [rowIdx[i]+1:rowIdx[i+1] for i in 1:numLayers]

    #Set up a coefficient matrix ZETA 
    numRow = sum(n[2:end])
    numCol = sum(n[1:end-1])
    ZETA = zeros(numRow, numCol)
    ZETA[rowIdx[1], colIdx[1]] = THETA[1]
    for i in 2:numLayers
        ZETA[rowIdx[i], colIdx[i]] = 0.5 .* THETA[i]
        for j in 1:(i-1)
            ZETA[rowIdx[i], colIdx[j]] = 0.5 .* THETA[i] * ZETA[rowIdx[i-1], colIdx[j]]
        end #for 
    end #for 

    #Set up coefficient vector BETA
    BETA = zeros(numRow)
    BETA[rowIdx[1]] = CCOEFF[1]
    for i in 2:numLayers
        BETA[rowIdx[i]] = 0.5 .* THETA[i] * BETA[rowIdx[i-1]] + CCOEFF[i]
    end #for 

    #Seperate into ANF Coeffs
    Z_coeff = ZETA[1:numRow-n[end],  colIdx[1]]
    J_coeff = ZETA[rowIdx[end],     colIdx[1]]
    L_coeff = LowerTriangular(ZETA[1:numRow-n[end],  n[1]+1:numCol])
    Y_coeff = ZETA[rowIdx[end],     n[1]+1:numCol]
    c_coeff = BETA[1:numRow-n[end]]
    b_coeff = BETA[rowIdx[end]]

    return Z_coeff, J_coeff, L_coeff, Y_coeff, c_coeff, b_coeff
end #function 

function testDataSet(Z)
    ##Set up system:
    #Import weights and biases data from excel 
    xf_CCOEFF = XLSX.readxlsx(string("data/DataSet", Z, "/FitRNet_Storage_CCOEFF.xlsx"))
    xf_THETA = XLSX.readxlsx(string("data/DataSet", Z,"/FitRNet_Storage_THETA.xlsx"))

    numLayers, n, THETA, CCOEFF = pull_data(xf_CCOEFF, xf_THETA)
    Z_coeff, J_coeff, L_coeff, Y_coeff, c_coeff, b_coeff = generate_RELU(numLayers, n, THETA, CCOEFF)

    ##Solve system for root of the function, i.e., f(x) = 0: 
    anf = AbsNormal.AnfCoeffs(c_coeff, b_coeff, Z_coeff, L_coeff, J_coeff, Y_coeff)  
    try
        rootLCP, terminationStatusLCP = AbsNormal.solve_pa_equation(anf, approach=AbsNormal.BY_LCP);
        println("## Solving LCP...")
        @show round.(rootLCP, digits=3)
        @show terminationStatusLCP
    catch err
        println("LCP solver failed to run!\n")
    end 

    println("## Solving MLCP...")
    try
        rootMLCP, terminationStatusMLCP = AbsNormal.solve_pa_equation(anf, approach=AbsNormal.BY_MLCP);
        @show round.(rootMLCP, digits=3)
        @show terminationStatusMLCP
    catch err
        println("MLCP solver failed to run!\n")
    end 
end #function 

testDataSet(15)

