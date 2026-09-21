include("../src/AbsNormalWithoutBARON.jl")
using .AbsNormal, LinearAlgebra, TimerOutputs, JuMP
using Random, Distributions 
Random.seed!(12)

#Let n be the number of neurons per layer +1
#The first value is the number of initial variables x
#The last value is the number of output variabls y
#Consider a system of 4 layers 
n = [4, 8, 6, 5, 4]   #Random generation 

#function generate_random_RELU(n)
#Generate coefficient matrices.
numLayers = size(n,1)-1
THETA = Array{Any}(undef, numLayers);	#matrix to store Θ values 
CCOEFF = Array{Any}(undef, numLayers);	#matrix to store c values
for i in 1:numLayers #layer 1 to n
    # THETA[i] = randn(n[i+1], n[i])
    # CCOEFF[i] = randn(n[i+1], 1) * 0.1
    THETA[i] = rand(Normal(0, 1), n[i+1], n[i])
    CCOEFF[i] = rand(Normal(0, 0.33), n[i+1], 1)
end #for  

#Set up indexing for each layer
#Indexing = [1:n0, n0+1:n1, n1+:n2, n2+1:n3, ...]
colIdx = vcat(0, cumsum(n))
colIdx = [colIdx[i]+1:colIdx[i+1] for i in 1:numLayers]
rowIdx = vcat(0, cumsum(n[2:end]))
rowIdx = [rowIdx[i]+1:rowIdx[i+1] for i in 1:numLayers]

#Set up a coefficient matrix ZETA 
#LowerTriangular(zeros(numRow), zeros(numCol))
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

#return Z_coeff, J_coeff, L_coeff, Y_coeff, c_coeff, b_coeff
#end function 

#  Z_coeff, J_coeff, L_coeff, Y_coeff, c_coeff, b_coeff = generate_random_RELU(n)

#Solve the equation:
##Calculate coefficients
anf = AbsNormal.AnfCoeffs(
    c_coeff, 
    b_coeff, 
    Z_coeff, 
    L_coeff, 
    J_coeff, 
    Y_coeff
)  
rootLCP, terminationStatusLCP = AbsNormal.solve_pa_equation(anf, approach=AbsNormal.BY_LCP);
println("## Solving LCP...")
@show rootLCP
@show terminationStatusLCP

println("## Solving MLCP...")
try
    rootMLCP, terminationStatusMLCP = AbsNormal.solve_pa_equation(anf, approach=AbsNormal.BY_MLCP);
    @show rootMLCP
    @show terminationStatusMLCP
catch err
    println("MLCP solver failed to run!\n")
end 

#Testing RELU forward pass
relu(x) = max.(0, x)
xRand = [2.1; 3.1; 1.6; -0.4]

function test_forward_pass(xInput, THETA, CCOEFF)
    output = xInput
    for i in 1:numLayers
        output = relu(THETA[i] * output + CCOEFF[i])
    end #for 
    return output 
end #function

# output = test_forward_pass(xRand, THETA, CCOEFF)
