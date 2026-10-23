include("../src/AbsNormalNNInput.jl")
using CSV, SafeTensors

function safetensor_set_up(
    Z::Int64    #example dataset number 
)
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

    return THETA, CCOEFF_BASE
end #function 

Z = 31;
model_tensors = Dict{String, AbstractArray}()
THETA, CCOEFF_BASE = safetensor_set_up(Z)

for (i, (weights, biases)) in enumerate(zip(THETA, COEFF_BASE))
    model_tensors["layer $(i).weight"] = weights
    model_tensors["layer $(i).bias"] = biases
end #for 

SafeTensors.serialize("testfile$(Z).safetensors", model_tensors)

# SafeTensors.deserialize("testfile$(Z).safetensors")