#=
module AbsNormal
=================
Written by Kamil Khan

Last modified by Yulan Zhang on May 13, 2024:
(1) function verifying_minimum_of_f() in this module is to determine 
    whether a PA function f(x) with abs-normal form has a global minimum
(2) function minimize_pa_equation() in this module is to minimizing 
    a PA function f(x) in abs-normal form
(3) Randomly generate examples
=#

import JuMP, PATHSolver, BARON

using LinearAlgebra


struct AnfCoeffs
    c::Vector{Float64}
    b::Vector{Float64}
    Z::Matrix{Float64}
    L::LowerTriangular{Float64, Matrix{Float64}}
    J::Matrix{Float64}
    Y::Matrix{Float64}
end

@enum SolutionApproach1 begin
    BY_MLCP
    BY_LCP
end

@enum SolutionApproach2 begin
    BY_LPCC
    BY_MILP
end

function modify_anf_coeffs(a::AnfCoeffs)
    IMinusL = UnitLowerTriangular(I - a.L)
    (cm, Zm, Lm) = map(X -> IMinusL\X, (a.c, a.Z, I+a.L))
    bm = a.b + a.Y*cm
    Ym = a.Y*(I + Lm)
    Jm = a.J + a.Y*Zm
    return AnfCoeffs(cm, bm, Zm, Lm, Jm, Ym)
end

function modify_horizf_coeffs(a::AnfCoeffs)
    IMinusL = UnitLowerTriangular(I - a.L)
    (Zm, Lm) = map(X -> IMinusL\X, (a.Z, I+a.L))
    cm = a.c * 0.0
    bm = [1.0]
    Ym = a.Y*(I + Lm)
    Jm = a.J + a.Y*Zm
    return AnfCoeffs(cm, bm, Zm, Lm, Jm, Ym)
end

function modify_minf_coeffs(a::AnfCoeffs)
    IMinusL = UnitLowerTriangular(I - a.L)
    (cm, Zm, Lm) = map(X -> IMinusL\X, (a.c, a.Z, I+a.L))
    bm = a.b + a.Y*cm
    Ym = a.Y*(I + Lm)
    Jm = a.J + a.Y*Zm
    return AnfCoeffs(cm, bm, Zm, Lm, Jm, Ym)
end


recover_anf_coeffs(a::AnfCoeffs) = a.c, a.b, a.Z, a.L, a.J, a.Y
recover_horizf_coeffs(a::AnfCoeffs) = a.c, a.b, a.Z, a.L, a.J, a.Y
recover_minf_coeffs(a::AnfCoeffs) = a.c, a.b, a.Z, a.L, a.J, a.Y


function solve_pa_equation(
    a::AnfCoeffs;
    approach::SolutionApproach1,
    optimizer = PATHSolver.Optimizer,
    solverAttributes = ("output" => "no",)
)
    aMod = modify_anf_coeffs(a::AnfCoeffs)
    (cm, bm, Zm, Lm, Jm, Ym) = recover_anf_coeffs(aMod)

    anfModel = JuMP.Model(JuMP.optimizer_with_attributes(optimizer, solverAttributes...))

    if approach == BY_MLCP
        JuMP.@variable(anfModel, x[1:size(Zm)[2]])
        JuMP.@variable(anfModel, w[1:length(cm)] >= 0.0)

        #JuMP.@constraint(anfModel, bm + Jm*x + Ym*u .== 0.0) # not supported by PATHSolver.jl
        JuMP.@constraint(anfModel, complements(bm + Jm*x + Ym*w, x)) 
        JuMP.@constraint(anfModel, complements(cm + Zm*x + Lm*w, w))

        JuMP.optimize!(anfModel)
        xStar = JuMP.value.(x)
        wStar = JuMP.value.(w)
        terminationStatus = JuMP.termination_status(anfModel)

    elseif approach == BY_LCP
        ZJmInv = Zm/Jm
        gamma = cm - ZJmInv*bm
        S = Lm - ZJmInv*Ym

        JuMP.@variable(anfModel, w[1:length(cm)] >= 0.0)

        JuMP.@constraint(anfModel, complements(gamma + S*w, w))

        JuMP.optimize!(anfModel)
        wStar = JuMP.value.(w)
        xStar = -Jm\(bm + Ym*wStar)
        terminationStatus = JuMP.termination_status(anfModel)
        
    else
        throw(DomainError(:approach, "unsupported equation-solving approach"))
    end

    return xStar, terminationStatus, aMod
end


function verify_minimum_of_f(
    a::AnfCoeffs;
    approach::SolutionApproach1
)
    aMod = modify_horizf_coeffs(a::AnfCoeffs)
    (cm, bm, Zm, Lm, Jm, Ym) = recover_horizf_coeffs(aMod)
    
    if approach == BY_MLCP
        
        optimizer = BARON.Optimizer
        #solverAttributes = ("PrLevel" => 0,"DeltaTerm" => 1,"EpsA" => 1e-3,)
        solverAttributes = ("PrLevel" => 0,)
        
        horizfModel = JuMP.Model(JuMP.optimizer_with_attributes(optimizer, solverAttributes...))
        JuMP.@variable(horizfModel, xi[1:size(Zm)[2]])
        JuMP.@variable(horizfModel, omega[1:size(Zm)[1]] >= 0.0)

        
        JuMP.@constraint(horizfModel, [1.0] + Jm*xi + Ym*omega .== 0.0) 
        
        for i in eachindex(omega)
            JuMP.@constraints(horizfModel, begin
                        ((Zm*xi)[i] + (Matrix(Lm)*omega)[i]) * omega[i] == 0.0
                         (Zm*xi)[i] + (Matrix(Lm)*omega)[i] >= 0.0
                         omega[i] >= 0.0
            end)
            
        end
        
        JuMP.optimize!(horizfModel)
        terminationStatus_horizf = JuMP.termination_status(horizfModel)

    elseif approach == BY_LCP

        optimizer = PATHSolver.Optimizer
        solverAttributes = ("output" => "no",)
        horizfModel = JuMP.Model(JuMP.optimizer_with_attributes(optimizer, solverAttributes...))
        
        ZJmInv = Zm/Jm
        gamma = -ZJmInv*bm
        S = Lm - ZJmInv*Ym

        JuMP.@variable(horizfModel, omega[1:size(Zm)[1]] >= 0.0)

        JuMP.@constraint(horizfModel, complements(gamma + S*omega, omega))

        JuMP.optimize!(horizfModel)
        terminationStatus_horizf = JuMP.termination_status(horizfModel)
        
    else
        throw(DomainError(:approach, "unsupported equation-solving approach"))
    end
    
    if string(terminationStatus_horizf) in ["OPTIMAL", "LOCALLY_SOLVED"]
        global_minimum = "PA function f does not have a global minimum"
        
    else
        global_minimum = "PA function f has a global minimum"
    end

    return terminationStatus_horizf, global_minimum
end


function minimize_pa_equation(
    a::AnfCoeffs;
    approach::SolutionApproach2,
    optimizer = BARON.Optimizer,
    solverAttributes = ("PrLevel" => 0,)
)
    
    aMod = modify_minf_coeffs(a::AnfCoeffs)
    (cm, bm, Zm, Lm, Jm, Ym) = recover_minf_coeffs(aMod)
    
    minfModel = JuMP.Model(JuMP.optimizer_with_attributes(optimizer, solverAttributes...))
    
    if approach == BY_LPCC
        
        JuMP.@variable(minfModel, x[1:size(Zm)[2]])
        JuMP.@variable(minfModel, w[1:length(cm)] >= 0.0)

        JuMP.@objective(minfModel, Min, (bm + Jm * x + Ym * w)[1])

        for i in eachindex(cm)
            JuMP.@constraints(minfModel, begin
                        (cm[i] + (Zm*x)[i] + (Matrix(Lm)*w)[i]) * w[i] == 0.0
                         cm[i] + (Zm*x)[i] + (Matrix(Lm)*w)[i] >= 0.0
                         w[i] >= 0.0
            end)
            
        end
    
        JuMP.optimize!(minfModel)
        
        terminationStatus_minif = JuMP.termination_status(minfModel) 
        
        wStar = JuMP.value.(w)
        xStar = JuMP.value.(x)
        optimal = bm + Jm * xStar + Ym * wStar
        
    
    elseif approach == BY_MILP
        
        mu = 1e5
        
        JuMP.@variable(minfModel, x[1:size(Zm)[2]])
        JuMP.@variable(minfModel, 0.0 <= w[1:length(cm)] <= mu)
        JuMP.@variable(minfModel, y[1:length(cm)], Bin)
        
        JuMP.@objective(minfModel, Min, (bm + Jm * x + Ym * w)[1])
        
        JuMP.@constraint(minfModel,w <= mu * y)

        e = ones(length(cm))

        JuMP.@constraint(minfModel, zeros(length(cm)) <= cm + Zm*x + Matrix(Lm)*w)
        JuMP.@constraint(minfModel, cm + Zm*x + Matrix(Lm)*w <= mu*(e-y))
        
        
        JuMP.optimize!(minfModel)
        
        terminationStatus_minif = JuMP.termination_status(minfModel)
        
        wStar = JuMP.value.(w)
        xStar = JuMP.value.(x)
        optimal = bm + Jm * xStar + Ym * wStar


    else
        throw(DomainError(:approach, "unsupported equation-solving approach"))
    end
    
    
    if string(terminationStatus_minif) in ["OPTIMAL", "LOCALLY_SOLVED"]
        #global_minimum = "PA function f has a global minimum"
        xStar = JuMP.value.(x)
    else
        #global_minimum = "PA function f does not have a global minimum"
        xStar = "Infeasible"
    end

    return xStar, optimal, terminationStatus_minif
    
end


@enum TypeofProblem begin
    root_finding
    optimal_finding
end

# For use in generating matrix Y and J
function generate_nonzero_matrix(m, n)
    A = round.(randn(m, n))
    for i in 1:m, j in 1:n
        while A[i, j] == 0
            A[i, j] = round(randn(), digits=0)
        end
    end
    return A
end


# Define the piecewise affine function
function generate_pa(problem::TypeofProblem, n)
    # Generate random values for m, n, and s
    if problem == root_finding
        min_dim = 1
        if n == 1
            max_dim = 1
        else
            max_dim = n - 1
        end
        # n = rand(min_dim:max_dim) 
        m = n
        s = rand(min_dim:max_dim) 
    elseif problem == optimal_finding
        min_dim = 1
        if n == 1
            max_dim = 1
        else
            max_dim = n - 1
        end
        # if this number is too large, the generated examples do not have minimum most of time
        # if the max_dim = n-1, when increasing n (about 8), almost all generated examples do not have global minimum. 
        m = 1
        s = rand(min_dim:max_dim) 
    end
    
    c = round.(randn(s), digits=0)
    b = round.(randn(m), digits=0)
    L = LowerTriangular(tril(round.(randn(s, s), digits=0), -1))
    Z = generate_nonzero_matrix(s, n)
    J = generate_nonzero_matrix(m, n)
    Y = generate_nonzero_matrix(m, s)
    
    return c, b, Z, L, J, Y
end

function is_nonsingular(matrix::Matrix)
    return det(matrix) != 0
end

# consider the PA function:
#  f_1(x_1,x_2) = ||x_1+2|+x_2-1| - x_2 -1
#  f_2(x_1,x_2) = |x_1+2| + 2x_2-1
# solve the equation system:
#  f(x) = 0

function example_2()

    c = [2.0, -1.0]
    b = [-1.0, -1.0]
    Z = [1.0 0.0; 0.0 1.0]
    L = LowerTriangular([0.0 0.0; 1.0 0.0])
    J = [0.0 -1.0; 0.0 2.0]
    Y = [0.0 1.0; 1.0 0.0]
    
    anf = AnfCoeffs(c, b, Z, L, J, Y)
    

    Root1, terminationStatus1 = solve_pa_equation(anf,approach=BY_LCP)
    Root2, terminationStatus2 = solve_pa_equation(anf,approach=BY_MLCP)

    println("==============Root finding of f(x):==========================================")
    @show Root1
    @show terminationStatus1

    @show Root2
    @show terminationStatus2

end
example_2()

# consider the PA function:
#   f(x)= abs(x1+abs(2x2-1))+abs(3+x3)
# verify the existence of the global minimum of f(x)
# and minimizing f(x)
function example_3()

    c = [-1.0, 0.0, 3.0]
    b = [0.0]
    Z = [0.0 2.0 0.0; 1.0 0.0 0.0; 0.0 0.0 1.0]
    L = LowerTriangular([0.0 0.0 0.0; 1.0 0.0 0.0; 0.0 0.0 0.0])
    J = [0.0 0.0 0.0]
    Y = [0.0 1.0 1.0]
    
    anf = AnfCoeffs(c, b, Z, L, J, Y)
    
    terminationStatus_horizf1, global_minimum1 = verify_minimum_of_f(anf,approach=BY_LCP)

    terminationStatus_horizf2, global_minimum2 = verify_minimum_of_f(anf,approach=BY_MLCP)

    
    if global_minimum1 == global_minimum2 && global_minimum1 == "PA function f has a global minimum"
        println("==============Finding global minimum of f(x):==========================================")
        # println("================================================")
        xStar, terminationStatus_minif = minimize_pa_equation(anf,approach=BY_LPCC)
        @show global_minimum1
        @show xStar
    end

end

example_3()

# Define an array of values for n
values_of_n = [1,5,10,50,100,200,300,400,500]  # Add more values as needed


# Initialize a dictionary to store CPU times for each value of n
cpu_times_MLCP = Dict{Int, Float64}()
cpu_times_LCP = Dict{Int, Float64}()

function example(n, cpu_time_MLCP, cpu_time_LCP, count_iter)
    
    c, b, Z, L, J, Y = generate_pa(root_finding, n)
    anf = AnfCoeffs(c, b, Z, L, J, Y)
    aMod = modify_anf_coeffs(anf)

    if is_nonsingular(aMod.J)
        cpu_time_MLCP += @elapsed begin
            Root, terminationStatus = solve_pa_equation(anf, approach=BY_MLCP)
            #@show Root
        end
        
        cpu_time_LCP += @elapsed begin
            Root, terminationStatus = solve_pa_equation(anf, approach=BY_LCP)
           #@show Root
        end
        count_iter += 1
    end
    
    return cpu_time_MLCP, cpu_time_LCP, count_iter
end
    

# Loop over each value of n
for n in values_of_n
    
    println("Running loop for n = $n")
    cpu_time_MLCP = 0.0
    cpu_time_LCP = 0.0
    count_iter = 0

    # Measure CPU time
    while count_iter < 100
        # Update cpu_time_MLCP, cpu_time_LCP, and count_iter by calling the example function with them as arguments
        cpu_time_MLCP, cpu_time_LCP, count_iter = example(n, cpu_time_MLCP, cpu_time_LCP, count_iter)
        #print(cpu_time_LPCC)
    end

    # Store the total CPU time for running the loop for the current value of n in the dictionary
    cpu_times_MLCP[n] = cpu_time_MLCP
    cpu_times_LCP[n] = cpu_time_LCP
    
    # Print total CPU time for running the loop for the current value of n
    println("Average CPU time for solving f(x)=0 using MLCP for n = $n: ", cpu_time_MLCP/count_iter)
    println("Average CPU time for solving f(x)=0 using LCP for n = $n: ", cpu_time_LCP/count_iter)
    
end

# Define the piecewise affine function
function generate_pa_n(n)
    # Define the range for random values
    m = 1
    s = n

    c = ones(n) * 0.0
    b = ones(m) 
    L = LowerTriangular(zeros(s, s)) # Initialize a matrix of zeros with size n x n
    for i in 2:s
        L[i, i-1] = 1.0 # Set the elements just above the diagonal to 1
    end
    #L = LowerTriangular(ones(s, s) - I)
    Z = Diagonal(ones(n) * 1000)
    J = zeros(m, n)
    Y = zeros(m, s)
    Y[end] = 1.0
    
    return c, b, Z, L, J, Y
end

# Define an array of values for n
values_of_n = [1,5,10,50,100,200,300,400,500]  # Add more values as needed

# Initialize a dictionary to store CPU times for each value of n
cpu_times_MILP = Dict{Int, Float64}()
cpu_times_LPCC = Dict{Int, Float64}()

function example(n, cpu_time_MILP, cpu_time_LPCC, count_iter)
    
    c, b, Z, L, J, Y = generate_pa_n(n)
    anf = AnfCoeffs(c, b, Z, L, J, Y)
    terminationStatus_horizf, global_minimum = verify_minimum_of_f(anf, approach=BY_MLCP)
    #println("================================checking============================================")
    if global_minimum == "PA function f has a global minimum"
        #println("================================Example============================================")
        cpu_time_MILP += @elapsed begin
            xStar, optimal, terminationStatus_mini = minimize_pa_equation(anf,approach=BY_MILP)
            # @show global_minimum
            # @show xStar
            # @show optimal
        end

        cpu_time_LPCC += @elapsed begin
            xStar, optimal, terminationStatus_mini = minimize_pa_equation(anf,approach=BY_LPCC)
            # @show global_minimum
            # @show xStar
            # @show optimal
        end
        count_iter += 1
    end
    
    return cpu_time_MILP, cpu_time_LPCC, count_iter
end


# Loop over each value of n
for n in values_of_n
    
    println("Running loop for n = $n")
    cpu_time_MILP = 0.0
    cpu_time_LPCC = 0.0
    count_iter = 0

    # Measure CPU time
    while count_iter < 10
        # Update cpu_time_MLCP, cpu_time_LCP, and count_iter by calling the example function with them as arguments
        cpu_time_MILP, cpu_time_LPCC, count_iter = example(n, cpu_time_MILP, cpu_time_LPCC, count_iter)
        #print(cpu_time_LPCC)
    end

    # Store the total CPU time for running the loop for the current value of n in the dictionary
    cpu_times_MILP[n] = cpu_time_MILP
    cpu_times_LPCC[n] = cpu_time_LPCC
    
    # Print total CPU time for running the loop for the current value of n
    println("Average CPU time for minimizing f(x) using MILP for n = $n: ", cpu_time_MILP/count_iter)
    println("Average CPU time for minimizing f(x) using LPCC for n = $n: ", cpu_time_LPCC/count_iter)
end
