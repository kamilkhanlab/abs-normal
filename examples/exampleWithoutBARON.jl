include("../src/AbsNormalWithoutBARON.jl")

using .AbsNormal, LinearAlgebra

@enum TypeofProblem begin
    root_finding
    optimal_finding
end

# For use in generating matrices Y and J
function generate_nonzero_matrix(m, n)
    A = round.(randn(m, n))
    for i in 1:m, j in 1:n
        while A[i, j] == 0
            A[i, j] = round(randn(), digits=0)
        end
    end
    return A
end


# Generate a somewhat arbitrary piecewise affine function
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
    

    rootLCP, terminationStatusLCP = solve_pa_equation(anf,approach=BY_LCP)
    rootMLCP, terminationStatusMLCP = solve_pa_equation(anf,approach=BY_MLCP)

    println("## Solving f(x) = 0:")
    @show rootLCP
    @show terminationStatusLCP

    @show rootMLCP
    @show terminationStatusMLCP

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
    
    terminationStatus_horizf1, existence = verify_minimum_existence(anf,approach=BY_LCP)

    xStar, terminationStatus_minif = minimize_pa_function(anf,approach=BY_MILP)
        @show existence
        @show xStar
    #=
    terminationStatus_horizf2, global_minimum2 = verify_minimum_existence(anf,approach=BY_MLCP)

    
    if global_minimum1 == global_minimum2 && global_minimum1 == "PA function f has a global minimum"
        println("==============Finding global minimum of f(x):==========================================")
        # println("================================================")
        xStar, terminationStatus_minif = minimize_pa_function(anf,approach=BY_MILP)
        @show global_minimum1
        @show xStar
    end
    =#

end

example_3()

# Matrix dimensions to consider:
values_of_n = [1,5,10,50,100,200,300,400,500]  # Add more values as needed


# Initialize a dictionary to store CPU times for each value of n
cpu_times_MLCP = Dict{Int, Float64}()
cpu_times_LCP = Dict{Int, Float64}()

function exampleRoot(n, cpu_time_MLCP, cpu_time_LCP, count_iter)
    
    c, b, Z, L, J, Y = generate_pa(root_finding, n)
    anf = AnfCoeffs(c, b, Z, L, J, Y)
    aMod = modify_anf_coeffs(anf)

    is_nonsingular(m::Matrix) = (cond(m) < 1e6)

    if is_nonsingular(aMod.J)
        cpu_time_MLCP += @elapsed begin
            solve_pa_equation(anf, approach=BY_MLCP)
        end
        
        cpu_time_LCP += @elapsed begin
           solve_pa_equation(anf, approach=BY_LCP)
        end
        
        count_iter += 1
    end
    
    return cpu_time_MLCP, cpu_time_LCP, count_iter
end
    

# Solve 5 somewhat arbitrary root-finding instances for each considered dimension:
max_iters = 5

for n in values_of_n
    
    println("Running loop for n = $n")
    cpu_time_MLCP = 0.0
    cpu_time_LCP = 0.0
    count_iter = 0

    # Measure CPU time
    while count_iter < max_iters
        # Update cpu_time_MLCP, cpu_time_LCP, and count_iter by calling the example function with them as arguments
        cpu_time_MLCP, cpu_time_LCP, count_iter = exampleRoot(n, cpu_time_MLCP, cpu_time_LCP, count_iter)
        #print(cpu_time_LPCC)
    end

    # Store the total CPU time for running the loop for the current value of n in the dictionary
    cpu_times_MLCP[n] = cpu_time_MLCP
    cpu_times_LCP[n] = cpu_time_LCP
    
    # Print total CPU time for running the loop for the current value of n
    println("Average CPU time for solving f(x)=0 using MLCP for n = $n: ", cpu_time_MLCP/count_iter)
    println("Average CPU time for solving f(x)=0 using LCP for n = $n: ", cpu_time_LCP/count_iter)
    
end

# Generate a somewhat arbitrary piecewise affine function
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

# Define an array of considered matrix dimensions
values_of_n = [1,5,10,50,100,200,300,400,500]  # Add more values as needed

# Initialize a dictionary to store CPU times for each value of n
cpu_times_MILP = Dict{Int, Float64}()
cpu_times_LPCC = Dict{Int, Float64}()

function exampleOptim(n, cpu_time_MILP, cpu_time_LPCC, count_iter)
    
    c, b, Z, L, J, Y = generate_pa_n(n)
    anf = AnfCoeffs(c, b, Z, L, J, Y)
    terminationStatus_horizf, globalMinExists = verify_minimum_existence(anf, approach=BY_LCP)
    #println("================================checking============================================")
    if globalMinExists == true
        #println("================================Example============================================")
        cpu_time_MILP += @elapsed begin
            xStar, fStar, terminationStatus = minimize_pa_function(anf,approach=BY_MILP)
            # @show globalMinExists
            # @show xStar
            # @show fStar
        end

        #=
        cpu_time_LPCC += @elapsed begin
            xStar, fStar, terminationStatus = minimize_pa_function(anf,approach=BY_LPCC)
            # @show globalMinExists
            # @show xStar
            # @show fStar
        end
        =#
        count_iter += 1
    end
    
    return cpu_time_MILP, cpu_time_LPCC, count_iter
end


# minimize 5 somewhat arbitrary PA functions for each considered matrix dimension
max_iters = 5

for n in values_of_n
    
    println("Running loop for n = $n")
    cpu_time_MILP = 0.0
    cpu_time_LPCC = 0.0
    count_iter = 0

    # Measure CPU time
    while count_iter < 5
        # Update cpu_time_MLCP, cpu_time_LCP, and count_iter by calling the example function with them as arguments
        cpu_time_MILP, cpu_time_LPCC, count_iter = exampleOptim(n, cpu_time_MILP, cpu_time_LPCC, count_iter)
        #print(cpu_time_LPCC)
    end

    # Store the total CPU time for running the loop for the current value of n in the dictionary
    cpu_times_MILP[n] = cpu_time_MILP
    cpu_times_LPCC[n] = cpu_time_LPCC
    
    # Print total CPU time for running the loop for the current value of n
    println("Average CPU time for minimizing f(x) using MILP for n = $n: ", cpu_time_MILP/count_iter)
    #=
    println("Average CPU time for minimizing f(x) using LPCC for n = $n: ", cpu_time_LPCC/count_iter)
    =#
end
