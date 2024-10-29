include("../src/AbsNormalWithoutBARON.jl")

using .AbsNormal, LinearAlgebra, TimerOutputs, JuMP

@enum TypeofProblem begin
    root_finding
    optimal_finding
end

# For use in generating matrix Y in root finding problem 
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
function generate_pa(problem::TypeofProblem, n; compare_with_Griewank::Bool=false)
    if problem == root_finding
        m = n
        s = n

        c = round.(randn(s), digits=0)
        L = LowerTriangular(zeros(s, s)) # Initialize a matrix of zeros with size s x s
        for i in 2:s
            L[i, i-1] = 1.0 # Set the elements just below the diagonal to 1
        end
        
        Z = Diagonal(ones(n))
        
        # Conditional setting of J based on compare_with_Griewank flag
        if compare_with_Griewank
            J = Diagonal(ones(n)) # Set J as a diagonal matrix with ones
        else
            J = zeros(n, n) # Set J as a zero matrix
        end
        
        Y = generate_nonzero_matrix(m, s)
        b = zeros(m) 
        
        x = round.(randn(n) * 10, digits=0)
        # Adjusting the randomly generated function by modifying b
        # This could ensure the generated function has a root
        z = zeros(s)
        for i in 1:s
            z[i] = c[i] + dot(Z[i, :], x) + dot(L[i, :], abs.(z))
        end
        f_x = b .+ J * x .+ Y * abs.(z)
        b = -f_x
        
        return c, b, Z, L, J, Y, x, f_x
    
    elseif problem == optimal_finding
        m = 1
        s = n

        c = zeros(n) 
        b = ones(m) 
        L = LowerTriangular(zeros(s, s)) # Initialize a matrix of zeros with size s x s
        for i in 2:s
            L[i, i-1] = 1.0 # Set the elements just below the diagonal to 1
        end
        
        Z = Diagonal(ones(n) * 1000)
        J = zeros(m, n)
        Y = zeros(m, s)
        Y[end] = 1.0
        
        return c, b, Z, L, J, Y
    else
        error("Unsupported problem type")
    end 
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
    
    terminationStatus_horizf, existence = verify_minimum_existence(anf,approach=BY_MLCP)

    xStar, terminationStatus_minif = minimize_pa_function(anf,approach=BY_MILP)
        @show existence
        @show xStar
end

example_3()


# Matrix dimensions to consider:
values_of_n = [2,5,10,20,50,100,200,300,400,500]  # Add more values as needed

# Solve 10 somewhat arbitrary root-finding instance for each considered dimension
max_iters = 10

function exampleRoot()
    # Create an array to store TimerOutputs for each nest
    timer_outputs = []

    for (i, n) in enumerate(values_of_n)
        # Create a new TimerOutput for each "nest" and store it in the array
        to = TimerOutput()
        push!(timer_outputs, to)

        @timeit to "nest(n=$n)" begin
            
            count_iter = 0
        
            while count_iter < max_iters 
                @timeit to "generate_PA_function" begin
                    c, b, Z, L, J, Y = generate_pa(root_finding, n)
                    anf = AnfCoeffs(c, b, Z, L, J, Y)
                    aMod = modify_anf_coeffs(anf)
                end
                
                is_nonsingular(m::Matrix) = (cond(m) < 1e6)
                if is_nonsingular(aMod.J)
                    count_iter += 1
                    # Measure CPU time

                    xStar_MLCP, terminationStatus_MLCP = solve_pa_equation(anf, approach=BY_MLCP)
                    if terminationStatus_MLCP in [JuMP.OPTIMAL, JuMP.LOCALLY_SOLVED]
                        @timeit to "solve_by_MLCP_PATH" solve_pa_equation(anf, approach=BY_MLCP)
                    end

                    xStar_LCP, terminationStatus_LCP = solve_pa_equation(anf, approach=BY_LCP)
                    if terminationStatus_LCP in [JuMP.OPTIMAL, JuMP.LOCALLY_SOLVED]
                        @timeit to "solve_by_LCP_PATH" solve_pa_equation(anf, approach=BY_LCP)
                    end
                end
            end
        end
        show(to, title = "Results for n = $n", allocations = false, linechars=:ascii, sortby=:firstexec)
        println("\n")
    end
    # Merge all TimerOutputs into a single TimerOutput
    merged_to = TimerOutput()
    for to in timer_outputs
        merge!(merged_to, to)
    end
    # Display the merged timing results
    show(merged_to, title="Results for all n", allocations = false, linechars=:ascii, sortby=:firstexec)
end

#exampleRoot()


# Matrix dimensions to consider:
values_of_n = [2,5,10,20,50,100]  # Add more values as needed

# Solve 10 somewhat arbitrary root-finding instance for each considered dimension
max_iters = 10

function compareWithGriewank()
    # Create an array to store TimerOutputs for each nest
    timer_outputs = []

    for (i, n) in enumerate(values_of_n)
        # Create a new TimerOutput for each "nest" and store it in the array
        to = TimerOutput()
        push!(timer_outputs, to)

        @timeit to "nest(n=$n)" begin
            
            count_iter = 0
            
            while count_iter < max_iters 
                @timeit to "generate_PA_function" begin
                    c, b, Z, L, J, Y = generate_pa(root_finding, n, compare_with_Griewank=true)
                    anf = AnfCoeffs(c, b, Z, L, J, Y)
                    aMod = modify_anf_coeffs(anf)
                end
                
                is_nonsingular(m::Matrix) = (cond(m) < 1e6)
                if is_nonsingular(aMod.J)
                    count_iter += 1
    
                    # Measure CPU time
                    rootLCP_Griewank, terminationStatusLCP_Griewank = solve_pa_equation_Griewank(anf, approach=BY_GriewankLCP)
                    rootMLCP, terminationStatusMLCP = solve_pa_equation(anf, approach=BY_MLCP)
                    rootLCP, terminationStatusLCP = solve_pa_equation(anf, approach=BY_LCP)

                    if terminationStatusLCP_Griewank in [JuMP.OPTIMAL, JuMP.LOCALLY_SOLVED]
                        @timeit to "Griewank_solve_by_LCP_PATH" solve_pa_equation_Griewank(anf, approach=BY_GriewankLCP)
                    end

                    if terminationStatusMLCP in [JuMP.OPTIMAL, JuMP.LOCALLY_SOLVED]
                        @timeit to "solve_by_MLCP_PATH" solve_pa_equation(anf, approach=BY_MLCP)
                    end

                    if terminationStatusLCP in [JuMP.OPTIMAL, JuMP.LOCALLY_SOLVED]
                        @timeit to "solve_by_LCP_PATH" solve_pa_equation(anf, approach=BY_LCP)
                    end 
                        
                end
            end
        end
        show(to, title = "Results for n = $n", allocations = false, linechars=:ascii, sortby=:firstexec)
        println("\n")
    end
    # Merge all TimerOutputs into a single TimerOutput
    merged_to = TimerOutput()
    for to in timer_outputs
        merge!(merged_to, to)
    end
    # Display the merged timing results
    show(merged_to, title="Results for all n", allocations = false, linechars=:ascii, sortby=:firstexec)
end

#compareWithGriewank()

# Define an array of considered matrix dimensions
values_of_n = [2,5,10,20,50,100,200,300,400,500]  # Add more values as needed

# minimize 10 somewhat arbitrary PA functions for each considered matrix dimension
max_iters = 10

function exampleOptim()
    # Create an array to store TimerOutputs for each nest
    timer_outputs = []

    for (i, n) in enumerate(values_of_n)
        # Create a new TimerOutput for each "nest" and store it in the array
        to = TimerOutput()
        push!(timer_outputs, to)

        @timeit to "nest(n=$n)" begin
            
            count_iter = 0
        
            while count_iter < max_iters 
                @timeit to "generate_PA_function" begin
                    c, b, Z, L, J, Y = generate_pa(optimal_finding, n)
                    anf = AnfCoeffs(c, b, Z, L, J, Y)
                end
                
                @timeit to "verify_minimum_existence" terminationStatus_horizf, globalMinExists = verify_minimum_existence(anf, approach=BY_MLCP)
                
                if globalMinExists == true
                    @timeit to "minimize_by_MILP" minimize_pa_function(anf,approach=BY_MILP)
                    #@timeit to "minimize_by_LPCC" minimize_pa_function(anf,approach=BY_LPCC)
                    count_iter += 1
                end

            end
        end
        show(to, title = "Results for n = $n", allocations = false, linechars=:ascii, sortby=:firstexec)
        println("\n")
    end
    # Merge all TimerOutputs into a single TimerOutput
    merged_to = TimerOutput()
    for to in timer_outputs
        merge!(merged_to, to)
    end
    # Display the merged timing results
    show(merged_to, title="Results for all n", allocations = false, linechars=:ascii, sortby=:firstexec)
end

#exampleOptim()

