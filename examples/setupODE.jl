###
#TASK: Try implementing implicit ODE solver by hand 
# Time will end up being a constant in every iteration of the solve
# Use NLSolve
# Read ODE book (Ascher and Petzold) on ODE and DAE solvers - first chapters (ask Sahand if needed)

using NLsolve, LinearAlgebra, PlotlyJS

#ODE FUNCTION:
function f_ode(y, t)
    # dy1 = -(y[1] + 0.25) + (y[1] + 0.5)*exp((25*y[1])/(y[1] + 2))
    # dy2 = 0.50 - y[2] - (y[2] + 0.5)*exp((25*y[1])/(y[1] + 2))
    # dy3 = y[1]^2 + y[2]^2 + 0.1 

    # dy1 = (y[1] + 0.25) - sin(y[3])
    # dy2 = 0.5 - y[2] * cos(y[1])
    # dy3 = y[1]^2 + sin(y[3])

    k1 = 5.7; k2 = 6.1;
    dy1 = -k1 * y[1] * t
    dy2 =  k2 * y[1] * t - k2 * y[2] * t
    dy3 =  k2 * y[2] * t 
    dy4 =  k1 * y[1] * t + k2 * y[2] * t

    return [dy1, dy2, dy3, dy4]
end

function implicit_step(F, y_n1, y_n0, t_n1, dt)
    #Use f_ode to calculate 'dy' at t_n1
    dy_n1 = f_ode(y_n1, t_n1)
    for i in eachindex(F)  
        F[i] = y_n1[i] - y_n0[i] - dt * dy_n1[i]
    end #for 
end #function 

function solve_implicit_ode(y0, tspan, dt)
    #Set up time vector:
    t_val = collect(tspan[1]:dt:tspan[2])
    num_steps = length(t_val)

    #Set up y-vector with initial guess:
    y_val = zeros(length(y0), num_steps)
    y_val[:, 1] = y0

    for n in 1:(num_steps - 1)
        #Define function at time step 't':
        solve_step(F, y_n1) = implicit_step(F, y_n1, y_val[:, n], t_val[n+1], dt)
        #Use non-linear solver to locate value:
        sol = nlsolve(solve_step, y_val[:, n])
        y_val[:, n+1] = sol.zero
    end #for 

    return t_val, y_val
end #function 

# y0 = [0.09, 0.09, 0.0]
# y0 = [10, 5, 1]
# y0 = [1, 2]
y0 = [4.0, 1.0, 0.5, 0.0]
tspan = [0.0, 1.0]
dt = 0.001
times, states = solve_implicit_ode(y0, tspan, dt)
println("Final states at t = $(times[end]): ", states[:, end])



