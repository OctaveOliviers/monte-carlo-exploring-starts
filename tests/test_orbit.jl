# @Created by: octave
# @        on: 2021-04-20T16:00:50+02:00
# @Last modified by: octave
# @              on: 2021-08-10T07:49:37+02:00

using DifferentialEquations
using LinearAlgebra
using Plots ; plotly() # plotly() gr() pyplot()
using Random

Random.seed!(42)

num_sim = 10

# w1 = 0.4
# w2 = 0.2
# w3 = 0.8
w1 = 0.2
w2 = 0.8
w3 = 0.6

if false
    # sim_2d = true
    q1 = [1.0, 0.5]
    q2 = [0.2, 0.0]
else
    # sim_3d = true
    # q1 = [1.0, 0.5, 0.9]
    # q2 = [0.5, 0.3, 0.1]
    # q3 = [-0.5, 0.6, 0.4]
    q1 = [1.1, 0.6, 0.9]
    q2 = [1.0, 0.7, 0.9]
    q3 = [0.3, 0.2, 0.0]
end

function mces!(dq, q, p, t)
    dq .= W*(q_pols*(argmax(q).==1:length(q)) - q)
end

plot_min, plot_max = -1, 1.5

q_min, q_prec, q_max = 0.0, 0.1, 1.0
grid_side = q_min:q_prec:q_max
#
# q1_grid = repeat(grid_side, 1, length(grid_side))
# q2_grid = repeat(grid_side', length(grid_side), 1)

# q1_grid, q2_grid, q3_grid = rand(1,num_sim), rand(1,num_sim), rand(1,num_sim)
# q1_grid, q2_grid, q3_grid = 0.0*ones(length(grid_side)), grid_side, 0.0*ones(length(grid_side))
q1_grid, q2_grid, q3_grid = [0.3], [0.5], [0.4]

p = plot()
# plot!(q_min:q_max, q_min:q_max, color=:black, linewidth=2)
tspan = (0.0,1000.0)

# 2 dimensional simulation
if length(q1) == 2
    W = Diagonal([w1, w2])
    q_pols = [q1 q2]

    for q0 = eachrow([q1_grid[:] q2_grid[:]])
        prob = ODEProblem(mces!, q0, tspan)
        sol = solve(prob, Tsit5(), reltol=1e-4, abstol=1e-4)
        plot!(sol, vars=(1,2), linewidth=1,
              xaxis="q1", yaxis="q2",
              xlims=(q_min,q_max), ylims=(q_min,q_max),
              color=:blue, legend=false) # , aspect_ratio=:equal
    end

    plot!(grid_side, grid_side, color=:grey)

    scatter!(Tuple(q1), markershape=:cross, color=:green)
    scatter!(Tuple(q2), markershape=:cross, color=:red)
    q_ = (-q1[2]*q2[1] + q1[1]*q2[2])/(q1[1]-q1[2]-q2[1]+q2[2])
    scatter!((q_, q_), markershape=:circle, color=:blue)
end


# 3 dimensional simulation
if length(q1) == 3
    W = Diagonal([w1, w2, w3])
    q_pols = [q1 q2 q3]

    for q0 = eachrow([q1_grid[:] q2_grid[:] q3_grid[:]])
        prob = ODEProblem(mces!, q0, tspan)
        sol = solve(prob, Tsit5(), reltol=1e-8, abstol=1e-8, maxiters=1e7)
        plot!(sol, linewidth=1,
        # plot!(sol, vars=(1,2,3), linewidth=1,
              # xaxis="q1", yaxis="q2", zaxis="q3",
              # xlims=(plot_min, plot_max), ylims=(plot_min, plot_max), zlims=(plot_min, plot_max),
              legend=false) # , aspect_ratio=:equal, color=:blue,
    end

    # plot!(grid_side, grid_side, grid_side, st=:surface)
    # mesh3d(x, y, z)

    # scatter!(Tuple(q1), markershape=:cross, color=:green)
    # scatter!(Tuple(q2), markershape=:cross, color=:red)
    # scatter!(Tuple(q3), markershape=:cross, color=:blue)

    # # plot boundaries
    # # plot!()
    # x=range(-2,stop=2,length=100)
    # y=range(-1,stop=2,length=100)
    # # f(x,y) = x - y
    # plot!(x, y, x-y, st=:surface)
    # plot!(x-y, x, y, st=:surface)
    # plot!(y, x-y, x, st=:surface)
end

display(p)
