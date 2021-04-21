# @Created by: octave
# @        on: 2021-04-20T16:00:50+02:00
# @Last modified by: octave
# @              on: 2021-04-20T17:44:35+02:00

using DifferentialEquations
using LinearAlgebra
using Plots ; plotly()

e = 0.1

w1 = 1.8
w2 = 0.4
q11 = 1.0 - e
q12 = 1.0
q21 = 0.0 + e
q22 = 0.0

function mces!(dq, q, p, t)
    dq[1] = w1*(q11+(q[1]<=q[2])*(q21-q11) - q[1])
    dq[2] = w2*(q12+(q[1]<=q[2])*(q22-q12) - q[2])
end

q_min = -1.0
q_prec = 0.5
q_max = 2.0

grid_side = q_min:q_prec:q_max
q1_grid = repeat(grid_side, 1, length(grid_side))
q2_grid = repeat(grid_side', length(grid_side), 1)

p = plot()
plot!(q_min:q_max, q_min:q_max, color=:black, linewidth=2)
tspan = (0.0,10.0)

for q0 = eachrow([q1_grid[:] q2_grid[:]])
    prob = ODEProblem(mces!, q0, tspan)
    sol = solve(prob, Tsit5(), reltol=1e-8, abstol=1e-8)
    plot!(sol,vars=(1,2), linewidth=1, xaxis="q1", yaxis="q2", color=:blue, legend=false)
end

scatter!((q11, q12), markershape=:cross, color=:green)
scatter!((q21, q22), markershape=:cross, color=:red)
q_ = (-q12*q21 + q11*q22)/(q11-q12-q21+q22)
scatter!((q_, q_), markershape=:circle, color=:blue)

display(p)
