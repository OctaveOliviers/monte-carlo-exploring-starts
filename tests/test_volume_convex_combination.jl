# @Created by: octave
# @        on: 2021-05-19T12:16:05+01:00
# @Last modified by: octave
# @              on: 2021-05-25T17:46:01+01:00

using DifferentialEquations
using LinearAlgebra
using Plots ; plotly() # plotly() gr() pyplot()
using Random

Random.seed!(13)

# w1 = 0.7
# w2 = 0.3
# W = Diagonal([w1, w2])

t = -1.0/3.0
a = 0.5
W1 = tu leur
W2 = [t 0.0 0.0 ; 1.0 t 0.0 ; 2.0 1.0 t]
W = a*W1 + (1-a)*W2

function sys1!(dx, x, p, t)
    dx .= W1*x
end

function sys2!(dx, x, p, t)
    dx .= W2*x
end

function sys!(dx, x, p, t)
    dx .= W*x
end

stp = 1e1
num = 1e2
tspan = (0, 2*1e0)
# dt = 1e-3
# num_steps = 1e2

p = plot()

for s in [sys1!, sys2!, sys!]

    circle = stp.*hcat(cos.(2π.*(0:1/num:1)), sin.(2π.*(0:1/num:1)))
    for (n, q0) in enumerate(eachrow(circle))
        prob = ODEProblem(s, q0, tspan)
        sol = solve(prob, Tsit5(), reltol=1e-8, abstol=1e-8)
        circle[n,:] = sol[:,end]
    end

    plot!(circle[:,1], circle[:,2], legend=false, aspect_ratio=:equal)
end

display(p)
