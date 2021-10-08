# @Created by: octave
# @        on: 2021-05-02T14:47:21+01:00
# @Last modified by: octave
# @              on: 2021-05-03T15:41:35+01:00

using DifferentialEquations
using LinearAlgebra
using Plots ; plotly() # plotly() gr() pyplot()
using Random

Random.seed!(13)

w1 = 0.7
w2 = 0.3
W = Diagonal([w1, w2])

e = 0.1
q1 = [1.0+e, 1.0]
q2 = [0.0+5.0*e, 0.0]
q_pols = [q1 q2]

function mces_bw!(dq, q, p, t)
    dq .= -1e-2*W*(q_pols*(argmax(q).==1:length(q)) - q)
end

stp = 1e-3
num = 1e2
circle = q1' .+ stp.*hcat(cos.(2π.*(0:1/num:1)), sin.(2π.*(0:1/num:1)))

# dt = 1e-3
num_steps = 1e2
t_span = (0, 1e-3)

p = plot()
for i in 1:num_steps

    for (n, q0) in enumerate(eachrow(circle))
        prob = ODEProblem(mces_bw!, q0, tspan)
        sol = solve(prob, Tsit5(), reltol=1e-5, abstol=1e-5)
        circle[n,:] = sol[:,end]
    end

    plot!(circle[:,1], circle[:,2], legend=false, aspect_ratio=:equal)
end

scatter!(Tuple(q1), markershape=:cross, color=:green)
scatter!(Tuple(q2), markershape=:cross, color=:red)

display(p)
