# @Created by: octave
# @        on: 2021-08-10T07:20:25+02:00
# @Last modified by: octave
# @              on: 2021-11-23T15:19:21+00:00

using DifferentialEquations
using LinearAlgebra
using Plots ; plotly()
using Random
using ProgressBars
using LaTeXStrings

include("../mdp/utils.jl")
include("../mces/utils.jl")

# Random.seed!(317)

# num_sa = 40
# num_s = 10
# mdp = generate_mdp(num_sa, num_s)

# num_s = 2
# num_sa = 3
# discount = 0.9
# structure = [[1 0];
#              [1 0];
#              [0 1]]
# transitions = [[0 0 1];
#                [1 1 0]]
# rewards = [1., 0.5, 1.]
# policy = [[1 0];
#           [0 0];
#           [0 1]]
# term_states = []

# for weird nonmonotone and spiral behaviour
# num_s = 2
# num_sa = 4
# discount = 0.8
# structure = [[1 0];
#              [1 0];
#              [0 1];
#              [0 1]]
# transitions = [[0 0 1 1];
#                [1 1 0 0]]
# rewards = [1., 0., 1., 0.]
# policy = [[1 0];
#           [0 0];
#           [0 1];
#           [0 0]]
# term_states = []

num_s = 2
num_sa = 4
discount = 0.8
structure = [[1 0];
             [1 0];
             [0 1];
             [0 1]]
transitions = [[0 0 1 1];
               [1 1 0 0]]
# rewards = [1., 0., 1., 0.]
rewards = [1., 0., 0.9, 0.]
policy = [[1 0];
          [0 0];
          [0 1];
          [0 0]]
term_states = []

# num_s = 3
# num_sa = 5
# discount = 0.9
# structure = [[1 0 0];
#              [1 0 0];
#              [0 1 0];
#              [0 1 0];
#              [0 0 1]]
# transitions = [[0 0 1 0 0];
#                [1 0 0 0 0];
#                [0 1 0 1 1]]
# rewards = [1., 0., 1., -10., 0.]
# policy = [[1 0 0];
#           [0 0 0];
#           [0 1 0];
#           [0 0 0];
#           [0 0 1]]
# term_states = [3]

# num_s = 2
# num_sa = 4
# discount = 0.1
# structure = [[1 0];
#              [1 0];
#              [0 1];
#              [0 1]]
# transitions = [[0 0 1 1];
#                [1 1 0 0]]
# rewards = [1., 0.9, 1., 0.9]
# policy = [[1 0];
#           [0 0];
#           [0 1];
#           [0 0]]
# term_states = []

q = compute_q_policy(policy, transitions, rewards, discount)
mdp = MDP(num_s, num_sa, discount, structure, policy, transitions, rewards, q, term_states)

# initialise
# q = 100*rand(Float32, num_sa)
# prior = Diagonal(rand(Float32, num_sa))
# prior = Diagonal([0.1, 0.4, 0.5])
# prior = Diagonal([0.01, 0.2, 0.2, 0.01])
prior = Diagonal([0.01, 0.4, 0.4, 0.01]) # for weird spiral behaviour
# prior = Diagonal([0.1, 0.4, 0.4, 0.1]) # for weird nonmonotone behaviour
# prior = Diagonal([0.01, 0.5, .5, 0.01])
# prior = Diagonal(0.5 .+ 0.5*rand(Float32, num_sa))
#
# q0 = 20*rand(Float32, num_sa) .- 10
# q0 = [6.0, 5., 9.5]
# q0 = [1.1, 0.1, 4.0, 3.9] # for weird nonmonotone behaviour
q0 = [1.9, 2, 2.95, 3] # for weird spiral behaviour
# q0 = [3.9, 3.9, 4.9, 3]

function mces!(dq, q, p, t)
    pol = compute_policy(mdp.structure, q)
    q_pol = compute_q_policy(pol, mdp)
    dq .= prior*(q_pol - q)
end

# solve
tspan = (0.0, 5.0)
prob = ODEProblem(mces!, q0, tspan)
# sol = solve(prob, Tsit5(), reltol=1e-6, abstol=1e-6, maxiters=1e7)
sol = solve(prob, Rosenbrock23(), reltol=1e-7, abstol=1e-7, maxiters=2*1e7)
# sol = solve(prob, Rodas4P(), reltol=1e-8, abstol=1e-8, maxiters=1e7)
q_sol = reduce(hcat, sol.u)
v_sol = zeros(Real, (size(q_sol)[2], num_s))
for s = 1:num_s
    sa_idx = findall(x->x==1, mdp.structure[:,s])
    v_sol[:, s] = q_sol[sa_idx, :][argmax(q_sol[sa_idx, :], dims=1)]
end

# plot
# gr(size=(500,380))
plot(
    sol.t,
    # v_sol,
    q_sol',
    # q_sol[findall(x->x==1, mdp.structure[:,1]), :]',
    # labels=[L"q(s_1, a_1)" L"q(s_1, a_2)" L"q(s_2, a_1)"],
    labels=[L"q(s_1, a_1)" L"q(s_1, a_2)" L"q(s_2, a_1)" L"q(s_2, a_2)"],
    # legend=:bottomright,
    legend=false,
    linewidth=1.5,
    xaxis="time",
    yaxis="value",
    # xlims=(7.0, 8.0),
    # ylims=(1.75, 2.25),
    tickfont=font(14,"Computer Modern"),
    guidefont=font(18,"Computer Modern"),
    legendfont=font(18,"Computer Modern"),
    # aspect_ratio=:equal
    )
# display(p)
# savefig("not-monotone-inside-L-U-bounds-spiral-closeup.pdf")



# idx = 60001
# q1 = sol.u[idx][1] # choose action-values that move slowly
# q2 = sol.u[idx][4] # choose action-values that move slowly
#
# pol1 = [[0 0]; [1 0]; [0 0]; [0 1]]
# q_pol1 = compute_q_policy(pol1, mdp)
# #
# pol2 = [[1 0]; [0 0]; [0 0]; [0 1]]
# q_pol2 = compute_q_policy(pol2, mdp)
# #
# pol3 = [[1 0]; [0 0]; [0 1]; [0 0]]
# q_pol3 = compute_q_policy(pol3, mdp)
# #
# pol4 = [[0 0]; [1 0]; [0 1]; [0 0]]
# q_pol4 = compute_q_policy(pol4, mdp)
#
#
# # policy, state
# dq_1_1 = prior[1,1] * (q_pol1[1] - q1) - prior[2,2] * (q_pol1[2] - q1)
# dq_1_2 = prior[3,3] * (q_pol1[3] - q2) - prior[4,4] * (q_pol1[4] - q2)
# #
# dq_2_1 = prior[1,1] * (q_pol2[1] - q1) - prior[2,2] * (q_pol2[2] - q1)
# dq_2_2 = prior[3,3] * (q_pol2[3] - q2) - prior[4,4] * (q_pol2[4] - q2)
# #
# dq_3_1 = prior[1,1] * (q_pol3[1] - q1) - prior[2,2] * (q_pol3[2] - q1)
# dq_3_2 = prior[3,3] * (q_pol3[3] - q2) - prior[4,4] * (q_pol3[4] - q2)
# #
# dq_4_1 = prior[1,1] * (q_pol4[1] - q1) - prior[2,2] * (q_pol4[2] - q1)
# dq_4_2 = prior[3,3] * (q_pol4[3] - q2) - prior[4,4] * (q_pol4[4] - q2)
#
# # check identity => should be equal to 1
# (dq_1_2*dq_2_1*dq_3_2*dq_4_1) / (dq_1_1*dq_2_2*dq_3_1*dq_4_2)
#
# # compute lambdas
# S = dq_1_1*dq_2_2*dq_3_1*dq_4_2 - dq_1_2*dq_1_1*dq_3_1*dq_4_2 +
#     dq_1_2*dq_1_1*dq_2_1*dq_4_2 - dq_1_2*dq_2_1*dq_3_2*dq_1_1
#
# lam1 =  dq_1_1*dq_2_2*dq_3_1*dq_4_2 / S
# lam2 = -dq_1_2*dq_1_1*dq_3_1*dq_4_2 / S
# lam3 =  dq_1_2*dq_1_1*dq_2_1*dq_4_2 / S
# lam4 = -dq_1_2*dq_2_1*dq_3_2*dq_1_1 / S
#
# prior*(lam1*q_pol1 + lam2*q_pol2 + lam3*q_pol3 + lam4*q_pol4 - [q1;q1;q2;q2])
