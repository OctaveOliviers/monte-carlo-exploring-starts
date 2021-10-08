# @Created by: octave
# @        on: 2021-08-10T07:20:25+02:00
# @Last modified by: octave
# @              on: 2021-10-02T11:37:01+01:00

using DifferentialEquations
using LinearAlgebra
using Plots ; gr()
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

# for weird nonmonotone behaviour
num_s = 2
num_sa = 4
discount = 0.8
structure = [[1 0];
             [1 0];
             [0 1];
             [0 1]]
transitions = [[0 0 1 1];
               [1 1 0 0]]
rewards = [1., 0., 1., 0.]
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

q = compute_q_policy(policy, transitions, rewards, discount)
mdp = MDP(num_s, num_sa, discount, structure, policy, transitions, rewards, q, term_states)

# initialise
# q = 100*rand(Float32, num_sa)
# prior = Diagonal(rand(Float32, num_sa))
# prior = Diagonal([0.1, 0.4, 0.5])
prior = Diagonal([0.1, 0.4, 0.4, 0.1])
# prior = Diagonal([0.1, 0.4, 0.4, 0.1]) # for weird nonmonotone behaviour
# prior = Diagonal([0.5, 0.5, 0.9, 0.1, 1.])
# prior = Diagonal(0.5 .+ 0.5*rand(Float32, num_sa))
# sol =

function mces!(dq, q, p, t)
    pol = compute_policy(mdp.structure, q)
    q_pol = compute_q_policy(pol, mdp)
    dq .= prior*(q_pol - q)
end

# solve
tspan = (0.0,40.0)
# q0 = 20*rand(Float32, num_sa) .- 10
# q0 = [6.0, 5., 9.5]
q0 = [1.1, 0.1, 4.0, 3.9] # for weird nonmonotone behaviour
prob = ODEProblem(mces!, q0, tspan)
sol = solve(prob, Tsit5(), reltol=1e-5, abstol=1e-5, maxiters=1e7)
q_sol = reduce(hcat, sol.u)
v_sol = zeros(Real, (size(q_sol)[2], num_s))
for s = 1:num_s
    sa_idx = findall(x->x==1, mdp.structure[:,s])
    v_sol[:, s] = q_sol[sa_idx, :][argmax(q_sol[sa_idx, :], dims=1)]
end

# plot
gr(size=(500,380))
plot(
    sol.t,
    # v_sol,
    q_sol',
    # q_sol[findall(x->x==1, mdp.structure[:,1]), :]',
    # labels=[L"q(s_1, a_1)" L"q(s_1, a_2)" L"q(s_2, a_1)"],
    labels=[L"q(s_1, a_1)" L"q(s_1, a_2)" L"q(s_2, a_1)" L"q(s_2, a_2)"],
    legend=:bottomright,
    # legend=false,
    linewidth=1.5,
    xaxis="time",
    yaxis="value",
    # xlims=(0.0, 15.0),
    # ylims=(plot_min, plot_max),
    tickfont=font(14,"Computer Modern"),
    guidefont=font(18,"Computer Modern"),
    legendfont=font(18,"Computer Modern"),
    # aspect_ratio=:equal
    )
# display(p)
# savefig("not-monotone-inside-L-U-bounds-2.pdf")
