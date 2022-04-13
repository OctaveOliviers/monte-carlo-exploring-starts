# @Created by: octave
# @        on: 2022-04-13T08:01:08+01:00
# @Last modified by: octave
# @              on: 2022-04-13T09:20:59+01:00



include("../libraries.jl")
include("../mdp.jl")
include("../mces.jl")
include("data_problem.jl")

# mdp = MDP(num_s, num_sa, discount, structure, policy, transitions, rewards, q, term_states)

Random.seed!(317)
mdp = create_rand_mdp(num_s=1, num_sa=2, num_term=0, discount=0.9)

# create grid over q and r space
qmin = pol2val(hcat([1,0]), mdp)
qmax = pol2val(hcat([0,1]), mdp)
X, Y = range(qmin[1], qmax[1], length=10), range(qmin[2], qmax[2], length=10)
# step = 0.5
# X, Y = qmin[1]:step:qmax[1], qmin[2]:step:qmax[2]
q_grid = vcat((ones(length(Y))*X')[:]', (Y*ones(length(X))')[:]')
r_grid = aval2rval(mdp, q_grid)

# plot grid
plot(
    # q_grid[1,:], q_grid[2,:],
    r_grid[1,:], r_grid[2,:],
    seriestype=:scatter,
    xaxis="value 1",
    yaxis="value 2",
    # xlims=(7.0, 8.0),
    # ylims=(1.75, 2.25),
    tickfont=font(14,"Computer Modern"),
    guidefont=font(18,"Computer Modern"),
    legendfont=font(18,"Computer Modern"),
    aspect_ratio=:equal
    )
