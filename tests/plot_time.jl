# @Created by: octave
# @        on: 2021-08-10T07:20:25+02:00
# @Last modified by: octave
# @              on: 2022-04-12T15:35:16+01:00



include("../libraries.jl")
include("../mdp.jl")
include("../mces.jl")
include("data_problem.jl")

mdp = MDP(num_s, num_sa, discount, structure, policy, transitions, rewards, q, term_states)

sol = simulate_continuous_mces(mdp, q0, prior)

q_sol = reduce(hcat, sol.u)
v_sol = aval2sval(mdp, q_sol)
r_sol = aval2rval(mdp, q_sol)

# check whether error on rewards monotonically decreases
r_norm = norm.(eachcol(rewards .- r_sol))

# plot
# gr(size=(500,380))
plot(
    sol.t,
    # v_sol,
    # q_sol',
    # r_sol',
    r_norm,
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
