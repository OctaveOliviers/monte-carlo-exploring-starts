# @Created by: OctaveOliviers
# @        on: 2021-04-01T15:07:04+02:00
# @Last modified by: octave
# @              on: 2021-05-25T17:48:47+01:00


using Plots ; plotly()
using ProgressBars

include("../mdp/utils.jl")
include("../mces/utils.jl")


# define MDP
begin
    gam = 0.9 ;
    S = [[1 0 0];
         [1 0 0];
         [0 1 0];
         [0 0 1]]
    T = [[0.9 0.8 0 0];
         [0.1 0 1 0];
         [0 0.2 0 1]]
    r = [1.1, 1.2, 0, 0] ;
    P = [[1 0 0];
         [0 0 0];
         [0 1 0];
         [0 0 1]]
    q = compute_q_policy(P, T, r, gam)

    mdp = MDP(size(S,2), size(S,1), gam, S, P, T, r, q, [3,4])
end

# initialise mces solver
mces = MCES(mdp, seed=13)

### General episode-wise MCES
# run_mces!(mces, mdp, num_epi=1e6)

### Expected MCES
# run_mces_exp!(mces, mdp, num_epi=1e5)

num_epi = 1e5
p = plot()

q_min = 4.
q_prec = 0.5
q_max = 6.5

grid_side = q_min:q_prec:q_max
q1_grid = repeat(grid_side, 1, length(grid_side))
q2_grid = repeat(grid_side', length(grid_side), 1)
#
# q1_grid = grid_side
# q2_grid = grid_side .- 0.05
#
# q1_grid = [4.6]
# q2_grid = [4.59]

# bias in update
p_max = 0.9
p_last = 1e-3
p_min = 1. - p_max

for q0 = eachrow([q1_grid[:] q2_grid[:]])
    println("Simulating from ", q0)

    # total number of visits
    tot_vis = zeros(Float32, mdp.num_sa)

    # initialise MCES solver
    mces.q[1:2] = q0
    mces.policy = compute_policy(mdp.structure, mces.q)

    path = []
    append!(path, [mces.q[1:2]])

    for k = ProgressBar(1:num_epi)
        # change prior of mces solver
        if mces.policy[1,1] == 1
            mces.prior = [ p_min, p_max, p_last/2, p_last/2]
        else
            mces.prior = [ p_min, p_max, p_last/2, p_last/2]
        end

        # perform one mces expected step
        # mces_exp_step!(mces, mdp, tot_vis, max_len_epi=1e2)
        epi_stp = compute_q_policy(mces.policy, mdp.transitions, mdp.rewards, mdp.discount) - mces.q

        # update q-estimates
        mces.q += epi_stp.*mces.prior/k
        # compute policy matrix from q-values
        mces.policy = compute_policy(mdp.structure, mces.q)

        append!(path, [mces.q[1:2]])

        # check for convergence
        if all(abs.(mdp.q-mces.q) .< TOLERANCE)
            break
        end
    end

    stp = 10^1
    plot!(hcat(path...)[1,1:stp:end], hcat(path...)[2,1:stp:end], color=:blue)
end
plot!(q_min:q_max, q_min:q_max, color=:yellow)

# plot limit points
# for opitmal policy
q_pol = copy(mdp.q)
d_g = q_pol[1]-q_pol[2]
q_g = q_pol[1] - p_min/(p_max-p_min)*d_g
scatter!((q_g, q_g), markershape=:circle, color=:green)
scatter!((q_pol[1], q_pol[2]), markershape=:xcross, color=:green)

# for suboptimal policy
P_ = copy(P); P_[1,1]=0; P_[2,1]=1;
q_pol =  compute_q_policy(P_, mdp.transitions, mdp.rewards, mdp.discount)
d_r = q_pol[1]-q_pol[2]
q_r = q_pol[1] - p_max/(p_min-p_max)*d_r
scatter!((q_r, q_r), markershape=:circle, color=:red)
scatter!((q_pol[1], q_pol[2]), markershape=:xcross, color=:red)

# point of convergence
q1_tilde = p_min/(p_max+p_min)*mdp.q[1] + p_max/(p_min+p_max)*q_pol[1]
scatter!((q1_tilde, q1_tilde), markershape=:vline, color=:blue)
q2_tilde = p_max/(p_max+p_min)*mdp.q[2] + p_min/(p_min+p_max)*q_pol[2]
scatter!((q2_tilde, q2_tilde), markershape=:hline, color=:blue)

display(p)

# (mdp.q[1] - q_pol[2])/(mdp.q[1]+mdp.q[2] - q_pol[1]-q_pol[2])

@info "results" mces.q mdp.q
