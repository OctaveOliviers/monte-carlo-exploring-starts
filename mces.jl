# @Created by: OctaveOliviers
# @        on: 2021-04-01T15:07:04+02:00
# @Last modified by: octave
# @              on: 2022-04-12T15:38:54+01:00



using DifferentialEquations
using LinearAlgebra
using StatsBase

include("mdp.jl")
include("params.jl")
include("libraries.jl")


mutable struct MCES
    """
    struct for Monte Carlo Exploring Starts solver
    """

    q::Vector
    policy::Matrix
    prior::Vector

    function MCES(
            mdp::MDP;
            seed::Integer=NO_SEED
        )
        """
        constructor
        """
        # set random number generator
        if seed != NO_SEED; Random.seed!(seed); end

        # initialise q-values
        q = rand(mdp.num_sa)
        # initialise policy
        policy = compute_policy(mdp.structure, q)
        # initialise prior
        prior = normalize(rand(mdp.num_sa), 1)

        new(q, policy, prior)
    end
end # struct MCES


function run_mces!(
        mces::MCES,
        mdp::MDP;
        max_num_epi::Real=EPISODE_MAX_NUMBER,
        max_len_epi::Real=EPISODE_MAX_LENGTH,
        tol::Real=TOLERANCE,
        seed::Integer=NO_SEED
    )::Nothing
    # c=Array{Float64}(undef, 0))
    """
    explain
    """
    # set random number generator
    if seed != NO_SEED; Random.seed!(seed); end

    # number of times that each state-action is visited
    tot_vis = zeros(Integer, mdp.num_sa)

    # loop until convergence
    for k = ProgressBar(1:max_num_epi)
        # run one simulation down the MDP and update the parameters
        mces_step!(mces, mdp, tot_vis, max_len_epi=max_len_epi, seed=rand(UInt64))

        # check for convergence
        if all(abs.(mdp.q-mces.q) .< tol); break; end

        # compute contraction
        # append!(c, maximum(abs.(mdp.q-mces.q)))
    end

    return nothing
end


function mces_step!(
        mces::MCES,
        mdp::MDP,
        total_visits::Vector;
        max_len_epi::Real=EPISODE_MAX_LENGTH,
        seed::Integer=NO_SEED
    )::Nothing
    """
    explain
    """
    # generate episode
    epi_sa, epi_r, epi_vis = generate_episode(mces, mdp, max_len_epi=max_len_epi, seed=seed)
    # update q-estimates
    update_q_values!(mces.q, epi_sa, epi_r, total_visits, epi_vis, mdp.discount)
    # update policy matrix from q-values
    update_policy!(mces.policy, mdp.structure, mces.q)

    return nothing
end


function mces_step_all_sa_update!(
        mces::MCES,
        mdp::MDP;
        max_num_update::Real=EPISODE_MAX_NUMBER,
        max_len_epi::Real=EPISODE_MAX_LENGTH,
        seed::Integer=NO_SEED
    )::Nothing
    """
    explain
    """
    # assert exploring starts assumption
    @assert all(mces.prior .> 0)

    # set random number generator
    if seed != NO_SEED; Random.seed!(seed); end

    # number of times that each state-action is visited
    tot_vis = zeros(Integer, mdp.num_sa)

    # apply MCES update until every state-action has been visited once
    while any(tot_vis .< 1)
        # run one simulation down the MDP and update the parameters
        mces_step!(mces, mdp, tot_vis, max_len_epi=max_len_epi, seed=rand(UInt64))

        if sum(tot_vis) >= max_num_update*max_len_epi;
            println("reached max number of updates.")
            break
        end
    end

    return nothing
end


function generate_episode(
        mces::MCES,
        mdp::MDP;
        max_len_epi::Real=EPISODE_MAX_LENGTH,
        seed::Integer=NO_SEED
    )::Tuple{Vector,Vector,Vector}
    """
    explain
    """
    # set random number generator
    if seed != NO_SEED; Random.seed!(seed); end

    # store the state-actions and rewards encountered in the episode
    sa, r = Vector{Integer}(undef, 0), Vector{AbstractFloat}(undef, 0)
    # store number of visits of each state-action in this episode
    n_vis = zeros(Integer, mdp.num_sa)

    function take_step(transition_prob)
        # choose state-action pair according to weights in transition_prob
        append!(sa, sample(1:mdp.num_sa, Weights(transition_prob)))
        # store reward of initial state-action
        append!(r, mdp.rewards[sa[end]])
        # update number of visits of initial state-action
        n_vis[sa[end]] += 1
    end

    # initialise episode
    take_step(mces.prior)
    # generate an episode from initial state
    while !(sa[end] in mdp.terminal_state_action)
        # go to new state-action
        take_step((mces.policy*mdp.transitions)[:,sa[end]])
        # exit if episode is too long
        if length(sa) >= max_len_epi
            println("Terminate because reached maximal length of episode.")
            break
        end
    end

    return sa, r, n_vis
end


function update_q_values!(
        q::Vector,
        episode_sa::Vector,
        episode_r::Vector,
        total_visits::Vector,
        episode_visits::Vector,
        discount::Real
    )::Nothing
    """
    explain
    """
    # update total number of visits of each state-action
    total_visits .+= episode_visits

    # goal value
    g = 0.
    # update backwards
    reverse!(episode_sa)
    reverse!(episode_r)
    # loop over each step of the episode
    for t in 1:length(episode_sa)
        # update goal value
        g = discount*g + episode_r[t]
        # update q-estimate with incremental mean formula
        episode_visits[episode_sa[t]] -= 1
        step_size = 1. / (total_visits[episode_sa[t]] - episode_visits[episode_sa[t]])
        q[episode_sa[t]] += step_size*(g - q[episode_sa[t]])
    end

    return nothing
end


function update_q_values(
        q::Vector,
        episode_sa::Vector,
        episode_r::Vector,
        total_visits::Vector,
        episode_visits::Vector,
        discount::Real
    )::Vector
    """
    explain
    """
    # new q-values
    q_new = deepcopy(q)
    # update q-values in place
    update_q_values!(q_new, episode_sa, episode_r, total_visits, episode_visits, discount)

    return q_new
end


function update_policy!(
        policy::Matrix,
        structure::Matrix,
        q::Vector
    )::Nothing
    """
    explain
    """
    policy .= val2pol(structure, q)

    return nothing
end


function update_policy(
        policy::Matrix,
        structure::Matrix,
        q::Vector
    )::Matrix
    """
    explain
    """
    # new policy
    policy_new = deepcopy(policy)
    # update policy in place
    update_policy!(policy_new, structure, q)

    return policy_new
end


function simulate_continuous_mces(
        mdp::MDP,
        q0::Vector,
        prior::Diagonal;
        t_init::Real=0.0,
        t_end::Real=5.0,
        alg=Rosenbrock23()
    )
    """
    Continuous time MCES step
    """

    function mces!(
            dq::Vector,
            q::Vector,
            prior::Diagonal,
            t::Real
        )
        pol = val2pol(mdp.structure, q)
        q_pol = pol2val(pol, mdp)
        dq .= prior*(q_pol - q)
    end

    tspan = (t_init, t_end)
    prob = ODEProblem(mces!, q0, tspan, prior)
    sol = solve(prob, alg, reltol=1e-6, abstol=1e-6, maxiters=1e7)

    return sol
end


function run_mces_exp!(
        mces::MCES,
        mdp::MDP;
        max_num_epi::Real=EPISODE_MAX_NUMBER,
        max_len_epi::Real=EPISODE_MAX_LENGTH,
        tol::Real=TOLERANCE
    )::Nothing
    """
    explain
    """
    # number of times that each state-action is visited
    tot_vis = zeros(Integer, mdp.num_sa)

    # loop until convergence
    for k = ProgressBar(1:max_num_epi)
        # perform one mces expected step
        mces_exp_step!(mces, mdp, tot_vis, max_len_epi=max_len_epi)

        # check for convergence
        if all(abs.(mdp.q-mces.q) .< tol); break; end
    end

    return nothing
end


function mces_exp_step!(
        mces::MCES,
        mdp::MDP,
        total_visits::Vector;
        max_len_epi::Real=EPISODE_MAX_LENGTH
    )::Nothing
    """
    explain
    """
    # episode step and number of visits
    epi_stp = pol2val(mces.policy, mdp.transitions, mdp.rewards, mdp.discount) - mces.q
    epi_vis = sum([(mces.policy*mdp.transitions)^i*mces.prior for i = 0:max_len_epi])

    total_visits += epi_vis

    # update q-estimates
    mces.q += epi_stp.*epi_vis./total_visits
    # compute policy matrix from q-values
    mces.policy = pol2val(mdp.structure, mces.q)

    return nothing
end
