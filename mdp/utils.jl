# @Created by: OctaveOliviers
# @        on: 2021-04-01T16:13:05+02:00
# @Last modified by: octave
# @              on: 2022-04-12T14:15:30+01:00


using Random
using StatsBase
using LinearAlgebra

include("mdp.jl")
# include("../mces/mces.jl")
include("../params.jl")


function create_rand_mdp(
        num_sa::Integer,
        num_s::Integer;
        discount::Real=DISCOUNT,
        seed::Integer=NO_SEED
    )::MDP
    """
    explain
    """
    # set random number generator
    if seed != NO_SEED; Random.seed!(seed); end

    # number of terminal states
    # TODO have single terminal state
    num_term = rand(1:Int(ceil(num_s/3)))
    # create structure
    structure = create_structure(num_s, num_sa, num_term)
    # create transition probabilities
    transitions = create_transitions(num_s, num_sa, num_term)
    # store terminal states of MDP
    term_states = [i for i=(num_s-num_term+1):num_s]
    # action-values
    q = create_q(num_s, num_sa, num_term)
    # optimal policy
    policy = val2pol(structure, q)
    # compute rewards
    rewards = (I - discount*transitions'*policy') * q

    return MDP(num_s, num_sa, discount, structure, policy, transitions, rewards, q, term_states)
end


function create_structure(
        num_s::Integer,
        num_sa::Integer,
        num_term::Integer
    )::Matrix
    """
    explain
    """
    # assert that there are not too many (state, action) pairs
    # @assert num_s <= num_sa <= num_s^2

    # structure matrix
    structure = zeros(Int8, num_sa, num_s)
    # ensure each state has at least one action
    for s = 1:num_s ; structure[s,s] = 1 ; end
    # assign the other actions randomly
    for sa = (num_s+1):num_sa ; structure[sa, Int(rand(1:(num_s-num_term)))] = 1 ; end

    return structure
end


function create_transitions(
        num_s::Integer,
        num_sa::Integer,
        num_term::Integer
    )::Matrix
    """
    explain
    """
    # transition matrix
    # need first identity to ensure that a terminal state has zero reward
    transitions = [I rand(num_s, num_sa-num_s)]
    transitions[:,1:num_s-num_term] = rand(num_s, num_s-num_term)
    # normalize
    transitions = transitions ./ sum(transitions, dims=1)

    return transitions
end


function create_q(
        num_s::Integer,
        num_sa::Integer,
        num_term::Integer
    )::Vector
    """
    explain
    """
    # q-values
    q = 5*rand(Float32, num_sa)
    # q-values in terminal states are zero
    q[num_s-num_term+1:num_s] .= 0.

    return q
end


function val2pol(
        structure::Matrix,
        q::Vector
    )::Matrix
    """
    Compute greedy policy
    """
    # extract useful info
    num_sa, num_s = size(structure)

    # policy matrix
    policy = zeros(Int8, num_sa, num_s)
    # for each state choose action with highest q-value
    for s = 1:num_s
        # find actions of that state
        sa = findall(!iszero, structure[:,s])
        # index of maximal q value in state s
        idx_max_q = argmax(q[sa])
        # choose the action with maximal q-value
        policy[sa[idx_max_q], s] = 1
    end

    return policy
end


function val2pol_sm(
        structure::Matrix,
        q::Vector,
        b::Real=1.
    )::Matrix
    """
    explain
    """
    # extract useful info
    num_sa, num_s = size(structure)

    # policy matrix
    policy = zeros(Int8, num_sa, num_s)
    # for each state softmax the q-values
    for s = 1:num_s
        # find actions of that state
        sa = findall(!iszero, structure[:,s])
        # softmax policy
        policy[sa, s] = exp.(b*q[sa]) ./ sum(exp.(b*q[sa]))
    end

    return policy
end


function pol2val(
        policy::Matrix,
        transitions::Matrix,
        rewards::Vector,
        discount::Real
    )::Vector
    """
    explain
    """
    # solve Bellman equation
    return (I-discount*transitions'*policy')\rewards
end


function pol2val(
        policy::Matrix,
        mdp::MDP
    )::Vector
    """
    explain
    """
    # solve Bellman equation
    return (I-mdp.discount*mdp.transitions'*policy')\mdp.rewards
end


function rand_pol(
        mdp::MDP,
        seed::Integer=NO_SEED
    )::Matrix
    """
    explain
    """
    # set random number generator
    if seed != NO_SEED; Random.seed!(seed); end

    # policy matrix
    policy = zeros(Int8, mdp.num_sa, mdp.num_s)
    # select random action in each state
    for s = 1:mdp.num_s ; policy[sample(findall(x->x==1, m.structure[:,s])),s] = 1 ; end

    return policy
end
