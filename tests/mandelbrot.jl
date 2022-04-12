# @Created by: octave
# @        on: 2022-01-06T14:08:49+01:00
# @Last modified by: octave
# @              on: 2022-01-06T15:31:08+01:00


using Distributed
using DifferentialEquations
using LinearAlgebra
using Plots ; plotly()
using Random

addprocs()
@everywhere using DifferentialEquations

include("../mdp/utils.jl")
include("../mces/utils.jl")

# Random.seed!(317)

### Define ODE

# num_sa = 40
# num_s = 10
# mdp = generate_mdp(num_sa, num_s)

@everywhere begin
    num_s, num_sa, discount = 2, 4, 0.8
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

    prior = Diagonal([0.01, 0.4, 0.4, 0.01])

    q = compute_q_policy(policy, transitions, rewards, discount)
    mdp = MDP(num_s, num_sa, discount, structure, policy, transitions, rewards, q, term_states)

    @everywhere function mces!(dq, q, p, t)
        pol = compute_policy(mdp.structure, q)
        q_pol = compute_q_policy(pol, mdp)
        dq .= prior*(q_pol - q)
    end


    ### Define ensemble problem

    # s1_q0 =
    s2_q0 = [2., 3.]

    q1_start,     q2_start     = 0., 0.
    q1_step,      q2_step      = 1., 1.
    q1_num_step,  q2_num_step  = 2, 2
    num_init_cond = q1_num_step * q2_num_step

    initial_conditions(i) = [q1_start + mod(i-1,q1_num_step)*q1_step,
                                         q2_start + floor((i-1)/q2_num_step)*q2_step,
                                         s2_q0[1],
                                         s2_q0[2]]

    function prob_func(prob,i,repeat)
      remake(prob, u0 = initial_conditions(i))
    end
end
tspan = (0.0, 5.0)
prob = ODEProblem(mces!, initial_conditions(1), tspan)
ensemble_prob = EnsembleProblem(prob, prob_func = prob_func)

### Solve ODE fro all initial conditions
sim = solve(ensemble_prob, Tsit5(), EnsembleDistributed(), trajectories = num_init_cond)
