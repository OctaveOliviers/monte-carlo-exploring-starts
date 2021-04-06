# @Created by: OctaveOliviers
# @        on: 2021-04-01T15:07:04+02:00
# @Last modified by: OctaveOliviers
# @              on: 2021-04-06T13:09:06+02:00


include("mdp.jl")
include("mces.jl")
include("utils.jl")

gam = 0.9 ;
p = [0.1, 0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 0.1]

S = [[1 0 0 0 0 0];
     [0 1 0 0 0 0];
     [0 1 0 0 0 0];
     [0 0 1 0 0 0];
     [0 0 1 0 0 0];
     [0 0 0 1 0 0];
     [0 0 0 0 1 0];
     [0 0 0 0 0 1]]

T = [[0 0 0 0 0 0 0 0];
     [1 0 0 .8 0 0 0 0];
     [0 0 1 0 0 0 0 0];
     [0 1 0 0 0 1 0 0];
     [0 0 0 0 1 0 1 0];
     [0 0 0 .2 0 0 0 1]]

r = [0., 10., 0., -10., -20., 0., 0., 0.] ;

P = [[1 0 0 0 0 0];
     [0 1 0 0 0 0];
     [0 0 0 0 0 0];
     [0 0 1 0 0 0];
     [0 0 0 0 0 0];
     [0 0 0 1 0 0];
     [0 0 0 0 1 0];
     [0 0 0 0 0 1]]
q = policy_q(P, T, r, gam)

mdp = MDP(6, 8, gam, S, P, T, r, q, [6,7,8])

q0 = 100*rand(Float64, 8)
P0 = compute_policy(S, q0)
mces = MCES(q0, P0, p)

run_mces!(mces, mdp, num_epi=1e6, seed=42)

@info "results" mces.q mdp.q
