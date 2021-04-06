# @Created by: OctaveOliviers
# @        on: 2021-04-01T15:07:04+02:00
# @Last modified by: OctaveOliviers
# @              on: 2021-04-06T13:08:54+02:00


include("mdp/utils.jl")
include("mces/utils.jl")

SEED = 42

# create markov decision problem
num_s, num_sa = 5, 20
mdp = generate_mdp(num_sa, num_s; discount=0.5, seed=SEED)

# create two MCES solvers
mces1 = MCES(mdp, seed=13)
mces2 = MCES(mdp, seed=11)

# apply MCES steps on each
mces_step_all_sa_update!(mces1, mdp; max_num_update=1e4, seed=NO_SEED)
mces_step_all_sa_update!(mces2, mdp; max_num_update=1e4, seed=NO_SEED)

# compute base distance
old_dist = maximum(abs.(mces1.q - mces2.q))
println(old_dist)

# update
mces_step_all_sa_update!(mces1, mdp; max_num_update=1e4, seed=NO_SEED)
mces_step_all_sa_update!(mces2, mdp; max_num_update=1e4, seed=NO_SEED)

# compute new distance
new_dist = maximum(abs.(mces1.q - mces2.q))
println(new_dist)

print(new_dist <= mdp.discount * old_dist)
