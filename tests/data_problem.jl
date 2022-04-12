# @Created by: octave
# @        on: 2022-04-12T13:36:33+01:00
# @Last modified by: octave
# @              on: 2022-04-12T14:18:53+01:00



include("../mdp/utils.jl")

"""
Single state two actions
"""
# mdp data
num_s = 1
num_sa = 2
discount = 0.8
structure = hcat([1, 1]) # matrix
transitions = [1 1] # matrix
rewards = [1., 0.] # vector
policy = hcat([1, 0]) # matrix
term_states = []
q = pol2val(policy, transitions, rewards, discount)

# mces data
q0 = [1., 2.]
prior = Diagonal([0.5, 0.5])


"""
2 states 2 actions for spiral behaviour
"""
# # mdp data
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
#
# # mces data
# q0 = [1.1, 0.1, 4.0, 3.9] # for weird nonmonotone behaviour
# q0 = [1.9, 2, 2.95, 3] # for weird spiral behaviour
# prior = Diagonal([0.01, 0.4, 0.4, 0.01]) # for weird spiral behaviour
# prior = Diagonal([0.1, 0.4, 0.4, 0.1]) # for weird nonmonotone behaviour


"""
Big random MDP
"""
# Random.seed!(317)
#
# # mdp data
# num_sa = 40
# num_s = 10
# mdp = generate_mdp(num_sa, num_s)
#
# # mces data
# q = 100*rand(Float32, num_sa)
# prior = Diagonal(rand(Float32, num_sa))


"""
Single state two actions
"""
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
