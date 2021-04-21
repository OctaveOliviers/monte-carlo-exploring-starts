# @Created by: OctaveOliviers
# @        on: 2021-04-01T15:07:04+02:00
# @Last modified by: octave
# @              on: 2021-04-14T07:42:57+02:00


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
        # q = [5, 5.1, 0, 0]
        # initialise policy
        policy = compute_policy(mdp.structure, q)
        # initialise prior
        prior = normalize(rand(mdp.num_sa), 1)

        new(q, policy, prior)
    end
end # struct MCES
