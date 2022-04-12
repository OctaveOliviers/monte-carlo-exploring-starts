# @Created by: OctaveOliviers
# @        on: 2021-04-01T15:07:04+02:00
# @Last modified by: octave
# @              on: 2022-01-31T13:26:07+00:00


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
