# @Created by: OctaveOliviers
# @        on: 2021-04-01T15:07:04+02:00
# @Last modified by: octave
# @              on: 2022-04-12T13:58:50+01:00


struct MDP
    """
    struct for Markov Decision Process
    """

    num_s::Integer
    num_sa::Integer
    discount::Real
    structure::Matrix
    policy::Matrix
    transitions::Matrix
    rewards::Vector
    q::Vector
    terminal_state_action::Vector

    function MDP(
            num_s::Integer,
            num_sa::Integer,
            discount::Real,
            structure::Matrix,
            policy::Matrix,
            transitions::Matrix,
            rewards::Vector,
            q::Vector,
            terminal_state_action::Vector
        )
        """
        constructor
        """
        # assert that all the variables have the appropriate size
        @assert all(map(!iszero, [num_s, num_sa, discount]))
        @assert size(structure) == size(policy) == size(transitions') == (num_sa, num_s)
        @assert size(rewards) == size(q) == (num_sa,)

        new(num_s, num_sa, discount, structure, policy, transitions, rewards, q, terminal_state_action)
    end
end # struct MDP
