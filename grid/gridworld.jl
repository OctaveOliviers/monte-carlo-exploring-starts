# @Created by: octave
# @        on: 2021-04-21T10:29:54+02:00
# @Last modified by: octave
# @              on: 2021-04-21T10:36:29+02:00


mutable struct GridWorld
    """
    struct for gridworld
    """

    grid::Matrix
    policy::Matrix

    function GridWorld(
            grid::Matrix,
            policy::Matrix
        )
        """
        constructor
        """
        new(grid, policy)
    end
end # struct GridWorld
