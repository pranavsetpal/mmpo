import abc

import chex
import jax
import jax.numpy as jnp

from jumanji.environments.routing.multi_cvrp.types import State
from jumanji.environments.routing.multi_cvrp.utils import worst_case_remaining_reward


class RewardFn(abc.ABC):
    def __init__(self, num_vechicles: int, num_customers: int, map_max: int) -> None:
        self._num_vehicles = num_vechicles
        self._num_customers = num_customers
        self._map_max = map_max

    @abc.abstractmethod
    def __call__(
        self,
        state: State,
        new_state: State,
        is_done: bool,
    ) -> chex.Numeric:
        """Compute the reward based on the current state, the next state and
        whether the episode is done.
        """


class DistanceReward(RewardFn):
    """
    The negative distance between the current city and the chosen next city to go to.
        An time penalty is also added when arriving early or late at a customer.
        If the maximum step limit is reached, the reward is set to an esitated of
        the worst case reward that can still be achieved.
    """

    def __call__(
        self,
        state: State,
        new_state: State,
        is_done: bool,
    ) -> chex.Numeric:
        def compute_reward(state: State, new_state: State) -> float:

            step_vehicle_distance_penalty = (
                state.vehicles.distances.sum() - new_state.vehicles.distances.sum()
            )

            return jax.lax.cond(  # type: ignore
                jnp.any(new_state.step_count > self._num_customers * 2),
                # Penalise for running into step limit. This is not including max time
                # penalties as the distance penalties are already enough.
                lambda: worst_case_remaining_reward(new_state),
                lambda: step_vehicle_distance_penalty,
            )

        # By default, returns the negative distance between the previous and new node.
        reward = compute_reward(state, new_state)

        return reward
