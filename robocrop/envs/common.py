import numpy as np
from dataclasses import dataclass

@dataclass
class Farm:
    """
    ### Action Space
    The action is a `ndarray` with shape `(4,)` which can take values `{0, 3}` indicating the direction
     of the fixed force the cart is pushed with.
    | Num | Action     |
    |-----|------------|
    | 0   | Plow       |
    | 1   | Seed       |
    | 2   | Water      |
    | 3   | Harvest    |

    
    ### Observation Space
    The observation is a `ndarray` with shape `(1,)` with the values corresponding to the following attributes:
    | Num | Observation      |
    |-----|------------------|
    | 0   | Ground Unplowed  |
    | 1   | Ground Plowed    |
    | 2   | Seed planted     |
    | 3   | Plant is mature  |
    """

    metadata = {'render_modes': ['human', 'text']}
    # Possible actions
    PLOW: np.ndarray = np.array([0], dtype=np.int32)
    SEED: np.ndarray = np.array([1], dtype=np.int32)
    WATER: np.ndarray = np.array([2], dtype=np.int32)
    HARVEST: np.ndarray = np.array([3], dtype=np.int32)
    # Possible states
    UNPLOWED: np.ndarray = np.array([0], dtype=np.int32)
    PLOWED: np.ndarray = np.array([1], dtype=np.int32)
    SEEDED: np.ndarray = np.array([2], dtype=np.int32)
    MATURE: np.ndarray = np.array([3], dtype=np.int32)

    def step(self, action):
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        assert self.observation_space is not None, "Call reset before using step method."
        self.lastaction = int(action)
        # Observation given action and state
        reward = self.get_reward(action)

        # Reward given action
        self.episode_steps += 1
        done = self.episode_steps >= self.max_episode_steps

        info = {}
        return self.state, reward, done, False, info

    
"""
    # Possible actions
    PLOW: 0
    SEED: 1
    WATER: 2
    HARVEST: 3
    # Possible states
    UNPLOWED: 0
    PLOWED: 1
    SEEDED: 2
    GROWING: 3
    MATURE: 4
"""