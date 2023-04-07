"""

"""
import numpy as np

import gymnasium as gym
from gymnasium import spaces

class RoboCropEnvV1(gym.Env):
    """
    ### Description
    This environment simulate a simple farming crop robot.
    
    The ground has to be seeded, watered and the crop has to be harvested in order to get a reward.

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
    

    ### Rewards
    The goal is to harvest a full size crop. The reward is 10, -1 for any other action.
    
    ### Starting State
    The first observation is `(0)`
    
    ### Episode End
    The episode ends if any one of the following occurs:
    1. Termination: Score equals or exceeds 100
    2. Truncation: Episode length is greater than 200

    ### Arguments
    ```
    gym.make('RoboCrop-v1')
    ```
    No additional arguments are currently supported.
    """
    # Possible actions
    PLOW = 0
    SEED = 1
    WATER = 2
    HARVEST = 3
    # Possible states
    UNPLOWED = 0
    PLOWED = 1
    SEEDED = 2
    MATURE = 3
    

    metadata = {'render.modes': ['human']}

    def __init__(self, max_episode_steps=200):
        super(RoboCropEnvV1, self).__init__()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=np.array([0]), high=np.array([4]), dtype=np.int32)
        self.state = 0
        self.max_episode_steps = max_episode_steps
        self.episode_steps = 0

    def get_reward(self, action):
        if action == self.PLOW:
            if self.state == self.UNPLOWED:
                self.state = self.PLOWED
                return 1
            else:
                self.state = self.PLOWED
                return -1
        elif self.state == self.PLOWED and action == self.SEED:
            self.state = self.SEEDED
            return 1
        elif self.state == self.SEEDED and action == self.WATER:
            self.state = self.MATURE
            return 1
        elif self.state == self.MATURE and action == self.HARVEST:
            self.state = self.UNPLOWED
            return 10
        else:
            return -1

    def step(self, action):
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        assert self.observation_space is not None, "Call reset before using step method."
        # Observation given action and state
        reward = self.get_reward(action)

        # Reward given action
        self.episode_steps += 1
        done = self.episode_steps >= self.max_episode_steps

        info = {}
        return self.state, reward, done, info

    
    def reset(self, **kwargs):
        # Start as plowed
        self.state = self.PLOWED
        self.episode_steps = 0
        return self.state

    def render(self, mode='human'):
        pass

    def close(self):
        pass