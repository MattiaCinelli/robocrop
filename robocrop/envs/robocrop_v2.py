"""

"""
import math
from typing import Optional, Union

import numpy as np

import gymnasium as gym
from gymnasium import logger, spaces
from gymnasium.error import DependencyNotInstalled
from robocrop.common import Farm

class RoboCropEnvV2(Farm, gym.Env):
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
    | 3   | Plant is growing |
    | 4   | Plant is mature  |
    

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
    gym.make('RoboCrop-v2')
    ```
    No additional arguments are currently supported.
    """    

    metadata = {'render.modes': ['human']}

    def __init__(self, max_episode_steps=200):
        super(RoboCropEnvV2, self).__init__()
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
        elif action == self.WATER:
            if self.state == self.GROWING:
                self.state = self.MATURE
                return 1
            elif self.state == self.SEEDED:
                self.state = self.GROWING
                return 1
            else:
                return -1
        elif self.state == self.MATURE and action == self.HARVEST:
            self.state = self.UNPLOWED
            return 10
        else:
            return -1

    
    def reset(self):
        # Start as plowed
        self.state = self.PLOWED
        self.episode_steps = 0
        return self.state

    def render(self, mode='human'):
        pass

    def close(self):
        pass