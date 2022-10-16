"""

"""
import math
from typing import Optional, Union

import numpy as np

import gym
from gym import logger, spaces
from gym.error import DependencyNotInstalled


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

    def __init__(self, max_episode_steps=2000):
        super(RoboCropEnvV1, self).__init__()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(5)
        self.state = 0
        self.max_episode_steps = max_episode_steps
        self.episode_steps = 0

    def get_reward(self, action):
        if self.state == self.UNPLOWED and action == self.PLOW:
            self.state = self.PLOWED
            return 1
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
        done = self.episode_steps >= self.max_episode_steps
        self.episode_steps += 1
        info = {}
        return self.state, reward, done, info
        # return self.state.astype(np.int), reward, done, info
        # return np.array([self.state]).astype(np.int32), reward, done, info
        # return self.state.astype(np.float32), reward, done, info
        
    
    def reset(self):
        # super().reset()
        # Start as plowed
        self.state = 0
        # here we convert to float32 to make it more general (in case we want to use continuous actions)
        # return np.array([self.state]).astype(np.int32)
        # return self.astype(np.int)
        return self.state

    def render(self, mode='human'):
        pass

    def close(self):
        pass