"""

"""
import math
from typing import Optional, Union

import numpy as np

import gym
from gym import logger, spaces
from gym.error import DependencyNotInstalled


class RoboCropEnvV3(gym.Env):
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
    gym.make('RoboCrop-v3')
    ```
    No additional arguments are currently supported.
    """
    # Possible actions
    NONE = 0
    PLOW = 1
    SEED = 2
    WATER = 3
    HARVEST = 4
    # Possible states
    UNPLOWED = 0
    PLOWED  = 1
    SEEDED  = 2
    GROWING = 3
    MATURE  = 4
    

    metadata = {'render.modes': ['human']}

    def __init__(self, max_episode_steps=500):
        super(RoboCropEnvV3, self).__init__()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=np.array([0, 0]), high=np.array([4, 24]), dtype=np.int32)
        # self.observation_space = spaces.Dict({
        #     "soil": spaces.Box(low=np.array([0]), high=np.array([4]), dtype=np.int32), 
        #     "time": spaces.Discrete(3)}) 
        self.state = 0
        self.max_episode_steps = max_episode_steps
        self.episode_steps = 0

    def get_reward(self, action, soil, time):
        if action == self.NONE:
            self.state = (soil, time+1)
            return 0
        elif action == self.PLOW:
            if self.state == self.UNPLOWED:
                self.state = (self.PLOWED, time+1)
                return 1
            else:
                self.state = (self.PLOWED, time+1)
                return -1
        elif self.state == self.PLOWED and action == self.SEED:
            self.state = (self.SEEDED, time+1)
            return 1
        elif action == self.WATER:
            if self.state == self.GROWING:
                self.state = (self.MATURE, time+1)
                return 1
            elif self.state == self.SEEDED:
                self.state = (self.GROWING, time+1)
                return 1
            else:
                return -1
        elif self.state == self.MATURE and action == self.HARVEST:
            self.state = (self.UNPLOWED, time+1)
            return 10
        else:
            self.state = (soil, time+1)
            return -1

    def step(self, action):
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        assert self.observation_space is not None, "Call reset before using step method."
        # Observation given action and state
        soil, time = self.state
        reward = self.get_reward(action, soil, time)

        # Reward given action
        self.episode_steps += 1
        done = self.episode_steps >= self.max_episode_steps
        info = {}

        return np.array(self.state, dtype=np.int32), reward, done, info

    
    def reset(self):
        # Start as plowed
        self.state = self.PLOWED
        self.episode_steps = 0
        return self.state

    def render(self, mode='human'):
        pass

    def close(self):
        pass