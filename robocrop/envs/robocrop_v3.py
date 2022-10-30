"""

"""
import math
from typing import Optional, Union

import numpy as np

import gymnasium as gym
from gymnasium import logger, spaces
from gymnasium.error import DependencyNotInstalled


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
    | Num | Observation      | Min | Max |
    |-----|------------------|-----|-----|
    | 0   | Ground Conditions| 0   | 4   |
    | 1   | Time in hours    | 0   | 23  |
    | 2   | Soil humidity    | 0   | 1   |
    
    #### Ground Conditions
    | 0   | Ground Unplowed 
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

    DRY = 0
    WET = 1
    
    metadata = {'render.modes': ['human']}

    def __init__(self, max_episode_steps=500):
        super(RoboCropEnvV3, self).__init__()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=np.array([0, 0, 0]), high=np.array([4, 23, 1]), dtype=np.int32)
        self.optimal_irrigation_time = 9
        self.state = np.array((self.PLOWED, 0, 0), dtype=np.int32)
        self.max_episode_steps = max_episode_steps
        self.episode_steps = 0

    def _get_obs(self, action, soil_state, reward_multiplier, soil_h20): # Returns soil_state, reward, soil_h20
        if action == self.NONE:
            return soil_state, 0, soil_h20

        elif action == self.PLOW:
            if soil_state == self.UNPLOWED:
                return self.PLOWED, 1, self.DRY
            else:
                return self.PLOWED, -1, self.DRY

        elif soil_state == self.PLOWED and action == self.SEED:
            return self.SEEDED, 1, self.DRY

        elif action == self.WATER:
            if soil_h20 != self.DRY:
                return soil_state, -1, self.WET
            if soil_state == self.GROWING:
                return self.MATURE, 1*reward_multiplier, self.WET
            elif soil_state == self.SEEDED:
                return self.GROWING, 1*reward_multiplier, self.WET
            else:
                return soil_state, -1, self.WET
        
        elif soil_state == self.MATURE and action == self.HARVEST:
            return self.UNPLOWED, 10*reward_multiplier, self.DRY

        else:
            return soil_state, -1, soil_h20

    def _get_reward_multiplier(self, time):
        return 1-min(abs(time-self.optimal_irrigation_time), 24-abs(time-self.optimal_irrigation_time))/12

    def _get_time(self, time, action):
        time += 4 if action == self.PLOW else 1
        return time if time<=23 else time-24

    def step(self, action):
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        assert self.observation_space is not None, "Call reset before using step method."

        # Observation given action and state
        soil_state, time, soil_h20 = self.state
        reward_multiplier = self._get_reward_multiplier(time)
        soil_state, reward, soil_h20 = self._get_obs(action, soil_state, reward_multiplier, soil_h20)
        new_time = self._get_time(time, action)
        soil_h20 = 0 if time == 0 else soil_h20 # reset soil_h20 to dry at midnight
        self.state = np.array((soil_state, new_time, soil_h20), dtype=np.int32)

        # Episode progress
        self.episode_steps += 1
        done = self.episode_steps >= self.max_episode_steps
        info = {}

        return self.state, reward, done, info

    
    def reset(self):
        # Start as plowed
        self.state = np.array((self.PLOWED, 0, 0), dtype=np.int32)
        self.episode_steps = 0
        return self.state

    def render(self, mode='human'):
        pass

    def close(self):
        pass