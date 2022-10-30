"""

"""
import math
from typing import Optional, Union

import numpy as np
from ..logs import logging

import gym
import gymnasium
from gymnasium import logger, spaces
from gymnasium.error import DependencyNotInstalled


class RoboCropEnvV4(gym.Env):
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
    # Possible actions
    PLOW = 0
    SEED = 1
    WATER = 2
    HARVEST = 3

    # Possible states
    UNPLOWED = 0
    PLOWED  = 1
    SEEDED  = 2
    GROWING = 3
    MATURE  = 4
    

    metadata = {'render.modes': ['human']}

    def __init__(self, max_episode_steps=200):
        super(RoboCropEnvV4, self).__init__()
        # self.action_space = spaces.Box(low=np.array([0, 0]), high=np.array([4, 4]), dtype=np.int32)
        self.action_space = spaces.MultiDiscrete(np.array([4, 4]))
        self.observation_space = spaces.Box(low=np.array([0, 0]), high=np.array([4, 4]), dtype=np.int32)

        self.states = np.array([self.PLOWED, self.PLOWED], dtype=np.int32)
        self.max_episode_steps = max_episode_steps
        self.episode_steps = 0

        self.logger = logging.getLogger(__name__)
        self.logger.info("Starting RoboCropEnvV4")

    def get_reward(self, action, state):
        if action == self.PLOW:
            return (1, self.PLOWED) if state == self.UNPLOWED else (-1, self.PLOWED)
        elif state == self.PLOWED and action == self.SEED:
            return 1, self.SEEDED
        elif action == self.WATER:
            if state == self.GROWING:
                return 1, self.MATURE
            elif state == self.SEEDED:
                return 1, self.GROWING
            else:
                return -1, state
        elif state == self.MATURE and action == self.HARVEST:
            return 10, self.UNPLOWED
        else:
            return -1, state

    def step(self, actions):
        self.logger.debug(f"actions: {actions}")
        err_msg = f"{actions!r} ({type(actions)}) invalid"
        assert self.action_space.contains(actions), err_msg
        assert self.observation_space is not None, "Call reset before using step method."
        # Observation given action and state
        reward = 0
        new_states = []
        for a, s in enumerate(self.states):
            reward, new_state = self.get_reward(actions[a], s)
            reward += reward
            new_states.append(new_state)
        new_states = np.array(new_states, dtype=np.int32)

        # Reward given action
        self.episode_steps += 1
        done = self.episode_steps >= self.max_episode_steps
        
        info = {}
        self.logger.debug(f"new_states: {new_states}, reward: {reward}, done: {done}, info: {info}")
        # return self.action_space.sample(), 1, False, {}
        return new_states, reward, done, info

    
    def reset(self):
        # Start as plowed
        self.episode_steps = 0
        self.max_episode_steps = 200
        # self.states = np.array([self.PLOWED, self.PLOWED], dtype=np.int32)
        self.states = np.array([self.PLOWED], dtype=np.int32)
        return self.states

    def render(self, mode='human'):
        pass

    def close(self):
        pass