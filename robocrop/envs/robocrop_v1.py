"""

"""
# %%
import numpy as np
import gymnasium as gym
from gymnasium import logger, spaces
from typing import Optional
from common import Farm

# %%
class RoboCropEnvV1(Farm, gym.Env):
    """
    ### Description
    This environment simulate a simple farming crop robot.
    
    The ground has to be seeded, watered and the crop has to be harvested in order to get a reward.

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
    
    def __init__(self, max_episode_steps=200):
        super(RoboCropEnvV1, self).__init__()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=np.array([0]), high=np.array([4]), dtype=np.int32)
        self.state = np.array([0], dtype=np.int32)
        self.max_episode_steps = max_episode_steps
        self.episode_steps = 0
        self.state_hystory = []
        self.action_hystory = []


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


    def reset(
            self, 
            seed: Optional[int] = None, 
            options: Optional[dict] = None,
            ):
        if options is None:
            options = {}
        # Start as plowed
        super().reset(seed=seed)
        self.state = self.PLOW
        self.episode_steps = 0
        self.state_hystory = []
        self.action_hystory = []
        self.lastaction = None
        return (self.state, options)


    def render(self, render_mode='text'):
        self._render_text()


    def _render_text(self):
        self.state_hystory.extend(self.state)
        self.action_hystory.append(self.lastaction)

    
    def close(self):
        pass
