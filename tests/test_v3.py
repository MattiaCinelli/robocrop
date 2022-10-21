from contextlib import contextmanager
import pytest
import numpy as np

from robocrop.envs.robocrop_v3 import RoboCropEnvV3

# os.chdir(os.path.dirname(__file__) == 
env = RoboCropEnvV3()

@contextmanager
def does_not_raise():
    '''A context manager that does not raise an exception.'''
    yield

def test_reset():
    assert np.array_equal(env.reset(), np.array((1, 0, 0), dtype=np.int32))
    assert env.episode_steps == 0

def test_get_time():
    '''
    Test if the result of reset and step is correct
    '''
    env.reset()
    assert env._get_time(0, 0) == 1
    assert env._get_time(0, 1) == 4
    assert env._get_time(0, 2) == 1
    assert env._get_time(0, 3) == 1
    assert env._get_time(0, 4) == 1
    assert env._get_time(23, 0) == 0
    assert env._get_time(23, 1) == 3

def test_get_reward_multiplier():
    env.reset()
    assert env._get_reward_multiplier(0) == 0.25
    assert env._get_reward_multiplier(3) == 0.5
    assert env._get_reward_multiplier(6) == 0.75
    assert env._get_reward_multiplier(9) == 1.0
    assert env._get_reward_multiplier(12) == 0.75
    assert env._get_reward_multiplier(15) == 0.5
    assert env._get_reward_multiplier(18) == 0.25
    assert env._get_reward_multiplier(21) == 0.0


def test_get_obs():
    env.reset()
    assert env._get_obs(0, 0, 1, 0) == (0, 0, 0)
    assert env._get_obs(0, 1, 1, 0) == (1, 0, 0)
    assert env._get_obs(0, 2, 1, 0) == (2, 0, 0)
    assert env._get_obs(0, 3, 1, 0) == (3, 0, 0)
    assert env._get_obs(0, 4, 1, 0) == (4, 0, 0)
    assert env._get_obs(1, 0, 1, 0) == (1, 1, 0)
    assert env._get_obs(1, 1, 1, 0) == (1, -1, 0)








