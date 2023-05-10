from pyparsing import Any
import pytest
import numpy as np
import gymnasium as gym
from robocrop.common import Farm
from contextlib import contextmanager

# os.chdir(os.path.dirname(__file__))
from stable_baselines3.common.monitor import Monitor
# python3 -m pytest -v -p no:warnings
# python3 -m pytest -v -s

@contextmanager
def does_not_raise():
    '''A context manager that does not raise an exception.'''
    yield

@pytest.fixture
def env():
    'return the enviroment to be tested later'
    return Monitor(gym.make("robocrop.envs:RoboCrop-v3"))

@pytest.mark.filterwarnings("ignore:env") #but it doesn't work TODO
def test_action_space(env: Any):
    assert isinstance(env.action_space, gym.spaces.Discrete)
    assert env.action_space.n == 4

def test_observation_space(env: Any):
    assert isinstance(env.observation_space, gym.spaces.Box)
    assert np.array_equal(env.observation_space.low, np.array([0]))
    assert np.array_equal(env.observation_space.high, np.array([4]))
    assert env.observation_space.dtype == np.int32

def test_reset(env: Any):
    state, options = env.reset()
    assert np.array_equal(state, np.array([0]))
    assert options == {}

# def test_step(env: Any):
#     env.reset()
#     # Test plow action
#     state, reward, done, _, _ = env.step(env.PLOW)
#     assert np.array_equal(state, np.array([1]))
#     assert reward == 1
#     assert state == env.PLOWED
#     assert not done
#     # Test seeding action
#     state, reward, done, _, _ = env.step(env.SEED)
#     assert np.array_equal(state, np.array([2]))
#     assert reward == 1
#     assert state == env.SEEDED
#     assert not done
#     # Test watering action
#     state, reward, done, _, _ = env.step(env.WATER)
#     assert np.array_equal(state, np.array([3]))
#     assert reward == 1
#     assert state == env.MATURE
#     assert not done
#     # Test harvest action
#     state, reward, done, _, _ = env.step(env.HARVEST)
#     assert np.array_equal(state, np.array([0]))
#     assert reward == 10
#     assert state == env.UNPLOWED

