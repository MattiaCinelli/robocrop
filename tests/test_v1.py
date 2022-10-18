from contextlib import contextmanager
import os
import pytest

from robocrop.envs.robocrop_v1 import RoboCropEnvV1

# os.chdir(os.path.dirname(__file__))
env = RoboCropEnvV1()

@contextmanager
def does_not_raise():
    '''A context manager that does not raise an exception.'''
    yield

def test_get_reward_steps():
    '''
    Test if the result of reset and step is correct
    '''
    assert env.reset() == 0
    assert env.step(0) == (1, 1, False, {})
    assert env.step(1) == (2, 1, False, {})
    assert env.step(2) == (3, 1, False, {})
    assert env.step(3) == (0, 10, False, {})

def test_get_reward_invalid_action():
    '''
    '''
    assert env.reset() == 0
    with pytest.raises(AssertionError):
        env.step(4)

