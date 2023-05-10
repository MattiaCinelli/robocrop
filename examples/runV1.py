#%%
import os
os.chdir(os.path.dirname(__file__))

import numpy as np
from icecream import ic

import gymnasium as gym
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam 


from robocrop.logs import logging
logger = logging.getLogger(__name__)
# %%
env = gym.make("robocrop.envs:RoboCrop-v1")

# %%
# Create a Deep Learning Model with Keras
def build_model(states, actions):
    model = Sequential()    
    model.add(Dense(24, activation='relu', input_shape=states))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    model.add(Flatten())
    return model

env = gym.make("robocrop.envs:RoboCrop-v1")
states = env.observation_space.shape
actions = env.action_space.n
# logger.info(states, actions)

# %%
model = build_model(states, actions)
model.summary()
# %%
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
# %%

def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    ic(policy)
    memory = SequentialMemory(limit=1_000, window_length=1)
    ic(memory)
    return DQNAgent(
        model=model, memory=memory, policy=policy, 
        nb_actions=actions, nb_steps_warmup=10, 
        target_model_update=1e-2)


# %%
del model 
model = build_model(states, actions)
dqn = build_agent(model, actions)
# ic(dqn)
# """
dqn.compile(Adam(lr=1e-3), metrics=['mae'])
dqn.fit(env, nb_steps=1_000, visualize=False, verbose=1)
# %%

scores = dqn.test(env, nb_episodes=10, visualize=False)
print(np.mean(scores.history['episode_reward']))
# %%
_ = dqn.test(env, nb_episodes=10, visualize=True)
# %%
