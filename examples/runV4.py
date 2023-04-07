# %%
import gym
import numpy as np
# env = gym.make("robocrop.envs:RoboCrop-v4")
from robocrop.envs.robocrop_v4 import RoboCropEnvV4
env = RoboCropEnvV4()
#%%
states = env.observation_space.shape
actions = env.action_space
states, actions

# %%
def simulate_random_actions(env, render=False):
    episodes = 10
    all_rewards = []
    for _ in range(1, episodes):
        state = env.reset() # Restart the agent at the beginning
        done = False # If the agent has completed the level
        score = 0 # Called score not return cause it's python
        while not done:
            random_action = env.action_space.sample() # Do random actions
            # print(random_action)
            _, reward, done, _ = env.step(random_action) 
            score += reward
        all_rewards.append(score)
        print(f"score: {score}")
    env.reset()
    print(f"Mean reward:{np.mean(all_rewards)} Num episodes:{episodes}")

# simulate_random_actions(env)

# %%
import gymnasium as gym
import numpy as np

# from stable_baselines import PPO
from stable_baselines import DQN, A2C
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines.common.evaluation import evaluate_policy

# %%
env = RoboCropEnvV4()
env = DummyVecEnv([lambda: env])

model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10_000)


# %%
from stable_baselines.common.evaluation import evaluate_policy
mean_reward, std_reward = evaluate_policy(model,  env , n_eval_episodes=10)
print(f"mean_reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# %%
from robocrop.envs.robocrop_v4 import RoboCropEnvV4
for rep in range(3):
    print(f"\nRepetition {rep}")
    # env = gym.make("robocrop.envs:RoboCrop-v4")
    env = RoboCropEnvV4()
    model = A2C("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=10_000) # model = PPO2(MlpPolicy, Monitor(env, filename=f'logs/CartPole-v1/PPO2/'), verbose=0).learn(10000)

    # evaluate
    # mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    # print(f"Mean Reward: {mean_reward} (+/-{std_reward})")

    # test and visualize
    action = env.reset()
    rewards = 0
    for i in range(500):
        obs, _states = model.predict(action)
        action, reward, done, info = env.step(obs)
        rewards = rewards + reward
        if done:
            print(f"Failed after {i} steps.")
            break
    print(f"Repetition {rep} reward: {rewards}")