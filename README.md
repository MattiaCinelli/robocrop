# robocrop

A gym-like environment to simulate robot planting and harvesting crop and to test different RL algorithms.

Clone and install the package
```bash
pip install -e .
```


## Version 1
<img src="commons\RobocropV1.svg" width=300 height=256 align='right'>
To succeed the algorithm has to 
Plow -> Seed -> Water -> Harvest

Try this code to test it:

```python
import gym
import numpy as np
from stable_baselines import DQN
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import MlpPolicy

for rep in range(3):
    print(f"\nRepetition {rep}")
    env = gym.make("robocrop.envs:RoboCrop-v1")
    model = DQN(MlpPolicy, env, verbose=0)
    model.learn(total_timesteps=1_000)

    # evaluate
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=200)
    print(f"Eval reward: {mean_reward} (+/-{std_reward})")

    # test and visualize
    action = env.reset()
    for i in range(500):
        action, rewards, done, info = env.step(action)
        if done:
            print(f"Failed after {i} steps.")
            break
```



## Version 1.1
<img src="commons\RobocropV1.1.svg" width=550 height=256 align='right'>
To succeed the algorithm has to 
Plow -> Seed -> Water -> Water -> Harvest
<!-- 


## Version 2
Observation:
Between 0 and 3: Empty, Seeded, Small plant, Ready for harvest
Soil moisture: Between 0 - 100. 100 for each watering, decrease 25 point every step

Observation:
Between 0 and 3: Empty, Seeded, Small plant, Ready for harvest
Soil moisture: Between 0 - 100. 100 for each watering, decrease 25 point every step -->