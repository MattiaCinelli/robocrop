# robocrop

A gym-like environment to simulate robot planting and harvesting crop and to test different RL algorithms.

```bash
pip install -e .
```

## Version 1
<img src="commons\RobocropV1.svg" width=250 height=256 align='right'>
To succeed the algorithm has to 
Plow -> Seed -> Water -> Harvest

```python
import gym
import numpy as np
# from stable_baselines import PPO
from stable_baselines import DQN
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import MlpPolicy

env = gym.make("robocrop.envs:RoboCrop-v1")
model = DQN(MlpPolicy, env, verbose=0)

model.learn(total_timesteps=100_000)


```

<!-- 
## Version 1.1
To succeed the algorithm has to 
Plow -> Seed -> Water -> Water -> Harvest
## Version 2
Observation:
Between 0 and 3: Empty, Seeded, Small plant, Ready for harvest
Soil moisture: Between 0 - 100. 100 for each watering, decrease 25 point every step

Observation:
Between 0 and 3: Empty, Seeded, Small plant, Ready for harvest
Soil moisture: Between 0 - 100. 100 for each watering, decrease 25 point every step -->