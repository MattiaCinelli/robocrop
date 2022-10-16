from gym.envs.registration import register

register(
    id='RoboCrop-v1',
    entry_point='robocrop.envs:RoboCropEnvV1'
)