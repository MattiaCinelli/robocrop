from gym.envs.registration import register

register(
    id='RoboCrop-v1',
    entry_point='robocrop.envs:RoboCropEnvV1'
)

register(
    id='RoboCrop-v2',
    entry_point='robocrop.envs:RoboCropEnvV2'
)

register(
    id='RoboCrop-v3',
    entry_point='robocrop.envs:RoboCropEnvV3'
)