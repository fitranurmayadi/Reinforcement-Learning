from gymnasium.envs.registration import register

#register(
#    id="twsbr-v0",
#    entry_point="twsbr_env.envs:TwsbrEnv",
#)

register(
    id="TwsbrEnv-v0",
    entry_point="twsbr_env.envs:TwsbrEnv",
)

register(
    id="TwsbrEnvV2-v0",
    entry_point="twsbr_env.envs:TwsbrEnvV2",
)