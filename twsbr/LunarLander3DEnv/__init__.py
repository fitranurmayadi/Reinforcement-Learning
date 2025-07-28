from gymnasium.envs.registration import register

register(
    id="LunarLander3DEnv-v0",
    entry_point="LunarLander3DEnv.envs:LunarLander3DEnv",
)