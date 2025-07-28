from gymnasium.envs.registration import register

register(
    id="DroneEnv-v0",
    entry_point="DroneEnv.envs:DroneEnv",
)