from gymnasium.envs.registration import register

register(
    id="StarshipLanderEnv-v0",
    entry_point="StarshipLanderEnv.envs:StarshipLanderEnv",
)