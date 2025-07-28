import gymnasium as gym
from twsbr_env.envs import TwsbrEnv  # Import environment
import time

env = TwsbrEnv(
    render_mode="human",              # Pilihan render mode (human atau rgb_array)
    action_type="continuous",         # Pilihan action type (binary, discrete, atau continuous)
    wheels_controlled_together=False, # Apakah kedua roda dikontrol bersama atau secara terpisah
    roll_threshold_deg=30.0,          # Threshold untuk roll (kemiringan)
    x_threshold=10.0,                 # Batas pergerakan pada sumbu X
    y_threshold=10.0,                 # Batas pergerakan pada sumbu Y
)

obs, info = env.reset()
while True:
    time.sleep(1.0/env.metadata["simulation_fps"])
    act = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(action=act)
    print("Observasi:", obs)
    print("Reward:", reward)
    print("Terjadi terminasi:", terminated)
    print("Terjadi pemangkasan:", truncated)
    print("Informasi tambahan:", info)
    if terminated or truncated:
        break
