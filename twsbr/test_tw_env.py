import gymnasium as gym
from twsbr_env.envs.TwEnv import TwEnv

model_path = 'twsbr/twsbr_env/envs/assets/model.xml'
env = TwEnv(model_path)

obs = env.reset()
for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    env.render()
    
    if done:
        break

env.close()
