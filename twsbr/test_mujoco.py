import gymnasium as gym
import numpy as np

# Create the MuJoCo environment
env = gym.make("Humanoid-v4", render_mode="human")

# Reset the environment to get the initial observation
obs, info = env.reset()

# Run a few episodes to test the environment
for episode in range(3):
    obs, info = env.reset()
    done = False
    total_reward = 0

    while not done:
        # Sample a random action from the action space
        action = env.action_space.sample()
        
        # Take a step in the environment with the sampled action
        obs, reward, done, truncated, info = env.step(action)
        
        # Render the environment
        env.render()
        
        # Accumulate the reward
        total_reward += reward
        
        if done or truncated:
            break

    print(f"Episode {episode + 1}: Total Reward = {total_reward}")

# Close the environment
env.close()
