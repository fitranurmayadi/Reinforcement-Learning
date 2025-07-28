import gymnasium as gym
from twsbr_env.envs import TwsbrEnv  # Import environment
import numpy as np
import time
import matplotlib.pyplot as plt

env = gym.make("TwsbrEnv-v0", render_mode="human")
obs, info = env.reset()

# PID controller parameters for balance, steering, and distance
Kp_balance = 25.0
Kd_balance = 0.1
Ki_balance = 0.001

Kp_steer = 0.1
Kd_steer = 0.01
Ki_steer = 0.001

Kp_distance = 0.1
Kd_distance = 0.01
Ki_distance = 0.001

# Variables for PID calculations
integral_balance = 0.0
prev_error_balance = 0.0

integral_steer = 0.0
prev_error_steer = 0.0

integral_distance = 0.0
prev_error_distance = 0.0

# Define target positions
target_positions = [(1, 0, 0), (2, 0, 0), (2, 1, 0), (2, 2, 0)]
current_target_index = 0

# Simulation parameters
dt = 1. / 240.
t = np.arange(0, 30, dt)
states = []

# Functions for PID calculations
def pid_distance(current_position, target_position):
    global integral_distance, prev_error_distance
    error = np.linalg.norm(np.array(target_position[:2]) - np.array(current_position[:2]))  # Distance error in x-y plane
    integral_distance += error * dt
    derivative = (error - prev_error_distance) / dt
    prev_error_distance = error
    return Kp_distance * error + Ki_distance * integral_distance + Kd_distance * derivative

def pid_balance(current_angle, target_angle=0.0):
    global integral_balance, prev_error_balance
    error = target_angle - current_angle
    integral_balance += error * dt
    derivative = (error - prev_error_balance) / dt
    prev_error_balance = error
    return Kp_balance * error + Ki_balance * integral_balance + Kd_balance * derivative

def pid_steer(current_angle, target_angle):
    global integral_steer, prev_error_steer
    error = target_angle - current_angle
    integral_steer += error * dt
    derivative = (error - prev_error_steer) / dt
    prev_error_steer = error
    return Kp_steer * error + Ki_steer * integral_steer + Kd_steer * derivative

# Function to check if the robot has reached the target
def has_reached_target(current_position, target_position, threshold=0.1):
    distance = np.linalg.norm(np.array(target_position[:2]) - np.array(current_position[:2]))
    return distance < threshold

for time_step in t:
    # Get the current target position
    target_position = target_positions[current_target_index]
    
    # Calculate distance and steering actions
    distance_action = pid_distance(obs[:2], target_position)
    steering_action = pid_steer(obs[5], np.arctan2(target_position[1] - obs[1], target_position[0] - obs[0]))
    
    # Calculate balance action to keep the robot upright
    balance_action = pid_balance(obs[13], 0.0)

    # Combine actions
    left_wheel_action = balance_action + distance_action - steering_action
    right_wheel_action = balance_action + distance_action + steering_action
    action = np.array([left_wheel_action, right_wheel_action])
    
    # Perform action in environment
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    
    # Check if the robot has reached the current target position
    if has_reached_target(obs[:2], target_position):
        current_target_index = (current_target_index + 1) % len(target_positions)  # Move to next target

    # Check for termination or truncation
    if terminated or truncated:
       obs, info = env.reset()

env.close()
