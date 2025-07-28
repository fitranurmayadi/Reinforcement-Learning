import pybullet as p
import pybullet_data
import numpy as np
import time
import matplotlib.pyplot as plt

# Connect to PyBullet and setup environment
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)

# Load plane and set up surface friction
plane_id = p.loadURDF("plane.urdf")
# Set friction parameters for the plane
plane_static_friction = 1.0
plane_dynamic_friction = 0.9

p.changeDynamics(plane_id, -1,lateralFriction=plane_static_friction,spinningFriction=plane_dynamic_friction)

# Load the robot
robot_id = p.loadURDF("twsbr_env/envs/urdf/twsbr.urdf")
# Define the indices of the wheels
wheel_indices = [0, 1]

# Set friction parameters for each wheel (simulating rubber wheels)
static_friction = 1.0
dynamic_friction = 0.9

for wheel_index in wheel_indices:
    p.changeDynamics(robot_id, wheel_index,
                     lateralFriction=static_friction,
                     spinningFriction=dynamic_friction)
# Fullscreen window and hide default visual elements
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=45, cameraPitch=-30, cameraTargetPosition=[0, 0, 0])

# Define motor indices
left_wheel_joint = 0
right_wheel_joint = 1

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

# Display function for robot state
def display_robot_state_in_window(roll_deg, pitch_deg, yaw_deg, x, y, z, x_dot, y_dot, z_dot, left_linear_velocity, right_linear_velocity):
    text_lines = [
        f"Roll: {roll_deg:.2f}°",
        f"Pitch: {pitch_deg:.2f}°",
        f"Yaw: {yaw_deg:.2f}°",
        f"Position (X, Y, Z): ({x:.2f}, {y:.2f}, {z:.2f}) m",
        f"Velocity (X, Y, Z): ({x_dot:.2f}, {y_dot:.2f}, {z_dot:.2f}) m/s",
        f"Left Wheel Velocity: {left_linear_velocity:.2f} m/s",
        f"Right Wheel Velocity: {right_linear_velocity:.2f} m/s",
    ]
    
    for i, line in enumerate(text_lines):
        p.addUserDebugText(line, [x, y, 0.2 + i * 0.05], lifeTime=0.2, textSize=1, textColorRGB=[0, 0, 0])

# Main simulation loop
for time_step in t:
    pos, orn = p.getBasePositionAndOrientation(robot_id)
    linear_velocity, angular_velocity = p.getBaseVelocity(robot_id)

    roll, pitch, yaw = p.getEulerFromQuaternion(orn)
    roll_deg = np.degrees(roll)
    pitch_deg = np.degrees(pitch)
    yaw_deg = np.degrees(yaw)
    
    x, y, z = pos
    x_dot, y_dot, z_dot = linear_velocity

    # Check if the robot has reached the current target position
    target_position = target_positions[current_target_index]
    distance_to_target = np.linalg.norm(np.array(target_position[:2]) - np.array([x, y]))
    
    if distance_to_target < 0.1:  # Move to next target if within threshold
        current_target_index = (current_target_index + 1) % len(target_positions)
        target_position = target_positions[current_target_index]

    # Calculate desired distance and steering adjustments
    distance_output = pid_distance((x, y, z), target_position)
    angle_to_target = np.arctan2(target_position[1] - y, target_position[0] - x)
    steer_output = pid_steer(yaw_deg, np.degrees(angle_to_target))

    # Balancing PID
    balance_output = pid_balance(pitch_deg, 0.0)

    # Set wheel velocities
    left_wheel_velocity = balance_output + steer_output
    right_wheel_velocity = balance_output - steer_output

    p.setJointMotorControl2(robot_id, left_wheel_joint, p.VELOCITY_CONTROL, targetVelocity=-left_wheel_velocity)
    p.setJointMotorControl2(robot_id, right_wheel_joint, p.VELOCITY_CONTROL, targetVelocity=-right_wheel_velocity)

    left_wheel_velocity = p.getJointState(robot_id, left_wheel_joint)[1]
    right_wheel_velocity = p.getJointState(robot_id, right_wheel_joint)[1]
    
    wheel_radius = 0.045
    left_linear_velocity = left_wheel_velocity * wheel_radius
    right_linear_velocity = right_wheel_velocity * wheel_radius

    display_robot_state_in_window(roll_deg, pitch_deg, yaw_deg, x, y, z, x_dot, y_dot, z_dot, left_linear_velocity, right_linear_velocity)

    states.append(np.array([pitch_deg, x, y, x_dot, y_dot]))
    p.stepSimulation()
    time.sleep(dt)

p.disconnect()

# Plotting
states = np.array(states)
plt.figure()
plt.plot(t[:len(states)], states[:, 1], label='Position X [m]')
plt.plot(t[:len(states)], states[:, 2], label='Position Y [m]')
plt.xlabel('Time [s]')
plt.ylabel('Position')
plt.title('Self-Balancing Robot Path Following')
plt.legend()
plt.show()
