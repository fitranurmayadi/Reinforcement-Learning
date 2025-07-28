import pybullet as p
import pybullet_data
import numpy as np
from scipy.linalg import solve_continuous_are, inv
import time
import matplotlib.pyplot as plt

# Connect to PyBullet and setup environment
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)
plane_id = p.loadURDF("plane.urdf")
# Set gesekan pada plane
plane_static_friction = 5.0  # Gesekan statis untuk plane
plane_dynamic_friction = 5.0   # Gesekan dinamis untuk plane

p.changeDynamics(plane_id, -1, lateralFriction=plane_static_friction, spinningFriction=plane_dynamic_friction)

robot_id = p.loadURDF("twsbr_env/envs/urdf/twsbr.urdf")
# Set parameter gesekan untuk setiap roda
wheel_indices = [0, 1]  # Indeks untuk roda, sesuaikan dengan model Anda
static_friction = 5.0  # Gesekan statis
dynamic_friction = 5.0  # Gesekan dinamis

for wheel_index in wheel_indices:
    p.changeDynamics(robot_id, wheel_index, lateralFriction=static_friction, spinningFriction=dynamic_friction)

# --- Fullscreen window and hide default visual elements ---
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)  # Disable default GUI
p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)  # Disable RGB buffer preview
p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)  # Disable depth buffer preview
p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)  # Disable segmentation preview
p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)  # Enable shadows (optional)
p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0, cameraPitch=-30, cameraTargetPosition=[0, 0, 0])  # Adjust camera

# Define motor indices
left_wheel_joint = 0
right_wheel_joint = 1

# Example LQR controller setup (same as before)
M = 0.528
m_motor = 0.110
m_wheel = 0.015
b = 0.1
I_motor = 2.75e-5
I_wheel = 2.025e-6
I_chassis = 7.5e-3
I_total = I_chassis + I_wheel + I_motor
g = 9.8
l = 0.04

p_value = I_total * (M + 2 * m_wheel) + M * 2 * m_wheel * l**2

A = np.array([[0, 1, 0, 0],
              [0, -(I_total + 2 * l**2 * m_wheel) * b / p_value, (2 * m_wheel**2 * g * l**2) / p_value, 0],
              [0, 0, 0, 1],
              [0, -(2 * m_wheel * l * b) / p_value, 2 * m_wheel * g * l * (M + 2 * m_wheel) / p_value, 0]])

B = np.array([[0],
              [(I_total + 2 * l**2 * m_wheel) / p_value],
              [0],
              [2 * m_wheel * l / p_value]])

Q = np.array([[10, 0, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 10, 0],
              [0, 0, 0, 1]])

R = np.array([[1]])

P = solve_continuous_are(A, B, Q, R)
K = inv(R).dot(B.T.dot(P))

# Simulation parameters
dt = 1./240.  
t = np.arange(0, 10, dt)
state = np.array([0.1, 0, 0.1, 0])  # initial state [theta, theta_dot, x, x_dot]
states = []

# Simulate the system
for time_step in t:
    # Get the current state from PyBullet
    pos, orn = p.getBasePositionAndOrientation(robot_id)
    linear_velocity, angular_velocity = p.getBaseVelocity(robot_id)

    # Assume orientation in 2D (pitch angle, theta)
    theta = p.getEulerFromQuaternion(orn)[2]  # Pitch angle
    theta_dot = angular_velocity[2]           # Pitch angular velocity
    x = pos[0]                                # x position
    x_dot = linear_velocity[0]                # x velocity
    
    # Update state
    state = np.array([theta, theta_dot, x, x_dot])
    states.append(state)
    
    # Calculate control action using LQR
    control_input = -K.dot(state)

    # Set kecepatan roda dengan mode VELOCITY_CONTROL
    left_velocity = control_input  # Kecepatan roda kiri dalam radian/s
    right_velocity = control_input  # Kecepatan roda kanan dalam radian/s

    # Apply the control to the motors
    p.setJointMotorControl2(robot_id, left_wheel_joint, p.VELOCITY_CONTROL, targetVelocity=left_velocity)
    p.setJointMotorControl2(robot_id, right_wheel_joint, p.VELOCITY_CONTROL, targetVelocity=right_velocity)

    

    # Step the simulation
    p.stepSimulation()
    time.sleep(dt)  # Keep real-time pace

p.disconnect()

# Plotting results
states = np.array(states)
plt.figure()
plt.plot(t[:len(states)], states[:, 0], label='Angle (Theta) [deg]')
plt.plot(t[:len(states)], states[:, 2], label='Position (X) [m]')
plt.xlabel('Time [s]')
plt.ylabel('State')
plt.title('Self-Balancing Robot State Over Time')
plt.legend()
plt.show()