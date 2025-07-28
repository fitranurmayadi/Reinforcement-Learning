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
#plane_id = p.loadSDF("stadium.sdf")

# Set gesekan pada plane
plane_static_friction = 1  # Gesekan statis untuk plane
plane_dynamic_friction = 1   # Gesekan dinamis untuk plane

p.changeDynamics(plane_id, -1, lateralFriction=plane_static_friction, spinningFriction=plane_dynamic_friction)

robot_id = p.loadURDF("twsbr_env/envs/urdf/twsbr.urdf")
# Set parameter gesekan untuk setiap roda
wheel_indices = [0, 1]  # Indeks untuk roda, sesuaikan dengan model Anda
static_friction = 1.0  # Gesekan statis
dynamic_friction = 0.5  # Gesekan dinamis

for wheel_index in wheel_indices:
    p.changeDynamics(robot_id, wheel_index, lateralFriction=static_friction, spinningFriction=dynamic_friction)

# --- Fullscreen window and hide default visual elements ---
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 1)
p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 1)
p.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=0, cameraPitch=-30, cameraTargetPosition=[0, 0, 0])  # Adjust camera

# Define motor indices
left_wheel_joint = 0
right_wheel_joint = 1

# Example LQR controller setup
M = 0.478
m_wheel = 0.015
b = 0.1
I_wheel = 0.0013
I_chassis = 0.0013
I_total = I_chassis + I_wheel
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

Q = np.array([[100, 0, 0, 0],
              [0, 100, 0, 0],
              [0, 0, 50, 0],
              [0, 0, 0, 50]])

R = np.array([[0.1]])

P = solve_continuous_are(A, B, Q, R)
K = inv(R).dot(B.T.dot(P))

# Simulation parameters
dt = 1./240.  
t = np.arange(0, 10, dt)
state = np.array([0.1, 0, 0.1, 0])  # initial state [theta, theta_dot, x, x_dot]
states = []

# Function to display robot state in the window
def display_robot_state_in_window(robot_id):
    pos, orn = p.getBasePositionAndOrientation(robot_id)
    linear_velocity, angular_velocity = p.getBaseVelocity(robot_id)

    roll, pitch, yaw = p.getEulerFromQuaternion(orn)
    roll_deg = np.degrees(roll)
    pitch_deg = np.degrees(pitch)
    yaw_deg = np.degrees(yaw)

    x = pos[0]
    y = pos[1]
    z = pos[2]

    x_dot = linear_velocity[0]
    y_dot = linear_velocity[1]
    z_dot = linear_velocity[2]

    # Get joint velocities for the wheels
    left_wheel_velocity = p.getJointState(robot_id, left_wheel_joint)[1]  # Joint velocity
    right_wheel_velocity = p.getJointState(robot_id, right_wheel_joint)[1]  # Joint velocity

    # Calculate linear velocities of the wheels (assuming wheel radius is 0.1 for example)
    wheel_radius = 0.1
    left_linear_velocity = left_wheel_velocity * wheel_radius
    right_linear_velocity = right_wheel_velocity * wheel_radius

    # Prepare text for each attribute
    text_lines = [
        f"Roll: {roll_deg:.2f}°",
        f"Pitch: {pitch_deg:.2f}°",
        f"Yaw: {yaw_deg:.2f}°",
        f"Position (X, Y, Z): ({x:.2f}, {y:.2f}, {z:.2f}) m",
        f"Velocity (X, Y, Z): ({x_dot:.2f}, {y_dot:.2f}, {z_dot:.2f}) m/s",
        f"Left Wheel Velocity: {left_linear_velocity:.2f} m/s",
        f"Right Wheel Velocity: {right_linear_velocity:.2f} m/s",
        f"Left Wheel Direction: {'Forward' if left_wheel_velocity > 0 else 'Backward' if left_wheel_velocity < 0 else 'Stationary'}",
        f"Right Wheel Direction: {'Forward' if right_wheel_velocity > 0 else 'Backward' if right_wheel_velocity < 0 else 'Stationary'}"
    ]
    
    for i, line in enumerate(text_lines):
        p.addUserDebugText(line, [x, y, 0.2 + i * 0.05], lifeTime=0.2, textSize=1, replaceItemUniqueId=-1, textColorRGB=[0, 0, 0])

last_text_update_time = time.time()  # Simpan waktu terakhir teks diperbarui
text_update_interval = 1 / 50  # Interval 50 Hz, atau 0.02 detik
# Main simulation loop
for time_step in t:
    pos, orn = p.getBasePositionAndOrientation(robot_id)
    linear_velocity, angular_velocity = p.getBaseVelocity(robot_id)

    roll, pitch, yaw = p.getEulerFromQuaternion(orn)
    roll = np.degrees(roll)
    pitch = np.degrees(pitch)
    yaw = np.degrees(yaw)

    omega_x = angular_velocity[0]
    omega_y = angular_velocity[1]
    omega_z = angular_velocity[2]

    x = pos[0]
    y = pos[1]
    z = pos[2]

    x_dot = linear_velocity[0]
    y_dot = linear_velocity[1]
    z_dot = linear_velocity[2]

    theta = roll
    theta_dot = omega_x

    state = np.array([theta, theta_dot, x, x_dot])
    states.append(state)

    control_input = -K.dot(state)

    left_velocity = float(control_input)
    right_velocity = float(control_input)

    p.setJointMotorControl2(robot_id, left_wheel_joint, p.VELOCITY_CONTROL, targetVelocity=left_velocity, force=1)
    p.setJointMotorControl2(robot_id, right_wheel_joint, p.VELOCITY_CONTROL, targetVelocity=right_velocity, force=1)
    current_time = time.time()
    if current_time - last_text_update_time >= text_update_interval:
        #display_robot_state_in_window(robot_id)
        last_text_update_time = current_time

    p.stepSimulation()
    time.sleep(dt)

p.disconnect()

states = np.array(states)
plt.figure()
plt.plot(t[:len(states)], states[:, 0], label='Angle (Theta) [deg]')
plt.plot(t[:len(states)], states[:, 2], label='Position (X) [m]')
plt.xlabel('Time [s]')
plt.ylabel('State')
plt.title('Self-Balancing Robot State Over Time')
plt.legend()
plt.show()
