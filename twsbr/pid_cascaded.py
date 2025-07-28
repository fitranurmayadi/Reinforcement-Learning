import pybullet as p
import pybullet_data
import numpy as np
import time
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# Initialize PyBullet simulation
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)

# Load plane
plane_id = p.loadURDF("plane.urdf")
p.changeDynamics(plane_id, -1, lateralFriction=1.0, spinningFriction=0.0, rollingFriction=0.0)

# Load robot URDF
robot_id = p.loadURDF("twsbr_env/envs/urdf/twsbr.urdf")
# Define the indices of the wheels
wheel_indices = [0, 1]  # Adjust based on your robot's URDF

# Set friction parameters for each wheel (simulating rubber wheels)
static_friction = 1.0  # Lateral static friction for rubber wheels
dynamic_friction = 0.5  # Lateral dynamic friction for rubber wheels
rolling_friction = 0.0  # Rolling friction for the wheels (low value to simulate rolling)

# Apply friction settings to the wheels
for wheel_index in wheel_indices:
    p.changeDynamics(robot_id, wheel_index, 
                     lateralFriction=static_friction, 
                     spinningFriction=dynamic_friction, 
                     rollingFriction=rolling_friction)

# --- Fullscreen window and hide default visual elements ---
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=0, cameraPitch=-30, cameraTargetPosition=[0, 0, 0])  # Adjust camera


# Indices for left and right wheels
left_wheel_joint = 0
right_wheel_joint = 1

# PID controller parameters for balancing
Kp_angle = 50.0
Kd_angle = 1.0
Ki_angle = 0.001

# PID controller parameters for position
Kp_position = 0.0
Kd_position = 0.0
Ki_position = 0.001

# Initialize PID variables
integral_angle = 0.0
prev_error_angle = 0.0
integral_position_x = 0.0
integral_position_y = 0.0
prev_error_position = np.array([0.0, 0.0])

# Simulation parameters
dt = 1. / 240.
t = np.arange(0, 30, dt)

# Target points for path following
target_points = np.array([[0.0, 1.0, 0.0], [0.0, 2.0, 0.0], [1.0, 2.0, 0.0], [2.0, 2.0, 0.0], [2.0, 1.0, 0.0]])
current_target_index = 0
current_target = target_points[current_target_index]

# Path tracking variables
prev_position = np.array([0.0, 0.0])  # To track previous position for adding path points
path_points = []  # List to store all the points along the robot's path

# Tkinter GUI setup
root = tk.Tk()
root.title("Self-Balancing Robot Debug Info")

# Create labels to display PID outputs
angle_label = ttk.Label(root, text="Angle Error: ")
angle_label.grid(row=0, column=0, padx=10, pady=5)
position_label = ttk.Label(root, text="Position Error: ")
position_label.grid(row=1, column=0, padx=10, pady=5)
left_motor_label = ttk.Label(root, text="Left Motor Speed: ")
left_motor_label.grid(row=2, column=0, padx=10, pady=5)
right_motor_label = ttk.Label(root, text="Right Motor Speed: ")
right_motor_label.grid(row=3, column=0, padx=10, pady=5)

# Create Matplotlib figure for real-time plotting
fig, ax = plt.subplots(3, 1, figsize=(5, 4))
ax[0].set_title("Angle (Theta)")
ax[1].set_title("Position X and Y")
ax[2].set_title("Motor Velocities")

# Initialize plots
theta_plot, = ax[0].plot([], [], label="Theta")
position_x_plot, = ax[1].plot([], [], label="X")
position_y_plot, = ax[1].plot([], [], label="Y")
left_motor_plot, = ax[2].plot([], [], label="Left Motor")
right_motor_plot, = ax[2].plot([], [], label="Right Motor")

# Add canvas to Tkinter
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().grid(row=0, column=1, rowspan=4)

# Function to update plots in real-time
def update_plot(states, time_data):
    theta_plot.set_data(time_data, states[:, 0])
    position_x_plot.set_data(time_data, states[:, 1])
    position_y_plot.set_data(time_data, states[:, 2])
    left_motor_plot.set_data(time_data, states[:, 3])
    right_motor_plot.set_data(time_data, states[:, 4])
    
    ax[0].relim()
    ax[0].autoscale_view()
    ax[1].relim()
    ax[1].autoscale_view()
    ax[2].relim()
    ax[2].autoscale_view()
    canvas.draw()

# Real-time simulation loop
states = []
time_data = []

def simulation_step():
    global current_target_index, current_target, integral_angle, prev_error_angle
    global integral_position_x, integral_position_y, prev_error_position, prev_position

    pos, orn = p.getBasePositionAndOrientation(robot_id)
    linear_velocity, angular_velocity = p.getBaseVelocity(robot_id)

    roll, pitch, yaw = p.getEulerFromQuaternion(orn)
    theta = roll
    error_angle = 0 - theta  # Target is 0 degrees (upright position)

    # PID for angle control
    integral_angle += error_angle * dt
    derivative_angle = (error_angle - prev_error_angle) / dt
    control_input_angle = Kp_angle * error_angle + Ki_angle * integral_angle + Kd_angle * derivative_angle

    # Position control
    error_position = current_target[:2] - np.array(pos[:2])  # Only considering x and y
    integral_position_x += error_position[0] * dt
    integral_position_y += error_position[1] * dt
    derivative_position = error_position - prev_error_position

    control_input_position = (
        Kp_position * error_position +
        Ki_position * np.array([integral_position_x, integral_position_y]) +
        Kd_position * derivative_position
    )

    # Combine angle and position control
    left_motor_control = (-control_input_angle + control_input_position[0])  # For left wheel
    right_motor_control = (-control_input_angle - control_input_position[1])  # For right wheel

    # Apply control inputs to the wheels
    p.setJointMotorControl2(robot_id, left_wheel_joint, p.VELOCITY_CONTROL, targetVelocity=left_motor_control)
    p.setJointMotorControl2(robot_id, right_wheel_joint, p.VELOCITY_CONTROL, targetVelocity=right_motor_control)

    # Get motor velocities
    left_motor_speed = p.getJointState(robot_id, left_wheel_joint)[1]
    right_motor_speed = p.getJointState(robot_id, right_wheel_joint)[1]

    # Store state for plotting
    states.append([theta, pos[0], pos[1], left_motor_speed, right_motor_speed])
    time_data.append(time.time() - t[0])

    # Update Tkinter labels
    angle_label.config(text=f"Angle Error: {error_angle:.3f}")
    position_label.config(text=f"Position Error: X={error_position[0]:.3f}, Y={error_position[1]:.3f}")
    left_motor_label.config(text=f"Left Motor Speed: {left_motor_speed:.3f}")
    right_motor_label.config(text=f"Right Motor Speed: {right_motor_speed:.3f}")

    # Add path points (red dots) every 1 cm traveled
    distance_traveled = np.linalg.norm(np.array(pos[:2]) - prev_position)
    if distance_traveled >= 0.01:  # Add point if robot has moved at least 1 cm
        p.addUserDebugLine(prev_position.tolist() + [0], np.array(pos[:2]).tolist() + [0], [1, 0, 0], 1, lifeTime=0)
        prev_position = np.array(pos[:2])

    # Update plots every few steps
    if len(states) % 10 == 0:
        update_plot(np.array(states), np.array(time_data))

    # Switch to the next target if close enough to the current one
    if np.linalg.norm(error_position) < 0.1:  # Threshold to switch targets
        current_target_index += 1
        if current_target_index < len(target_points):
            current_target = target_points[current_target_index]

    prev_error_angle = error_angle
    prev_error_position = error_position

    # Step the simulation
    p.stepSimulation()

    # Schedule the next simulation step
    root.after(10, simulation_step)  # Update every 10 ms

# Start the simulation loop
simulation_step()

# Start Tkinter main loop
root.mainloop()

# Disconnect from PyBullet
p.disconnect()
