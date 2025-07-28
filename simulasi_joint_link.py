import pybullet as p
import pybullet_data
import time
import tkinter as tk
import numpy as np

# Fungsi untuk mengonversi quaternion ke sudut Euler (roll, pitch, yaw)
def quaternion_to_euler(quaternion):
    w, x, y, z = quaternion
    # Menghitung sudut Euler
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(2.0 * (w * y - z * x))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw

# Fungsi untuk memperbarui tampilan GUI
def update_gui():
    # Get base position and orientation (quaternion)
    base_position, base_orientation = p.getBasePositionAndOrientation(robot)

    # Convert quaternion to Euler angles
    euler_angles = quaternion_to_euler(base_orientation)

    # Extract angular velocity for the robot body
    angular_velocity = p.getBaseVelocity(robot)[1]  # Ambil kecepatan sudut dari robot

    # Extract wheel state
    left_wheel_state = p.getJointState(robot, left_wheel_joint)
    right_wheel_state = p.getJointState(robot, right_wheel_joint)

    # Update label values
    roll, pitch, yaw = euler_angles

    # Convert radians to degrees
    roll_degrees = np.degrees(roll)
    pitch_degrees = np.degrees(pitch)
    yaw_degrees = np.degrees(yaw)

    # Convert angular velocities from rad/s to deg/s
    left_wheel_velocity = left_wheel_state[1]  # Kecepatan sudut roda kiri (rad/s)
    right_wheel_velocity = right_wheel_state[1]  # Kecepatan sudut roda kanan (rad/s)
    
    # Convert to rotations per second
    left_wheel_rotations_per_second = left_wheel_velocity / (2 * np.pi)  # rotasi/s
    right_wheel_rotations_per_second = right_wheel_velocity / (2 * np.pi)  # rotasi/s

    # Convert angular velocity to degrees per second
    angular_velocity_degrees = np.degrees(angular_velocity[2])  # Ambil kecepatan sudut yaw

    # Update text labels
    roll_label.config(text=f"Roll: {roll_degrees:.2f}°")
    pitch_label.config(text=f"Pitch: {pitch_degrees:.2f}°")
    yaw_label.config(text=f"Yaw: {yaw_degrees:.2f}°")
    
    # Update angular velocity of the body robot and wheel velocities
    angular_velocity_label.config(text=f"Body Angular Velocity: {angular_velocity_degrees:.2f}°/s")
    left_wheel_velocity_label.config(text=f"Left Wheel Velocity: {left_wheel_rotations_per_second:.2f} rotations/s")
    right_wheel_velocity_label.config(text=f"Right Wheel Velocity: {right_wheel_rotations_per_second:.2f} rotations/s")

    # Step simulation
    p.stepSimulation()
    time.sleep(1. / 240.)  # Mengatur kecepatan simulasi

    # Panggil fungsi ini lagi setelah 100 ms
    root.after(1, update_gui)

# Start the physics engine
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # Set path for pybullet data
p.setGravity(0, 0, -9.8)
p.loadURDF("plane.urdf")  # Menambahkan tanah

# Load your URDF file
robot = p.loadURDF("urdf/twsbr.urdf")

# Get joint indices for controlling the wheels
left_wheel_joint = 0  # Pastikan ini sesuai dengan indeks joint di URDF Anda
right_wheel_joint = 1  # Pastikan ini sesuai dengan indeks joint di URDF Anda

# Create a tkinter window
root = tk.Tk()
root.title("Robot Angular Position and Velocity")

# Create labels to display roll, pitch, yaw, angular velocity, and wheel velocities
roll_label = tk.Label(root, text="Roll: 0.00°")
roll_label.pack()

pitch_label = tk.Label(root, text="Pitch: 0.00°")
pitch_label.pack()

yaw_label = tk.Label(root, text="Yaw: 0.00°")
yaw_label.pack()

angular_velocity_label = tk.Label(root, text="Body Angular Velocity: 0.00°/s")
angular_velocity_label.pack()

left_wheel_velocity_label = tk.Label(root, text="Left Wheel Velocity: 0.00 rotations/s")
left_wheel_velocity_label.pack()

right_wheel_velocity_label = tk.Label(root, text="Right Wheel Velocity: 0.00 rotations/s")
right_wheel_velocity_label.pack()

# Enable motors and set initial velocities
p.setJointMotorControl2(robot, left_wheel_joint, p.VELOCITY_CONTROL, targetVelocity=50)
p.setJointMotorControl2(robot, right_wheel_joint, p.VELOCITY_CONTROL, targetVelocity=50)

# Start the GUI update loop
update_gui()

# Start the Tkinter main loop
root.mainloop()
