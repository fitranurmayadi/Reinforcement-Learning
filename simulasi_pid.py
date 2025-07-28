# # Introduction

# Description of the implementation of Control Systems namely PID on our robot. 
# The "pidcontrol.py" file has been used for our project. I haven't written this library. 
# I have downloaded it from this GitHub project.[1](#pid)

# #### Importing Libraries

import sys
import pidcontrol as pid
import numpy as np
import pybullet as p
import time
import pybullet_data
import matplotlib.pyplot as plt

# ### PID
# The class below, "SelfBalance" does the main job. 
# In this class, we see functions to tune the PID gains.

class SelfBalance:
    def __init__(self):
        self.xvelMin = -0.01
        self.xvelMax = 0
        self.yMin = -0.01
        self.yMax = -0.001
        self.yPrev = 0
        self.delY = 0
        self.Kp = 100
        self.Ki = 0.001
        self.Kd = 0.1
        self.controller = pid.PID_Controller(self.Kp, self.Ki, self.Kd)
        
    def callback(self, data):
        setPoint = 0
        y = data[1][1] * 180 / np.pi  # Convert radians to degrees
        self.delY = y - self.yPrev
        
        # Update max/min delY
        self.yMax = max(self.yMax, self.delY)
        self.yMin = min(self.yMin, self.delY)
        
        # Calculate control output
        xvel = -self.controller.getCorrection(setPoint, y)
        
        # Limit xvel
        xvel = max(min(xvel, 26), -26)
        
        self.yPrev = y

        # Print PID output for real-time monitoring
        #print(f"Output PID: {xvel}, Angle (degrees): {y}, Kp: {self.Kp}, Ki: {self.Ki}, Kd: {self.Kd}")

        return xvel

    def update_pid_gain(self, Kp=None, Ki=None, Kd=None):
        if Kp is not None:
            self.Kp = Kp
        if Ki is not None:
            self.Ki = Ki
        if Kd is not None:
            self.Kd = Kd
        self.controller = pid.PID_Controller(self.Kp, self.Ki, self.Kd)

def synthesizeData(p, robot):
    position, orientation = p.getBasePositionAndOrientation(robot)
    euler_angles = np.array(p.getEulerFromQuaternion(orientation))  # Get Euler angles
    deg_orien = euler_angles * 180 / np.pi  # Convert radians to degrees
    theta = deg_orien[1]
    velocity, angular = p.getBaseVelocity(robot)
    data = [velocity, euler_angles]
    return data

# Main Function
id = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
plane = p.loadURDF("plane.urdf")
p.setGravity(0, 0, -9.8)

robot = p.loadURDF("urdf/twsbr.urdf", [0, 0, 0.2])

# Joint indices
left_joint = 0
right_joint = 1

maxForce = 0.05
max_rpm = 3000
mode = p.VELOCITY_CONTROL

# Initialize motor controls
p.setJointMotorControl2(robot, left_joint, controlMode=mode, force=maxForce)
p.setJointMotorControl2(robot, right_joint, controlMode=mode, force=maxForce)

# Create SelfBalance object
balance = SelfBalance()

# Data storage for plotting
pid_outputs = []
angles = []
time_stamps = []

# Set up the plot
plt.ion()  # Turn on interactive mode
fig, ax1 = plt.subplots()

# Create two y-axes for output PID and angle
ax2 = ax1.twinx()
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Output PID', color='tab:red')
ax2.set_ylabel('Angle (degrees)', color='tab:blue')

# Main loop
start_time = time.time()
while True:    
    data = synthesizeData(p, robot)
    vel = balance.callback(data)
    
    # Convert velocity to PWM and then to RPM
    pwm_value = vel
    rpm = (pwm_value / 255) * max_rpm  # Calculate RPM from PWM
    rad_per_s = rpm * (2 * np.pi / 60)  # Convert RPM to radians per second
    
    # Set motor control
    p.setJointMotorControl2(robot, left_joint, controlMode=mode, targetVelocity=rad_per_s, force=maxForce)
    p.setJointMotorControl2(robot, right_joint, controlMode=mode, targetVelocity=rad_per_s, force=maxForce)
    
    # Print current angle and velocity
    current_time = time.time() - start_time
    current_angle = data[1][1] * 180 / np.pi
    #print(f"Current Angle: {current_angle}, Velocity: {data[0]}")
    
    # Store data for plotting
    pid_outputs.append(vel)
    angles.append(current_angle)
    time_stamps.append(current_time)
    
    # Update the plot
    ax1.clear()
    ax2.clear()
    
    ax1.plot(time_stamps, pid_outputs, 'r-', label='Output PID')
    ax2.plot(time_stamps, angles, 'b-', label='Angle (degrees)')
    
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Output PID', color='tab:red')
    ax2.set_ylabel('Angle (degrees)', color='tab:blue')
    
    plt.legend()
    plt.draw()
    plt.pause(0.01)  # Pause to allow the plot to update

    # Step the simulation
    p.stepSimulation()
    time.sleep(0.01)

# # Reference
# [1] PyQuadSim [Repository](https://github.com/simondlevy/PyQuadSim/blob/master/pidcontrol.py)
