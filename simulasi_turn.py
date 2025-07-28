import pybullet as p
import pybullet_data
import time

# Start the physics engine
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # Set path for pybullet data
p.setGravity(0, 0, -9.8)
p.loadURDF("plane.urdf")  # Menambahkan tanah

# Load your URDF file
robot = p.loadURDF("urdf/twsbr.urdf")
# Get the number of joints
num_joints = p.getNumJoints(robot)
print(f"Number of joints in the robot: {num_joints}")

# Print joint info to find the correct indices
for i in range(num_joints):
    joint_info = p.getJointInfo(robot, i)
    print(f"Joint {i}: {joint_info[1]}")

# After determining the correct joint indices from the printed info
left_wheel_joint = 0  # Set based on the correct index for the left wheel
right_wheel_joint = 1  # Set based on the correct index for the right wheel

while True:
    # Apply motor control on both wheels
    p.setJointMotorControl2(robot, left_wheel_joint, p.VELOCITY_CONTROL, targetVelocity=1)
    p.setJointMotorControl2(robot, right_wheel_joint, p.VELOCITY_CONTROL, targetVelocity=1)

    # Step simulation
    p.stepSimulation()
    
    # Get the position and orientation of the base (chassis)
    base_pos, base_ori = p.getBasePositionAndOrientation(robot)
    print("Chassis Position:", base_pos)
    print("Chassis Orientation:", base_ori)

    # Get joint positions (left and right wheels)
    left_wheel_state = p.getJointState(robot, left_wheel_joint)
    right_wheel_state = p.getJointState(robot, right_wheel_joint)
    
    print("Left Wheel Position:", p.getLinkState(robot, left_wheel_joint)[0])
    print("Right Wheel Position:", p.getLinkState(robot, right_wheel_joint)[0])

    time.sleep(1./240.)
