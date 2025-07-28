import pybullet as p
import pybullet_data
import time

# Connect to PyBullet
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.resetDebugVisualizerCamera(cameraDistance=2, cameraYaw=90, cameraPitch=-45, cameraTargetPosition=[0, 0, 0])
# Set up simulation environment
p.setGravity(0, 0, -9.81)
plane_id = p.loadURDF("plane.urdf")
# Set friction parameters for the plane
plane_static_friction = 1.0
plane_dynamic_friction = 0.9

p.changeDynamics(plane_id, -1,
                 lateralFriction=plane_static_friction,
                 spinningFriction=plane_dynamic_friction)

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
# --- Fullscreen window and hide default visual elements ---

# Enable motor control
p.setJointMotorControl2(robot_id, 0, p.VELOCITY_CONTROL, targetVelocity=10, force=0.05)
p.setJointMotorControl2(robot_id, 1, p.VELOCITY_CONTROL, targetVelocity=10, force=0.05)

# Run simulation
while True:
    p.stepSimulation()
    time.sleep(1./240.)