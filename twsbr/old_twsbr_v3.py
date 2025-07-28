from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import os
import pybullet
import pybullet_data
from pybullet_utils import bullet_client
import os

class TwsbrEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 120}
    def __init__(self,
                urdf_path="urdf/twsbr.urdf",
                render_mode=None,
                action_type="continuous",
                robot_angle_limit = 30,
                max_torque = 1.0,
                max_velocity = 255,
                truncation_steps = 1000
                ):

        super(TwsbrEnv, self).__init__() # It ensures that the initialization code defined in the superclass (gym.Env) is run
        
        self._physics_client_id = -1
        self.robot_angle_limit = robot_angle_limit
        self.truncation_steps = truncation_steps
        self.max_torque = max_torque
        self.max_velocity = max_velocity
        
        self.discrete_torques = np.linspace(-self.max_torque, self.max_torque, num=21) # 21 steps from -100% to +100% with 10% step
        self.discrete_velocity = np.linspace(-self.max_velocity, self.max_velocity, num=21) # 21 steps from -100% to +100% with 10% step
        
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        
        assert action_type in ["discrete", "continuous"]
        self._action_type = action_type

        # State := [roll, pitch, yaw, omega_x, omega_y, omega_z, x, y, z, x_dot, y_dot, z_dot]
        state_limit = np.array([np.pi, np.pi, np.pi, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
        self.observation_space = gym.spaces.Box(low=-np.ones(state_limit.shape), high=np.ones(state_limit.shape))

        # Action space
        if self._action_type == "discrete":
            #Power of action: 2 * (-100, -75, -50, -25, -10, 0, 10, 25, 50, 75, 100 ) percents
            self.action_space = gym.spaces.Discrete(2 * len(self.discrete_torques))

        elif self._action_type == "continuous":
            #Continuous action
            self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,))

        ###--------------------------------------------------------------------------------------------###
        self.load_robot()
        return  

    def load_robot(self):
        #print(self._physics_client_id)
        #Pybullet Env and Actor
        if self._physics_client_id < 0:
            if self.render_mode == "human":
                self._bullet_client = bullet_client.BulletClient(pybullet.GUI, options="--width=1920 --height=1080")
            else:
                self._bullet_client = bullet_client.BulletClient(pybullet.DIRECT)

            self._physics_client_id = self._bullet_client._client
            
            self._bullet_client.resetSimulation()
            self._bullet_client.setGravity(0, 0, -9.8)
            self._bullet_client.setTimeStep(1.0 / self.metadata["render_fps"])

            # Load ground plane
        
            pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())

            self.plane_id = self._bullet_client.loadURDF("plane.urdf")
            self._bullet_client.changeDynamics(self.plane_id, -1, lateralFriction=1.0, spinningFriction=0.9)
        
            # Load robot
            self.urdf_file_name = os.path.join(os.path.abspath(os.path.dirname(__file__)), "urdf", "twsbr.urdf")
            # Set default position and orientation if not specified
            start_position = [0.0, 0.0, 0.0]
            start_orientation = pybullet.getQuaternionFromEuler([0, 0, 0])
            self.robot_id = self._bullet_client.loadURDF(self.urdf_file_name, basePosition=start_position, baseOrientation=start_orientation )
            self._bullet_client.changeDynamics(self.robot_id, 0, lateralFriction=1.0, spinningFriction=0.9)
            self._bullet_client.changeDynamics(self.robot_id, 1, lateralFriction=1.0, spinningFriction=0.9)
            self._bullet_client.setJointMotorControl2(self.robot_id, 0, pybullet.VELOCITY_CONTROL, force=0)
            self._bullet_client.setJointMotorControl2(self.robot_id, 1, pybullet.VELOCITY_CONTROL, force=0)
            
        elif self._physics_client_id >= 0:
            self._bullet_client.removeBody(self.robot_id)
            self._physics_client_id = self._bullet_client._client
            
            self._bullet_client.resetSimulation()
            self._bullet_client.setGravity(0, 0, -9.8)
            self._bullet_client.setTimeStep(1.0 / self.metadata["render_fps"])

            # Load ground plane
        
            pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())

            self.plane_id = self._bullet_client.loadURDF("plane.urdf")
            self._bullet_client.changeDynamics(self.plane_id, -1, lateralFriction=1.0, spinningFriction=0.9)
        
            # Load robot
            self.urdf_file_name = os.path.join(os.path.abspath(os.path.dirname(__file__)), "urdf", "twsbr.urdf")
            # Set default position and orientation if not specified
            start_position = [0.0, 0.0, 0.0]
            start_orientation = pybullet.getQuaternionFromEuler([0, 0, 0])
            self.robot_id = self._bullet_client.loadURDF(self.urdf_file_name, basePosition=start_position, baseOrientation=start_orientation )
            self._bullet_client.changeDynamics(self.robot_id, 0, lateralFriction=1.0, spinningFriction=0.9)
            self._bullet_client.changeDynamics(self.robot_id, 1, lateralFriction=1.0, spinningFriction=0.9)
            self._bullet_client.setJointMotorControl2(self.robot_id, 0, pybullet.VELOCITY_CONTROL, targetVelocity=0, force=0)
            self._bullet_client.setJointMotorControl2(self.robot_id, 1, pybullet.VELOCITY_CONTROL, targetVelocity=0, force=0)



    def reset(self, seed=None, options=None):
        super().reset(seed=seed)  # setting seed
        #self._bullet_client.removeBody(self.robot_id)
        # Turn off default joint velocity control
        self.load_robot()
        self.step_counter = 0
        observation = self._get_obs()
        info = self._get_info()
        #print(self._get_obs)
        return np.array(observation, dtype=np.float32), info


    def step(self, action):
        left_wheel_velocity, right_wheel_velocity = self._apply_action(action)  # Make sure this returns exactly two values
         # Apply torque
        self._bullet_client.setJointMotorControl2(self.robot_id, jointIndex=0,
                                                  controlMode=self._bullet_client.VELOCITY_CONTROL,
                                                  targetVelocity=left_wheel_velocity,
                                                  force=self.max_torque)
        self._bullet_client.setJointMotorControl2(self.robot_id, jointIndex=1,
                                                  controlMode=self._bullet_client.VELOCITY_CONTROL,
                                                  targetVelocity=right_wheel_velocity,
                                                  force=self.max_torque)
        self._bullet_client.stepSimulation()
        self.motor_power = (left_wheel_velocity + right_wheel_velocity)/2
        observation = self._get_obs()
        observation = np.array(observation, dtype=np.float32)
        roll, pitch, yaw, omega_x, omega_y, omega_z, x, y, z, x_dot, y_dot, z_dot = observation

        reward = self._get_reward()

        info = self._get_info()

        self.step_counter += 1
        if self.step_counter >= self.truncation_steps:
            truncated = True
            info["is_success"] = True
        else:
            truncated = False

        robot_angle = abs(self._denormalize_1d(pitch, -180, 180))
        #print(robot_angle)
        if robot_angle > self.robot_angle_limit:
            info["is_success"] = False
            terminated = True
        else:
            terminated = False
            
        return observation, reward, terminated, truncated, info

    def _apply_action(self, action):
        if self._action_type == "discrete":
            # Untuk diskrit, action akan memiliki 42 output nodes:
            # 21 nodes pertama untuk roda kiri, 21 nodes kedua untuk roda kanan
            if len(action) == 42:
                # Ambil bagian kiri (21 node pertama) dan kanan (21 node berikutnya)
                left_wheel_action = action[:21]
                right_wheel_action = action[21:]

                # Temukan indeks dengan nilai aktivasi terbesar untuk roda kiri dan kanan
                left_index = np.argmax(left_wheel_action)
                right_index = np.argmax(right_wheel_action)

                # Pilih nilai torque dari array discrete_torques sesuai indeks
                torque1 = self.discrete_torques[left_index]
                torque2 = self.discrete_torques[right_index]
                return torque1, torque2
                
        elif self._action_type == "continuous":
            # Ensure the action is structured correctly
            #torque1 = np.clip(action[0] * self.max_torque, -self.max_torque, self.max_torque)
            #torque2 = np.clip(action[1] * self.max_torque, -self.max_torque, self.max_torque)
            #return torque1, torque2  # Ensure these are defined correctly
            left_wheel_velocity = np.clip(action[0] * self.max_velocity, -self.max_velocity, self.max_velocity)
            right_wheel_velocity= np.clip(action[1] * self.max_velocity, -self.max_velocity, self.max_velocity)
            return left_wheel_velocity, right_wheel_velocity
        # Ensure you have a return statement if action type doesn't match any case
        #return 0, 0  # Fallback return values
    

    def _get_obs(self):
        pos, orientation = self._bullet_client.getBasePositionAndOrientation(self.robot_id)
        roll, pitch, yaw = pybullet.getEulerFromQuaternion(orientation)
        vel_linear, vel_angular = self._bullet_client.getBaseVelocity(self.robot_id)
        x_dot, y_dot, z_dot = vel_linear
        omega_x, omega_y, omega_z = vel_angular
        
        # Convert radians to degrees for roll, pitch, yaw
        roll = np.degrees(roll)
        pitch = np.degrees(pitch)
        yaw = np.degrees(yaw)
        
        x, y, z = pos
        obs = np.array([roll, pitch, yaw, omega_x, omega_y, omega_z, x, y, z, x_dot, y_dot, z_dot])
        
        # Normalize observations
        return self._normalize_obs(obs)

    def _normalize_obs(self, obs):
        # Define min and max ranges for each parameter in obs
        obs_min = np.array([-180, -180, -180, -10, -10, -10, -10, -10, -10, -10, -10, -10])
        obs_max = np.array([180, 180, 180, 10, 10, 10, 10, 10, 10, 10, 10, 10])
        
        # Normalization formula: (obs - obs_min) / (obs_max - obs_min) * 2 - 1
        normalized_obs = 2 * (obs - obs_min) / (obs_max - obs_min) - 1
        return normalized_obs

    def _denormalize_obs(self, normalized_obs):
        # Define min and max ranges for each parameter in obs
        obs_min = np.array([-180, -180, -180, -10, -10, -10, -10, -10, -10, -10, -10, -10])
        obs_max = np.array([180, 180, 180, 10, 10, 10, 10, 10, 10, 10, 10, 10])
        
        # Denormalization formula: (normalized_obs + 1) / 2 * (obs_max - obs_min) + obs_min
        obs = (normalized_obs + 1) / 2 * (obs_max - obs_min) + obs_min
        return obs

    def _denormalize_1d(self, value, min_value, max_value):
        # Denormalization formula
        denormalized_value = value * (max_value - min_value) / 2 + (max_value + min_value) / 2
        return denormalized_value

    def _get_reward(self):
        
        roll, pitch, yaw, omega_x, omega_y, omega_z, x, y, z, x_dot, y_dot, z_dot = self._get_obs()
        
        reward = 1  # Basic reward for each step
        #if abs pitch angle less than 1 degree, give it reward
        if abs(pitch) < 0.05:
            reward += 1
        # penalize for consume more power
        reward -= (self.motor_power / self.max_velocity) * 10

        # Penalize large angle tilt
        reward -= abs(pitch) * 10

        # Penalize large pitch angular velocity (forward/backward tilt)
        reward -= abs(omega_y) * 10
        
        return reward

    def _get_info(self):
        return {
            "step_count": self.step_counter,
        }

    def render(self):
        if self.render_mode == "human":
            if (self._physics_client_id >= 0):
                # Menyembunyikan elemen visual default
                self._bullet_client.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)
                self._bullet_client.configureDebugVisualizer(pybullet.COV_ENABLE_SHADOWS, 1)
            
                self._render_width = 1920  # Ganti dengan lebar layar Anda
                self._render_height = 1080  # Ganti dengan tinggi layar Anda
                # Reset kamera debug visualizer (opsional)
                self._bullet_client.resetDebugVisualizerCamera(
                    cameraDistance=1,  # Jarak kamera
                    cameraYaw=-45,         # Yaw kamera
                    cameraPitch=-45,      # Pitch kamera
                    cameraTargetPosition=[0, 0, 0]  # Posisi target kamera
                )

                # Render view dan projection matrix
                view_matrix = self._bullet_client.computeViewMatrixFromYawPitchRoll(
                    cameraTargetPosition=[0, 0, 0],
                    distance=1.5,  # Jarak kamera
                    yaw=0,         # Yaw kamera
                    pitch=-30,     # Pitch kamera
                    roll=0,
                    upAxisIndex=2
                )
                proj_matrix = self._bullet_client.computeProjectionMatrixFOV(
                    fov=60,  # Field of View
                    aspect=float(self._render_width) / self._render_height,  # Aspek rasio
                    nearVal=0.1,  # Nilai dekat
                    farVal=100.0  # Nilai jauh
                )

            # Mengambil gambar kamera dari PyBullet
            #(_, _, px, _, _) = self._bullet_client.getCameraImage(
            #    width=self._render_width,
            #    height=self._render_height,
            #    viewMatrix=view_matrix,
            #    projectionMatrix=proj_matrix
            #)

            # Kembalikan gambar dalam format RGB (tanpa alpha channel)
            #return np.array(px)[:, :, :3]
            return None

    def close(self):
        if self._physics_client_id >= 0:
            self._bullet_client.disconnect()
            self._physics_client_id = -1
