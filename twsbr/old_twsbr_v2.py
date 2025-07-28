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
    # Membuat environment
    parameters = {
        "urdf_file_name": "urdf/twsbr.urdf",
        "binary_action": {
            "torque_magnitude": 10.0
        },
        "discrete_action": {
            "torque_magnitudes": [5.0, 10.0, 15.0]
        },
        "continuous_action": {
            "max_torque_magnitude": 10.0
        },
        "reward": {
            "tilt_speed_penalty_scale": 0.1
        },
        "truncation_steps": 1000
    }

    def __init__(self,
                 render_mode=None,
                 action_type="continuous",
                 wheels_controlled_together=True,
                 pitch_threshold_deg=22.5,
                 x_threshold=1.0,
                 y_threshold=1.0,
                 parameters=parameters):

        if parameters is None:
            raise ValueError("Parameters dictionary must be provided")

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        assert action_type in ["binary", "discrete", "continuous"]
        self._action_type = action_type
        self._wheel_controlled_together = wheels_controlled_together

        self._render_height = 1080
        self._render_width = 1920
        self._physics_client_id = -1

        self.pitch_threshold_rad = pitch_threshold_deg / 180.0 * np.pi
        self.x_threshold = x_threshold
        self.y_threshold = y_threshold
        self.this_file_dir_path = os.path.abspath(os.path.dirname(__file__))
        self.urdf_file_name = parameters["urdf_file_name"]

        # State := [roll, pitch, yaw, omega_x, omega_y, omega_z, x, y, z, x_dot, y_dot, z_dot]
        state_limit = np.array([np.pi, np.pi, np.pi, 10.0, 10.0, 10.0, self.x_threshold, self.y_threshold, 1.0, 10.0, 10.0, 1.0])
        self.observation_space = gym.spaces.Box(low=-np.ones(state_limit.shape), high=np.ones(state_limit.shape))

        # Action space
        if self._action_type == "binary":
            self.binary_action_torque_magnitude = parameters["binary_action"]["torque_magnitude"]
            if self._wheel_controlled_together:
                self.action_space = gym.spaces.Discrete(2)
            else:
                self.action_space = gym.spaces.Discrete(4)

        elif self._action_type == "discrete":
            self.discrete_action_torque_magnitudes = parameters["discrete_action"]["torque_magnitudes"]
            if self._wheel_controlled_together:
                self.action_space = gym.spaces.Discrete(len(self.discrete_action_torque_magnitudes))
            else:
                self.action_space = gym.spaces.Discrete(2 * len(self.discrete_action_torque_magnitudes))

        elif self._action_type == "continuous":
            self.continuous_action_max_torque_magnitude = parameters["continuous_action"]["max_torque_magnitude"]
            if self._wheel_controlled_together:
                self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,))
            else:
                self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,))

    def reset(self,
          seed=None,
          options=None,
          start_position=None,
          start_tilt_deg=None,
          start_yaw_deg=0.0):
        super().reset(seed=seed)  # setting seed
    
        urdf_file_name = os.path.join(os.path.abspath(os.path.dirname(__file__)), "urdf", "twsbr.urdf")
        
        # Setup PyBullet client if not done yet
        if self._physics_client_id < 0:
            if self.render_mode == "human":
                self._bullet_client = bullet_client.BulletClient(connection_mode=pybullet.GUI)
            else:
                self._bullet_client = bullet_client.BulletClient()

            self._physics_client_id = self._bullet_client._client
            self._bullet_client.resetSimulation()
            self._bullet_client.setGravity(0, 0, -9.8)
            self._bullet_client.setTimeStep(1.0 / self.metadata["render_fps"])
        
            # Load ground plane
            pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
            self.ground_plane = self._bullet_client.loadURDF("plane.urdf")
            self._bullet_client.changeDynamics(self.ground_plane, -1, lateralFriction=1.0, spinningFriction=0.9)
            self.twsbr = self._bullet_client.loadURDF(urdf_file_name)
        
        # Set default position and orientation if not specified
        start_position = start_position or [0.0, 0.0, 0.0]
        start_orientation = pybullet.getQuaternionFromEuler([0, 0, start_yaw_deg])
        
        # Remove the previous robot if it exists
        self._bullet_client.removeBody(self.twsbr)
        # Load robot
        self.twsbr = self._bullet_client.loadURDF(urdf_file_name, basePosition=start_position, baseOrientation=start_orientation)
        self._bullet_client.changeDynamics(self.twsbr, 0, lateralFriction=1.0, spinningFriction=0.9)
        self._bullet_client.changeDynamics(self.twsbr, 1, lateralFriction=1.0, spinningFriction=0.9)

        # Turn off default joint velocity control
        self._bullet_client.setJointMotorControl2(self.twsbr, 0, pybullet.VELOCITY_CONTROL, force=0)
        self._bullet_client.setJointMotorControl2(self.twsbr, 1, pybullet.VELOCITY_CONTROL, force=0)

        self.step_counter = 0
        observation = self._get_obs()
        info = self._get_info()
        return np.array(observation, dtype=np.float32), info


    def step(self, action):
        torque1, torque2 = self._apply_action(action)  # Make sure this returns exactly two values


        #Proportional Multiplication for the action
        proportional_power = 1
        torque1 = torque1 * proportional_power
        torque2 = torque2 * proportional_power
         # Apply torque
        self._bullet_client.setJointMotorControl2(self.twsbr, jointIndex=0,
                                                  controlMode=self._bullet_client.TORQUE_CONTROL,
                                                  force=torque1)
        self._bullet_client.setJointMotorControl2(self.twsbr, jointIndex=1,
                                                  controlMode=self._bullet_client.TORQUE_CONTROL,
                                                  force=torque2)
        self._bullet_client.stepSimulation()

        observation = self._get_obs()
        observation = np.array(observation, dtype=np.float32)
        roll, pitch, yaw, omega_x, omega_y, omega_z, x, y, z, x_dot, y_dot, z_dot = observation

        reward = self._get_reward()

        info = self._get_info()

        self.step_counter += 1
        if self.step_counter >= self.parameters["truncation_steps"]:
            truncated = True
            info["is_success"] = True
        else:
            truncated = False

        robot_angle = abs(self._denormalize_1d(pitch, -180, 180))
        #print(robot_angle)
        if robot_angle > self.pitch_threshold_rad:
            info["is_success"] = False
            terminated = True
        else:
            terminated = False
            
        return observation, reward, terminated, truncated, info

    def _apply_action(self, action):
        action = self._denormalize_action(action)
        if self._action_type == "binary":
            if self._wheel_controlled_together:
                torque = self.binary_action_torque_magnitude if action == 1 else -self.binary_action_torque_magnitude
                return torque, torque  # Return both torques for both wheels
            else:
                # Unpacking for independent control
                # Here, make sure to return exactly two values
                # Example: control for both wheels independently
                # This needs to be adjusted based on your actual logic
                if action == 0:
                    torque1 = self.binary_action_torque_magnitude
                    torque2 = 0.0
                elif action == 1:
                    torque1 = self.binary_action_torque_magnitude
                    torque2 = 0.0
                elif action == 2:
                    torque1 = 0.0
                    torque2 = self.binary_action_torque_magnitude
                elif action == 3:
                    torque1 = 0.0
                    torque2 = self.binary_action_torque_magnitude
                return torque1, torque2  # Ensure these are defined correctly

        elif self._action_type == "discrete":
            if self._wheel_controlled_together:
                torque = self.discrete_action_torque_magnitudes[action]
                return torque, torque  # Return both torques for both wheels
            else:
                # Logic for independent control
                # Again, ensure exactly two values are returned
                if action < len(self.discrete_action_torque_magnitudes):
                    torque1 = self.discrete_action_torque_magnitudes[action]
                    torque2 = 0.0
                else:
                    torque1 = 0.0
                    torque2 = self.discrete_action_torque_magnitudes[action - len(self.discrete_action_torque_magnitudes)]
                return torque1, torque2  # Ensure these are defined correctly

        elif self._action_type == "continuous":
            if self._wheel_controlled_together:
                torque = np.clip(action, -self.continuous_action_max_torque_magnitude, self.continuous_action_max_torque_magnitude)
                return torque, torque  # Return both torques for both wheels
            else:
                # Ensure the action is structured correctly
                torque1 = np.clip(action[0], self.continuous_action_max_torque_magnitude, self.continuous_action_max_torque_magnitude)
                torque2 = np.clip(action[1], -self.continuous_action_max_torque_magnitude, self.continuous_action_max_torque_magnitude)
                return torque1, torque2  # Ensure these are defined correctly

        # Ensure you have a return statement if action type doesn't match any case
        return 0, 0  # Fallback return values
    
    def _denormalize_action(self, action):
        if self._action_type == "continuous":
            if self._wheel_controlled_together:
                return action[0] * self.continuous_action_max_torque_magnitude
            else:
                return action * self.continuous_action_max_torque_magnitude
        return action

    def _binary_wheel_torque(self, action):
        if action == 0:
            return self.binary_action_torque_magnitude, 0.0
        elif action == 1:
            return -self.binary_action_torque_magnitude, 0.0
        elif action == 2:
            return 0.0, self.binary_action_torque_magnitude
        elif action == 3:
            return 0.0, -self.binary_action_torque_magnitude

    def _discrete_wheel_torque(self, action):
        if self._wheel_controlled_together:
            return self.discrete_action_torque_magnitudes[action], self.discrete_action_torque_magnitudes[action]
        else:
            if action < len(self.discrete_action_torque_magnitudes):
                return self.discrete_action_torque_magnitudes[action], 0.0
            else:
                return 0.0, self.discrete_action_torque_magnitudes[action - len(self.discrete_action_torque_magnitudes)]

    def _continuous_wheel_torque(self, action):
        return action[0], action[1] if not self._wheel_controlled_together else action[0], action[0]

    def _get_obs(self):
        pos, orientation = self._bullet_client.getBasePositionAndOrientation(self.twsbr)
        roll, pitch, yaw = pybullet.getEulerFromQuaternion(orientation)
        vel_linear, vel_angular = self._bullet_client.getBaseVelocity(self.twsbr)
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
        reward = 1.0  # Basic reward for each step
        roll, pitch, yaw, omega_x, omega_y, omega_z, x, y, z, x_dot, y_dot, z_dot = self._get_obs()

        # Penalize large pitch movements (forward/backward tilt)
        reward -= abs(omega_x) * self.parameters["reward"]["tilt_speed_penalty_scale"]
        # reward -= abs(omega_x) * self.parameters["reward"]["tilt_speed_penalty_scale"]
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
