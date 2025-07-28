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
    metadata = {"render_modes": ["human", "rgb_array"], "simulation_fps": 60}
    # Membuat environment
    parameters = {
    "render": {
        "render_height": 1080,
        "render_width": 1960
    },
    "urdf_file_name": "urdf/twsbr.urdf",
    "binary_action": {
        "torque_magnitude": 10.0
    },
    "discrete_action": {
        "torque_magnitudes": [5.0, 10.0, 15.0]
    },
    "continuous_action": {
        "max_torque_magnitude": 20.0
    },
    "reward": {
        "tilt_speed_penalty_scale": 0.1
    },
    "truncation_steps": 1000
    }

    def __init__(self,
                 render_mode=None,
                 action_type="continuous",
                 wheels_controlled_together=False,
                 roll_threshold_deg=180.0,          # terminate after body tilt reaches roll_threshold deg (tilt)
                 x_threshold=10.0,                 # terminate after robot moves more than x_threshold [m]
                 y_threshold=10.0,                 # terminate after robot moves more than y_threshold [m]
                 parameters=parameters):                 # parameters dictionary
        # configs
        if parameters is None:
            raise ValueError("Parameters dictionary must be provided")

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        assert action_type in ["binary", "discrete", "continuous"]
        self._action_type = action_type
        self._wheel_controlled_together = wheels_controlled_together

        self._render_height = parameters["render"]["render_height"]  # For render_mode="rgb_array"
        self._render_width = parameters["render"]["render_width"]    # For render_mode="rgb_array"
        self._physics_client_id = -1

        self.roll_threshold_rad = roll_threshold_deg / 180.0 * np.pi
        self.x_threshold = x_threshold
        self.y_threshold = y_threshold
        self.this_file_dir_path = os.path.abspath(os.path.dirname(__file__))
        self.urdf_file_name = parameters["urdf_file_name"]

        # State := [roll, pitch, yaw, omega_x, omega_y, omega_z, x, y, z, x_dot, y_dot, z_dot]
        state_limit = np.array([np.pi,                      # roll: tilt angle constrained in range
                                np.pi,                      # pitch
                                np.pi,                      # yaw
                                np.finfo(np.float32).max,   # omega_x
                                np.finfo(np.float32).max,   # omega_y
                                np.finfo(np.float32).max,   # omega_z
                                self.x_threshold,           # x coordinate constrained in range
                                self.y_threshold,           # y coordinate constrained in range.
                                np.finfo(np.float32).max,   # z
                                np.finfo(np.float32).max,   # x_dot
                                np.finfo(np.float32).max,   # y_dot
                                np.finfo(np.float32).max])  # z_dot
        self.observation_space = gym.spaces.Box(low=-state_limit,
                                                high=state_limit)

        # Action space
        if self._action_type == "binary":
            self.binary_action_torque_magnitude = parameters["binary_action"]["torque_magnitude"]
            if self._wheel_controlled_together:
                self.action_space = gym.spaces.Discrete(2)  # 0: backward, 1: forward
            else:
                self.action_space = gym.spaces.Discrete(4)  # 2 wheels: 0/1 for each

        elif self._action_type == "discrete":
            self.discrete_action_torque_magnitudes = parameters["discrete_action"]["torque_magnitudes"]
            if self._wheel_controlled_together:
                self.action_space = gym.spaces.Discrete(len(self.discrete_action_torque_magnitudes))
            else:
                self.action_space = gym.spaces.Discrete(2 * len(self.discrete_action_torque_magnitudes))

        elif self._action_type == "continuous":
            self.continuous_action_max_torque_magnitude = parameters["continuous_action"]["max_torque_magnitude"]
            if self._wheel_controlled_together:
                self.action_space = gym.spaces.Box(low=-np.array([self.continuous_action_max_torque_magnitude]),
                                                   high=np.array([self.continuous_action_max_torque_magnitude]))
            else:
                self.action_space = gym.spaces.Box(low=-np.array([self.continuous_action_max_torque_magnitude] * 2),
                                                   high=np.array([self.continuous_action_max_torque_magnitude] * 2))

    def reset(self,
              seed=None,
              options=None,
              start_position=None,
              start_tilt_deg=None,
              start_yaw_deg=0.0):
        super().reset(seed=seed)  # setting seed
        
        this_file_dir_path = os.path.abspath(os.path.dirname(__file__))  # Menentukan direktori file ini
        urdf_file_name = os.path.join(this_file_dir_path, "twsbr_env", "urdf", "twsbr.urdf")

        # Setup PyBullet client if not done yet
        if self._physics_client_id < 0:
            if self.render_mode == "human":
                self._bullet_client = bullet_client.BulletClient(connection_mode=pybullet.GUI)
            else:
                self._bullet_client = bullet_client.BulletClient()

            self._physics_client_id = self._bullet_client._client
            self._bullet_client.resetSimulation()
            self._bullet_client.setGravity(0, 0, -9.8)
            self._bullet_client.setTimeStep(1.0 / self.metadata["simulation_fps"])

            # Load ground plane
            pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
            self.ground_plane = self._bullet_client.loadURDF("plane.urdf")

            # Load robot
            self.twsbr = self._bullet_client.loadURDF(os.path.join(self.this_file_dir_path, self.urdf_file_name))

        self._bullet_client.removeBody(self.twsbr)

        # Initialize new episode
        init_roll = self.np_random.uniform(low=-0.2, high=0.2, size=1) if start_tilt_deg is None else start_tilt_deg / 180.0 * np.pi
        init_yaw = self.np_random.uniform(low=-np.pi, high=np.pi, size=1) if start_yaw_deg is None else start_yaw_deg / 180.0 * np.pi
        start_orientation = pybullet.getQuaternionFromEuler([init_roll, 0, init_yaw])
        start_position = [0.0, 0.0, 0.001] if start_position is None else start_position
        
        self.twsbr = self._bullet_client.loadURDF(os.path.join(self.this_file_dir_path, self.urdf_file_name),
                                                  basePosition=start_position,
                                                  baseOrientation=start_orientation)

        # Turn off default joint velocity control
        self._bullet_client.setJointMotorControl2(self.twsbr, 0, pybullet.VELOCITY_CONTROL, force=0)
        self._bullet_client.setJointMotorControl2(self.twsbr, 1, pybullet.VELOCITY_CONTROL, force=0)

        self.step_counter = 0
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        torque1, torque2 = self._apply_action(action)  # Make sure this returns exactly two values

         # Apply torque
        self._bullet_client.setJointMotorControl2(self.twsbr, jointIndex=0,
                                                  controlMode=self._bullet_client.TORQUE_CONTROL,
                                                  force=torque1)
        self._bullet_client.setJointMotorControl2(self.twsbr, jointIndex=1,
                                                  controlMode=self._bullet_client.TORQUE_CONTROL,
                                                  force=torque2)
        self._bullet_client.stepSimulation()

        observation = self._get_obs()
        roll, pitch, yaw, omega_x, omega_y, omega_z, x, y, z, x_dot, y_dot, z_dot = observation
        reward = self._get_reward()
        info = self._get_info()

        self.step_counter += 1
        if self.step_counter >= self.parameters["truncation_steps"]:
            truncated = True
            info["is_success"] = True
        else:
            truncated = False

        terminated = abs(roll) > self.roll_threshold_rad
        if terminated:
            info["is_success"] = False

        return observation, reward, terminated, truncated, info

    def _apply_action(self, action):
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
        x, y, z = pos
        return np.array([roll, pitch, yaw, omega_x, omega_y, omega_z, x, y, z, x_dot, y_dot, z_dot])

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
            # Menyembunyikan elemen visual default
            self._bullet_client.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)
            self._bullet_client.configureDebugVisualizer(pybullet.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
            self._bullet_client.configureDebugVisualizer(pybullet.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
            self._bullet_client.configureDebugVisualizer(pybullet.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
            self._bullet_client.configureDebugVisualizer(pybullet.COV_ENABLE_SHADOWS, 1)
            
            self._render_width = 1920  # Ganti dengan lebar layar Anda
            self._render_height = 1080  # Ganti dengan tinggi layar Anda
            # Reset kamera debug visualizer (opsional)
            self._bullet_client.resetDebugVisualizerCamera(
                cameraDistance=1.5,  # Jarak kamera
                cameraYaw=0,         # Yaw kamera
                cameraPitch=-30,      # Pitch kamera
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
            (_, _, px, _, _) = self._bullet_client.getCameraImage(
                width=self._render_width,
                height=self._render_height,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix
            )

            # Kembalikan gambar dalam format RGB (tanpa alpha channel)
            return np.array(px)[:, :, :3]

    def close(self):
        if self._physics_client_id >= 0:
            self._bullet_client.disconnect()
            self._physics_client_id = -1
