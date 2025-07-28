from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import math
import time
import os
import pybullet
import pybullet_data
from pybullet_utils import bullet_client

class TwsbrEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self,
                 render_mode=None,
                 robot_angle_limit=45,
                 action_type="continuous",
                 max_velocity=255,
                 truncation_steps=1000,
                 debug_info=False):
        super(TwsbrEnv, self).__init__()

        # Indeks joint roda
        self.LEFT_WHEEL_JOINT_IDX = 0
        self.RIGHT_WHEEL_JOINT_IDX = 1

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.render_fps = self.metadata["render_fps"]
        self.robot_angle_limit = robot_angle_limit

        assert action_type in ["continuous"]
        self._action_type = action_type

        self.max_velocity = max_velocity
        self.left_motor_power = 0.0
        self.right_motor_power = 0.0
        self.truncation_steps = truncation_steps
        self.debug_info = debug_info

        # Batas nilai tiap elemen observasi (untuk normalisasi)
        # Format: [pitch, yaw, pos_x, pos_y, vel_x, vel_y, left_wheel, right_wheel,
        #          prev_pitch, prev_yaw, prev_pos_x, prev_pos_y, prev_vel_x, prev_vel_y, prev_left_wheel, prev_right_wheel,
        #          target_x, target_y]
        self.state_limit = np.array([
            np.pi,   # pitch
            np.pi,   # yaw
            10,      # pos_x
            10,      # pos_y
            10,      # target x
            10,      # target y
        ])

        self.observation_space = spaces.Box(low=-np.ones(self.state_limit.shape),
                                            high=np.ones(self.state_limit.shape),
                                            dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        self.obs_min = -self.state_limit
        self.obs_max = self.state_limit

        self._physics_client_id = -1
        self.load_robot()
        return

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.load_robot()
        self.step_counter = 0
        return self._get_obs().astype(np.float32), self._get_info()

    def load_robot(self):
        if self._physics_client_id < 0:
            if self.render_mode == "human":
                self._bullet_client = bullet_client.BulletClient(pybullet.GUI, options="--width=1920 --height=1000")
                self.rendered_status = False
                
            else:
                self._bullet_client = bullet_client.BulletClient(pybullet.DIRECT)
                
            self._init_physics_client()
        else:
            self._bullet_client.removeBody(self.robot_id)
            self._init_physics_client()

        

    def _init_physics_client(self):
        self._physics_client_id = self._bullet_client._client
        # Inisialisasi simulasi
        self._bullet_client.resetSimulation()
        self._bullet_client.setGravity(0, 0, -9.8)
        self._bullet_client.setTimeStep(1.0 / self.render_fps)
        self._bullet_client.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Muat plane dan robot
        self.plane_id = self._bullet_client.loadURDF("plane.urdf")

        if self.render_mode == "human": 
            # Generate target acak di bidang (x,y) dengan z = 0 (misal, rentang [-10,10])
            self.target_x = 5
            self.target_y = 5
            self.target = np.array([self.target_x, self.target_y, 0.0])
        else:
            # Generate target acak di bidang (x,y) dengan z = 0 (misal, rentang [-10,10])
            self.target_x = round(random.uniform(-10, 10), 2)
            self.target_y = round(random.uniform(-10, 10), 2)
            self.target = np.array([self.target_x, self.target_y, 0.0])
        

        # Posisi dan orientasi awal robot
        start_position = [0.0, 0.0, 0.0]
        start_orientation = pybullet.getQuaternionFromEuler([0, random.uniform(-np.pi/10, np.pi/10), 0])
        
        # Asumsikan file URDF robot berada di subfolder "urdf" yang sejajar dengan file ini
        self.urdf_file_name = os.path.join(os.path.abspath(os.path.dirname(__file__)), "urdf", "twsbr.urdf")
        self.robot_id = self._bullet_client.loadURDF(self.urdf_file_name,
                                                     basePosition=start_position,
                                                     baseOrientation=start_orientation)

        # Inisialisasi motor roda ke kecepatan 0
        self._bullet_client.setJointMotorControl2(self.robot_id, self.LEFT_WHEEL_JOINT_IDX,
                                                  pybullet.VELOCITY_CONTROL, targetVelocity=0)
        self._bullet_client.setJointMotorControl2(self.robot_id, self.RIGHT_WHEEL_JOINT_IDX,
                                                  pybullet.VELOCITY_CONTROL, targetVelocity=0)

    def step(self, action):
        # Terapkan aksi: ubah nilai aksi (rentang [-1,1]) menjadi kecepatan roda aktual
        left_wheel_velocity, right_wheel_velocity = self._apply_action(action)
        self._bullet_client.setJointMotorControl2(self.robot_id, self.LEFT_WHEEL_JOINT_IDX,
                                                  pybullet.VELOCITY_CONTROL, targetVelocity=left_wheel_velocity)
        self._bullet_client.setJointMotorControl2(self.robot_id, self.RIGHT_WHEEL_JOINT_IDX,
                                                  pybullet.VELOCITY_CONTROL, targetVelocity=right_wheel_velocity)
        self._bullet_client.stepSimulation()
        time.sleep(1./self.render_fps)

        self.left_motor_power = left_wheel_velocity
        self.right_motor_power = right_wheel_velocity

        observation = self._get_obs()
        self.observation_from_step = observation

        reward = self._get_reward()
        info = self._get_info()
        self.step_counter += 1

        # Terminasi: jika robot jatuh (pitch atau yaw melebihi batas) atau sudah mencapai target
        terminated = False
        truncated = False

        if abs(observation[0]) > 0.25 :
            terminated = True
            info["is_success"] = False
            reward -= 100.0

        if self.distance_to_target < 0.001 or self.step_counter >= self.truncation_steps:
            truncated = True
            info["is_success"] = True
            reward += 100.0

        # Sertakan informasi target pada info
        info["target"] = self.target
        return observation.astype(np.float32), reward, terminated, truncated, info

    def _apply_action(self, action):
        # Skala aksi ke kecepatan roda aktual
        left_wheel_velocity = np.clip(action[0] * self.max_velocity, -self.max_velocity, self.max_velocity)
        right_wheel_velocity = np.clip(action[1] * self.max_velocity, -self.max_velocity, self.max_velocity)
        return left_wheel_velocity, right_wheel_velocity

    def observation_denormalize(self):
        obs_normalized = self._get_obs()
        obs_denormalized = [
            self._denormalize_1d(value, min_val, max_val)
            for value, min_val, max_val in zip(obs_normalized, self.obs_min, self.obs_max)
        ]
        return np.array(obs_denormalized)

    def _get_current_angle(self):
        # Dapatkan kondisi robot: posisi, orientasi (pitch, yaw) dan kecepatan linear
        pos, orientation = self._bullet_client.getBasePositionAndOrientation(self.robot_id)
        roll, pitch, yaw = pybullet.getEulerFromQuaternion(orientation)
        vel_linear, _ = self._bullet_client.getBaseVelocity(self.robot_id)
        return np.array([pitch, yaw, pos[0], pos[1], vel_linear[0], vel_linear[1]])

    def _get_first_obs(self):
        # Observasi awal: kondisi saat ini dengan kecepatan roda = 0
        pitch, yaw, pos_x, pos_y, vel_x, vel_y = self._get_current_angle()
        obs = np.array([pitch, yaw, pos_x, pos_y, vel_x, vel_y, 0.0, 0.0])
        return obs

    def _get_obs(self):
        # Kondisi saat ini (8 nilai)
        pitch, yaw, pos_x, pos_y, vel_x, vel_y = self._get_current_angle()
        obs= np.array([pitch, yaw, pos_x, pos_y, self.target[0], self.target[1]])
        # Normalisasi observasi ke rentang [-1, 1]
        return self._normalize_obs(obs)

    def _normalize_obs(self, obs):
        return 2 * (obs - self.obs_min) / (self.obs_max - self.obs_min) - 1

    def _denormalize_1d(self, value, min_value, max_value):
        return value * (max_value - min_value) / 2 + (max_value + min_value) / 2

    def _get_reward(self):
        # Gunakan nilai mentah untuk reward
        reward = 0.0
        pitch, yaw, pos_x, pos_y, target_x, target_y = self.observation_from_step

        distance = math.sqrt((pos_x - target_x)**2 + (pos_y - target_y)**2)
        
        #stability_penalty = 10 * abs(pitch) + 10 * abs(yaw)
        
        # Reward lebih tinggi jika robot mendekati target dan tetap stabil
        reward -= 0.7  * ( distance **2)
        reward -= 0.3 * (abs(pitch ** 2))
        # Bonus jika target tercapai
        if distance < 0.001:
            reward += 50.0

        self.distance_to_target = distance

        if self.debug_info:
            print(f" \r Reward: {reward}, Distance: {distance}, Pitch: {pitch}, Yaw: {yaw} , Pos: {pos_x, pos_y} ,  Target: {target_x, target_y}", end="")
        return reward

    def _get_info(self):
        return {"step_count": self.step_counter}

    def render(self):
        if self.render_mode == "human" and self._physics_client_id >= 0 and not getattr(self, 'rendered_status', False):
            self._bullet_client.resetDebugVisualizerCamera(cameraDistance=1.5,
                                                             cameraYaw=45,
                                                             cameraPitch=-45,
                                                             cameraTargetPosition=[0, 0, 0])
            self.rendered_status = True
        return None

    def close(self):
        if self._physics_client_id >= 0:
            self._bullet_client.disconnect()
            self._physics_client_id = -1
