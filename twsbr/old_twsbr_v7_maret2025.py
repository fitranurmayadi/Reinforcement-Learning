import time
import math
import random
import os
import numpy as np
from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pybullet
import pybullet_data
from pybullet_utils import bullet_client

# ---------------- Environment Definition ----------------

class TwsbrEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 100}

    def __init__(self,
                 render_mode=None,
                 target_position=[0,0,0],
                 target_yaw=0,
                 max_velocity=255,
                 truncation_steps=1000,
                 debug_info=False):
        super(TwsbrEnv, self).__init__()

        self.manual_target_position = target_position
        self.manual_target_yaw = target_yaw

        # Indeks roda
        self.LEFT_WHEEL_JOINT_IDX = 0
        self.RIGHT_WHEEL_JOINT_IDX = 1

        self.render_mode = render_mode
        self.render_fps = self.metadata["render_fps"]
        self.max_velocity = max_velocity
        self.truncation_steps = truncation_steps
        self.debug_info = debug_info

        # Definisi state:
        # [pitch, pos_error_x, pos_error_y, yaw_error, linear_speed, left_motor_power, right_motor_power, target_x, target_y, target_yaw]
        # Batas state dalam satuan fisik:
        # pitch: ±π, pos_error: ±10 m, yaw_error: ±π, linear_speed: ±5 m/s, motor power: ±max_velocity, target_x, target_y: ±5, target_yaw: ±π
        self.state_limit = np.array([
            np.pi,              # pitch ±π rad
            np.pi,              # error yaw ±π rad
            10.0,               # error posisi x ±10 m
            10.0,               # error posisi y ±10 m
            10.0,                # linear speed ±10 m/s
            self.max_velocity,  # daya motor kiri
            self.max_velocity,  # daya motor kanan
            10.0,                # target posisi x ±10 m
            10.0,                # target posisi y ±10 m
            np.pi               # target yaw ±π rad
        ])
        # Observasi akan dinormalisasi ke rentang [-1, 1]
        self.observation_space = spaces.Box(low=-np.ones(self.state_limit.shape),
                                            high=np.ones(self.state_limit.shape), dtype=np.float32)
        self.obs_min = -self.state_limit
        self.obs_max = self.state_limit

        # Aksi: kontrol motor kiri dan kanan, kontinu antara [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self._physics_client_id = -1
        self.rendered_status = False

        # Inisialisasi motor
        self.left_motor_power = 0.0
        self.right_motor_power = 0.0

        # Target default (akan di-reset secara random)
        self.target_position = np.array([0.0, 0.0, 0.0])
        self.target_yaw = np.deg2rad(0)

        self.load_robot()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Load ulang robot dan simulasi dengan posisi awal random
        self.load_robot()
        self.step_counter = 0
        self.left_motor_power = 0.0
        self.right_motor_power = 0.0

        if self.render_mode == "human" and self.manual_target_position is not None and self.manual_target_yaw is not None:
            self.target_position = self.manual_target_position
            self.target_yaw = self.manual_target_yaw
        else:
            # Inisialisasi target secara random dalam area [-5, 5] untuk x dan y, dengan yaw random
            self.target_position = [0, 0, 0]#np.array([random.uniform(-1, 1), random.uniform(-1, 1), 0.0])
            self.target_yaw = 0 #random.uniform(-np.pi, np.pi)
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
        self._bullet_client.resetSimulation()
        self._bullet_client.setGravity(0, 0, -9.8)
        self._bullet_client.setTimeStep(1.0 / self.render_fps)
        self._bullet_client.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Muat ground plane
        plane_path = os.path.join(pybullet_data.getDataPath(), "plane.urdf")
        self.plane_id = self._bullet_client.loadURDF(plane_path, basePosition=[0, 0, 0])

        # Muat robot dari file URDF (pastikan file "twsbr_v1.urdf" tersedia di folder "urdf")
        urdf_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "urdf")
        self.urdf_file_name = os.path.join(urdf_dir, "twsbr_v1.urdf")

        # Inisialisasi posisi dan orientasi robot secara random (hanya yaw yang random, pitch dan roll = 0)
        if self.render_mode == "human":
            start_position = [random.uniform(-2, 2), random.uniform(-2, 2), 0.0]
            yaw = random.uniform(-np.pi, np.pi)
        else:
            start_position = [random.uniform(-2, 2), random.uniform(-2, 2), 0.0]
            yaw = random.uniform(-np.pi, np.pi)
        start_orientation = pybullet.getQuaternionFromEuler([0, 0, yaw])

        self.robot_id = self._bullet_client.loadURDF(self.urdf_file_name,
                                                     basePosition=start_position,
                                                     baseOrientation=start_orientation)

        # Inisialisasi kontrol motor roda
        self._bullet_client.setJointMotorControl2(self.robot_id, self.LEFT_WHEEL_JOINT_IDX,
                                                  pybullet.VELOCITY_CONTROL, targetVelocity=0)
        self._bullet_client.setJointMotorControl2(self.robot_id, self.RIGHT_WHEEL_JOINT_IDX,
                                                  pybullet.VELOCITY_CONTROL, targetVelocity=0)

    def step(self, action):
        left_wheel_velocity, right_wheel_velocity = self._apply_action(action)

        self._bullet_client.setJointMotorControl2(self.robot_id, self.LEFT_WHEEL_JOINT_IDX,
                                                  pybullet.VELOCITY_CONTROL, targetVelocity=left_wheel_velocity)
        self._bullet_client.setJointMotorControl2(self.robot_id, self.RIGHT_WHEEL_JOINT_IDX,
                                                  pybullet.VELOCITY_CONTROL, targetVelocity=right_wheel_velocity)
        self._bullet_client.stepSimulation()

        self.left_motor_power = left_wheel_velocity
        self.right_motor_power = right_wheel_velocity

        observation = self._get_obs()
        reward = self._get_reward()
        info = self._get_info()
        self.step_counter += 1

        # --- Gunakan state normalized langsung untuk kondisi terminasi ---
        # Ambil nilai dari observasi yang sudah dinormalisasi
        pitch = observation[0]
        pos_error_x = observation[1]
        pos_error_y = observation[2]
        yaw_error = observation[3]
        distance = np.linalg.norm([pos_error_x, pos_error_y])

        # Definisikan threshold dalam ruang normalized
        # Karena normalisasi: 
        #   - pitch dinormalisasi dari [-π, π] sehingga nilai normalized = pitch/π.
        #   - pos_error dinormalisasi dari [-10, 10] sehingga nilai normalized = pos_error/10.
        #   - yaw_error dinormalisasi dari [-π, π] sehingga nilai normalized = yaw_error/π.
        pitch_threshold = 0.25        # setara dengan 45 degree
        distance_threshold = 0.001         # setara dengan 0.5 m error (0.05 dalam normalized)
        yaw_threshold = 0.01

        if abs(pitch) > pitch_threshold:
            terminated = True
            info["is_success"] = False
            reward = -100  # Penalti jatuh
        elif distance < distance_threshold and abs(yaw_error) < yaw_threshold:
            terminated = True
            info["is_success"] = True
            reward += 100  # Bonus sukses
        else:
            terminated = False

        truncated =  True if self.step_counter >= self.truncation_steps else False

        return observation.astype(np.float32), reward, terminated, truncated, info

    def _apply_action(self, action):
        # Skala aksi kontinu ke kecepatan maksimum motor
        left_velocity = np.clip(action[0] * self.max_velocity, -self.max_velocity, self.max_velocity)
        right_velocity = np.clip(action[1] * self.max_velocity, -self.max_velocity, self.max_velocity)
        return left_velocity, right_velocity

    def _get_current_state(self):
        # Dapatkan posisi dan orientasi robot
        pos, orn = self._bullet_client.getBasePositionAndOrientation(self.robot_id)
        x, y, z = pos
        roll, pitch, yaw = pybullet.getEulerFromQuaternion(orn)
        return np.array([x, y, z]), roll, pitch, yaw

    def _angle_diff(self, target, current):
        # Normalisasi perbedaan sudut ke rentang [-π, π]
        diff = target - current
        while diff > np.pi:
            diff -= 2 * np.pi
        while diff < -np.pi:
            diff += 2 * np.pi
        return diff

    def _get_wheel_data(self):
        wheel_radius = 0.045  # Radius roda dalam meter
        left_wheel_state = self._bullet_client.getJointState(self.robot_id, self.LEFT_WHEEL_JOINT_IDX)
        right_wheel_state = self._bullet_client.getJointState(self.robot_id, self.RIGHT_WHEEL_JOINT_IDX)
        left_wheel_velocity = left_wheel_state[1]
        right_wheel_velocity = right_wheel_state[1]
        avg_angular_velocity = (left_wheel_velocity + right_wheel_velocity) / 2
        linear_speed = avg_angular_velocity * wheel_radius
        return linear_speed

    def _get_obs(self):
        # Dapatkan state robot
        pos, roll, pitch, yaw = self._get_current_state()
        # Hitung error posisi dan orientasi terhadap target (dalam satuan fisik)
        linear_speed = self._get_wheel_data()

        # Buat observasi fisik
        obs_physical = np.array([
            pitch,
            yaw,
            pos[0],
            pos[1],
            linear_speed,
            self.left_motor_power,
            self.right_motor_power,
            self.target_position[0],
            self.target_position[1],
            self.target_yaw
        ])
        # Normalisasi observasi ke rentang [-1, 1]
        return self._normalize_obs(obs_physical)

    def _normalize_obs(self, obs):
        # Normalisasi linear (state fisik) menjadi [-1, 1]
        return 2 * (obs - self.obs_min) / (self.obs_max - self.obs_min) - 1

    def _get_reward(self):
        # Ambil observasi
        obs = self._get_obs()
        pitch = obs[0]
        yaw = obs[1]
        pos_x = obs[2]
        pos_y = obs[3]
        target_x = obs[7]
        target_y = obs[8]
        target_yaw = obs[9]
    
        # 1. Hitung Error Posisi
        distance_error = np.linalg.norm([target_x - pos_x, target_y - pos_y])
    
        # 2. Hitung Error Yaw dengan pendekatan circular (wrap-around)
        yaw_error = np.arctan2(np.sin(target_yaw - yaw), np.cos(target_yaw - yaw))
    
        # 3. Reward dasar (positif)
        reward = 1.0  
    
        # 4. Penalti untuk error posisi (gunakan exponential decay agar lebih presisi di target)
        reward -= np.clip(5 * (distance_error**2), 0, 1)
    
        # 5. Penalti untuk error yaw (gunakan pendekatan quadratic untuk memastikan orientasi presisi)
        reward -= np.clip(0.2 * (yaw_error**2), 0, 0.5)
    
        # 6. Tambahkan bonus jika mendekati target posisi dengan sangat presisi
        if distance_error < 0.01:  # Hampir sampai (1 cm error)
            reward += 0.5
        if distance_error < 0.005:  # Tepat di target (0.5 cm error)
            reward += 1.0
    
        # 7. Tambahkan bonus jika yaw sangat presisi (dalam 1°)
        if abs(yaw_error) < np.deg2rad(5):  # 5 derajat
            reward += 0.3
        if abs(yaw_error) < np.deg2rad(1):  # 1 derajat
            reward += 0.5
    
        return reward

    def _get_info(self):
        return {"step_count": self.step_counter,
                "target_position": self.target_position,
                "target_yaw": self.target_yaw}

    def render(self):
        if self.render_mode == "human" and self._physics_client_id >= 0 and not self.rendered_status:
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