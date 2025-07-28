from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import cv2
import random
import math
import os
import sys
import pybullet
import pybullet_data
from pybullet_utils import bullet_client

class TwsbrEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 100}
    
    def __init__(self,
                 render_mode=None,
                 robot_angle_limit=45,
                 action_type="continuous",
                 max_velocity=255,
                 truncation_steps=1000,
                 debug_info=False):
        super(TwsbrEnv, self).__init__()

        # Bottom camera settings
        self.projection_matrix = pybullet.computeProjectionMatrixFOV(fov=120.0, aspect=1.0, nearVal=0.0055, farVal=0.1)
        
        # Variabel global untuk menyimpan posisi garis sebelumnya
        self.prev_error_steer = 0  
        self.line_last_position = None  # None, "left", atau "right"

        # Link indices
        self.CAMERA_IDX = 2
        self.CAMERA_TARGET_IDX = 3
        self.IMU_IDX = 4
        
        self.LEFT_WHEEL_JOINT_IDX = 0
        self.RIGHT_WHEEL_JOINT_IDX = 1

        # Default direction of the camera_up_vector
        self.camera_up_vector = np.array([0, -1, 0])

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.render_fps = self.metadata["render_fps"]
        self.robot_angle_limit = robot_angle_limit
        
        assert action_type in ["discrete", "continuous"]
        self._action_type = action_type
        
        self.max_velocity = max_velocity
        self.discrete_velocity = np.linspace(-self.max_velocity, self.max_velocity, num=21)
        self.truncation_steps = truncation_steps
        self.debug_info = debug_info

        # Observation dan action space
        # Perhatikan: _get_current_line_position() sekarang mengembalikan array 8 nilai sensor.
        # Komponen observasi:
        # [ target_pitch, target_speed, pitch, 8 sensor (line), linear_speed, linear_left_wheel, linear_right_wheel,
        #   left_motor_power, right_motor_power, left_wheel_revolutions, right_wheel_revolutions ]
        # Kemudian, observasi sebelumnya juga ditambahkan, sehingga total dimensi = 18 x 2 = 36.
        first_state = np.array(
            [ np.pi,              # target pitch
              12.0,               # target speed
              np.pi ] +          # pitch
            [45.0]*8 +          # 8 nilai sensor garis (limit masing-masing 45.0)
            [12.0,              # linear_speed
             12.0,              # linear_left_wheel
             12.0,              # linear_right_wheel
             255.0,             # left_motor_power
             255.0,             # right_motor_power
             100.0,             # left_wheel_revolutions
             100.0]             # right_wheel_revolutions
        )
        # Total observasi = current state (18 elemen) + previous state (18 elemen)
        self.state_limit = np.concatenate((first_state, first_state))

        # Perhatikan: observation_space harus disesuaikan dengan dimensi baru (36 dimensi)
        self.observation_space = spaces.Box(low=-np.ones(self.state_limit.shape), high=np.ones(self.state_limit.shape))
        self.action_space = spaces.Discrete(len(self.discrete_velocity)*2) if action_type == "discrete" else spaces.Box(low=-1.0, high=1.0, shape=(2,))  # versi normalized
        
        self.obs_min = -self.state_limit
        self.obs_max = self.state_limit
        self._physics_client_id = -1
        self.load_robot()
        return

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.load_robot()
        self.step_counter = 0
        self.initialize_wheel_variables()
        self.left_motor_power = 0.0
        self.right_motor_power = 0.0
        self.prev_error_steer = 0
        self.line_last_position = None
        
        self.target_speed = 1.0 
        self.target_pitch = np.pi/45.0

        self.previous_state = self._get_first_obs().astype(np.float32)
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

        # Load ground plane dan robot
        self.texture = os.path.join(os.path.abspath(os.path.dirname(__file__)), "texture", "random_line_trace_ground.png")
        self.plane_id = self._bullet_client.loadURDF("plane.urdf", basePosition=[-3.0, -4.297, 0.0], globalScaling=3.0)
        self.tex_uid =  self._bullet_client.loadTexture(self.texture)
        self._bullet_client.changeVisualShape(self.plane_id, -1, textureUniqueId=self.tex_uid)
        
        self.urdf_file_name = os.path.join(os.path.abspath(os.path.dirname(__file__)), "urdf", "twsbr.urdf")
        
        # DOMAIN RANDOMIZATION
        if self.render_mode == "human":
            start_position, start_orientation = [0.0, 0.0, 0.0], pybullet.getQuaternionFromEuler([0, 0, 0])
        else:
            start_position, start_orientation = [0.0, 0.0, 0.0], pybullet.getQuaternionFromEuler([0, random.uniform(-np.pi/36, np.pi/36), 0])
        
        self.robot_id = self._bullet_client.loadURDF(self.urdf_file_name, basePosition=start_position, baseOrientation=start_orientation)
        self._bullet_client.setJointMotorControl2(self.robot_id, self.LEFT_WHEEL_JOINT_IDX, pybullet.VELOCITY_CONTROL, targetVelocity=0)
        self._bullet_client.setJointMotorControl2(self.robot_id, self.RIGHT_WHEEL_JOINT_IDX, pybullet.VELOCITY_CONTROL, targetVelocity=0)

    def step(self, action):
        left_wheel_velocity, right_wheel_velocity = self._apply_action(action)

        self._bullet_client.setJointMotorControl2(self.robot_id, 0, pybullet.VELOCITY_CONTROL, targetVelocity=left_wheel_velocity)
        self._bullet_client.setJointMotorControl2(self.robot_id, 1, pybullet.VELOCITY_CONTROL, targetVelocity=right_wheel_velocity)
        self._bullet_client.stepSimulation()
        
        self.left_motor_power = left_wheel_velocity
        self.right_motor_power = right_wheel_velocity

        observation = self._get_obs()
        self.observation_from_step = observation
        reward = self._get_reward()
        info = self._get_info()
        self.step_counter += 1

        if self.step_counter >= self.truncation_steps:
            truncated = True
            if observation[4] > 0.8 * observation[1]:
                self.truncation_steps += 500
        else:
            truncated = False
        terminated = True if abs(observation[2]) >= 0.25 or abs(np.mean(observation[3:11])) >= 0.95 else False
        
        if truncated:
            info["is_success"] = True
            reward += 25.0
        if terminated:
            info["is_success"] = False
            reward -= 25
        
        return observation.astype(np.float32), reward, terminated, truncated, info

    def _apply_action(self, action):
        return (np.clip(action[0] * self.max_velocity, -self.max_velocity, self.max_velocity),
                np.clip(action[1] * self.max_velocity, -self.max_velocity, self.max_velocity))
                
    def observation_denormalize(self):
        obs_normalized = self._get_obs()
        obs_denormalized = [
            self._denormalize_1d(value, min_val, max_val)
            for value, min_val, max_val in zip(obs_normalized, self.obs_min, self.obs_max)
        ]
        return np.array(obs_denormalized)
    
    # Fungsi rotasi
    def _Rx(self, theta):
        return np.array([[1, 0, 0],
                         [0, np.cos(theta), -np.sin(theta)],
                         [0, np.sin(theta), np.cos(theta)]])
    
    def _Ry(self, theta):
        return np.array([[np.cos(theta), 0, np.sin(theta)],
                         [0, 1, 0],
                         [-np.sin(theta), 0, np.cos(theta)]])
    
    def _Rz(self, theta):
        return np.array([[np.cos(theta), -np.sin(theta), 0],
                         [np.sin(theta), np.cos(theta), 0],
                         [0, 0, 1]])
    
    # Sensor readings
    def _get_current_angle(self):
        pos, orn = pybullet.getBasePositionAndOrientation(self.robot_id)
        roll, pitch, yaw = pybullet.getEulerFromQuaternion(orn)
        return pitch

    def _get_current_line_position(self):
        """
        Memproses gambar kamera dan mengembalikan array 8 nilai sensor
        yang merepresentasikan kondisi garis (misal, 0 untuk hitam dan 255 untuk putih)
        pada 8 posisi sepanjang baris tengah gambar.
        """
        camera_link_pose = pybullet.getLinkState(self.robot_id, self.CAMERA_IDX)[0]
        camera_target_link_pose = pybullet.getLinkState(self.robot_id, self.CAMERA_TARGET_IDX)[0]

        mobile_robot_roll, mobile_robot_pitch, mobile_robot_yaw = pybullet.getEulerFromQuaternion(
            pybullet.getLinkState(self.robot_id, self.CAMERA_IDX)[1])
        R = self._Rz(np.deg2rad(90.0) + mobile_robot_yaw) @ self._Ry(mobile_robot_pitch) @ self._Rx(mobile_robot_roll)
        rotate_camera_up_vector = R @ self.camera_up_vector

        view_matrix = pybullet.computeViewMatrix(
            cameraEyePosition=camera_link_pose, 
            cameraTargetPosition=camera_target_link_pose, 
            cameraUpVector=rotate_camera_up_vector
        )

        # Gunakan resolusi 600x300 untuk mendapatkan detail yang cukup
        width, height, rgb_img, depth_img, seg_img = pybullet.getCameraImage(
            600, 300, view_matrix, self.projection_matrix,
            renderer=pybullet.ER_BULLET_HARDWARE_OPENGL
        )
        img = np.reshape(rgb_img, (height, width, 4))
        gray = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGBA2RGB), cv2.COLOR_RGB2GRAY)
        # Binarisasi dengan threshold 128 (metode THRESH_BINARY_INV agar konsisten dengan kode awal)
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

        # Ambil 8 sampel dari baris tengah gambar secara merata
        center_row = height // 2
        sensor_positions = np.linspace(0, width - 1, num=8, dtype=int)
        sensor_values = [binary[center_row, pos] for pos in sensor_positions]

        # Jika tidak ada garis terdeteksi (misal semua sensor bernilai 0), bisa diberi nilai default
        if all(val == 0 for val in sensor_values):
            sensor_values = [45.0 for _ in range(8)]
        return sensor_values

    def initialize_wheel_variables(self):
        global prev_left_angle, prev_right_angle, left_wheel_revolutions, right_wheel_revolutions
        prev_left_angle = 0.0
        prev_right_angle = 0.0
        left_wheel_revolutions = 0
        right_wheel_revolutions = 0

    def _get_wheel_data(self):
        global prev_left_angle, prev_right_angle, left_wheel_revolutions, right_wheel_revolutions
        wheel_radius = 0.045

        left_wheel_state = pybullet.getJointState(self.robot_id, self.LEFT_WHEEL_JOINT_IDX)
        right_wheel_state = pybullet.getJointState(self.robot_id, self.RIGHT_WHEEL_JOINT_IDX)

        left_wheel_velocity = left_wheel_state[1]
        right_wheel_velocity = right_wheel_state[1]
        left_wheel_angle = left_wheel_state[0]
        right_wheel_angle = right_wheel_state[0]

        delta_left_angle = left_wheel_angle - prev_left_angle
        delta_right_angle = right_wheel_angle - prev_right_angle

        if abs(delta_left_angle) > np.pi:
            delta_left_angle -= np.sign(delta_left_angle) * 2 * np.pi
        if abs(delta_right_angle) > np.pi:
            delta_right_angle -= np.sign(delta_right_angle) * 2 * np.pi

        left_wheel_revolutions += delta_left_angle / (2 * np.pi)
        right_wheel_revolutions += delta_right_angle / (2 * np.pi)

        prev_left_angle = left_wheel_angle
        prev_right_angle = right_wheel_angle

        avg_angular_velocity = (left_wheel_velocity + right_wheel_velocity) / 2
        linear_speed = avg_angular_velocity * wheel_radius
        linear_left_wheel = left_wheel_velocity * wheel_radius
        linear_right_wheel = right_wheel_velocity * wheel_radius

        return linear_speed, linear_left_wheel, linear_right_wheel, left_wheel_revolutions, right_wheel_revolutions

    def _get_first_obs(self):
        linear_speed, linear_left_wheel, linear_right_wheel, left_wheel_revolutions, right_wheel_revolutions = self._get_wheel_data()
        pitch = self._get_current_angle()
        sensors = self._get_current_line_position()  # 8 nilai sensor garis
        obs = np.array(
            [self.target_pitch, self.target_speed, pitch] +
            sensors +
            [linear_speed, linear_left_wheel, linear_right_wheel, self.left_motor_power, self.right_motor_power, left_wheel_revolutions, right_wheel_revolutions]
        )
        return obs

    def _get_obs(self):
        linear_speed, linear_left_wheel, linear_right_wheel, left_wheel_revolutions, right_wheel_revolutions = self._get_wheel_data()
        pitch = self._get_current_angle()
        sensors = self._get_current_line_position()  # 8 nilai sensor garis
        obs_temp = np.array(
            [self.target_pitch, self.target_speed, pitch] +
            sensors +
            [linear_speed, linear_left_wheel, linear_right_wheel, self.left_motor_power, self.right_motor_power, left_wheel_revolutions, right_wheel_revolutions]
        )
        obs = np.concatenate((obs_temp, self.previous_state))
        self.previous_state = obs_temp
        obs = self._normalize_obs(obs)
        return obs

    def _normalize_obs(self, obs):
        return 2 * (obs - self.obs_min) / (self.obs_max - self.obs_min) - 1

    def _denormalize_1d(self, value, min_value, max_value):
        return value * (max_value - min_value) / 2 + (max_value + min_value) / 2

    def _get_reward(self):
        target_pitch, target_speed, pitch = self.observation_from_step[0:3]
        # Untuk sensor garis, kita ambil rata-rata nilai sensor (atau bisa juga proses lain sesuai kebutuhan)
        sensor_values = self.observation_from_step[3:11]
        linear_speed = self.observation_from_step[11]
        left_motor_power = self.observation_from_step[15]
        right_motor_power = self.observation_from_step[16]
        
        reward = 0.0
        pitch_error = target_pitch - pitch
        reward -= 0.1 * abs(pitch_error ** 2)
        
        # Gunakan rata-rata sensor sebagai representasi error posisi garis
        yaw_error = abs(np.mean(sensor_values) - 127.5) / 127.5  # Normalisasi error antara 0 dan 1
        reward -= 1 * (yaw_error ** 2)
        
        linear_speed_error = abs(target_speed - linear_speed)
        reward -= 0.01 * (linear_speed_error ** 2)
        
        return reward

    def _get_info(self):
        return {"step_count": self.step_counter}

    def render(self):
        if self.render_mode == "human" and self._physics_client_id >= 0 and not getattr(self, "rendered_status", False):
            self._bullet_client.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=45, cameraPitch=-45, cameraTargetPosition=[0, 0, 0])
            self.rendered_status = True
            return None

    def close(self):
        if self._physics_client_id >= 0:
            self._bullet_client.disconnect()
            self._physics_client_id = -1
