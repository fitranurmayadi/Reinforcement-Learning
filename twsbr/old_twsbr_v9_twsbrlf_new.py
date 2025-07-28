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
        self.projection_matrix = pybullet.computeProjectionMatrixFOV(
            fov=120.0, aspect=1.0, nearVal=0.0055, farVal=0.1)

        # Variabel global untuk menyimpan posisi terakhir garis
        self.prev_error_steer = 0  # Posisi terakhir yang terdeteksi
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

        # Hanya observasi: pitch, yaw (error garis), forward_velocity
        # Batas state: pitch dalam [-π, π], yaw dalam [-45, 45] (satuan error), forward velocity dalam [-12, 12] m/s
        self.state_limit = np.array([np.pi, 45.0, 12.0])
        self.observation_space = spaces.Box(low=-self.state_limit,
                                            high=self.state_limit,
                                            dtype=np.float32)
        self.action_space = (spaces.Discrete(len(self.discrete_velocity) * 2)
                             if action_type == "discrete"
                             else spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32))

        self.obs_min = -self.state_limit
        self.obs_max = self.state_limit
        self._physics_client_id = -1
        self.load_robot()

        # Nilai target untuk reward (bisa disesuaikan)
        self.target_speed = 1.0
        self.target_pitch = np.pi / 45.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.load_robot()
        self.step_counter = 0
        self.left_motor_power = 0.0
        self.right_motor_power = 0.0
        self.prev_error_steer = 0
        self.line_last_position = None
        return self._get_obs().astype(np.float32), self._get_info()

    def load_robot(self):
        if self._physics_client_id < 0:
            if self.render_mode == "human":
                self._bullet_client = bullet_client.BulletClient(
                    pybullet.GUI, options="--width=1920 --height=1000")
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
        self.tex_uid = self._bullet_client.loadTexture(self.texture)
        self._bullet_client.changeVisualShape(self.plane_id, -1, textureUniqueId=self.tex_uid)

        self.urdf_file_name = os.path.join(os.path.abspath(os.path.dirname(__file__)), "urdf", "twsbr.urdf")

        # DOMAIN RANDOMIZATION
        if self.render_mode == "human":
            start_position = [0.0, 0.0, 0.0]
            start_orientation = pybullet.getQuaternionFromEuler([0, 0, 0])
        else:
            start_position = [0.0, 0.0, 0.0]
            start_orientation = pybullet.getQuaternionFromEuler([0, random.uniform(-np.pi/36, np.pi/36), 0])

        self.robot_id = self._bullet_client.loadURDF(self.urdf_file_name,
                                                     basePosition=start_position,
                                                     baseOrientation=start_orientation)

        self._bullet_client.setJointMotorControl2(self.robot_id, self.LEFT_WHEEL_JOINT_IDX,
                                                   pybullet.VELOCITY_CONTROL, targetVelocity=0)
        self._bullet_client.setJointMotorControl2(self.robot_id, self.RIGHT_WHEEL_JOINT_IDX,
                                                   pybullet.VELOCITY_CONTROL, targetVelocity=0)

    def _apply_action(self, action):
        return (np.clip(action[0] * self.max_velocity, -self.max_velocity, self.max_velocity),
                np.clip(action[1] * self.max_velocity, -self.max_velocity, self.max_velocity))

    # Fungsi untuk mengkonversi kecepatan global ke frame lokal (menggunakan forward vector)
    def _get_forward_velocity(self):
        pos, orn = pybullet.getBasePositionAndOrientation(self.robot_id)
        vel, _ = pybullet.getBaseVelocity(self.robot_id)
        vel = np.array(vel)
        rot_mat = pybullet.getMatrixFromQuaternion(orn)
        # Asumsi: sumbu x adalah arah depan robot
        forward_vector = np.array([rot_mat[0], rot_mat[3], rot_mat[6]])
        # Hilangkan komponen vertikal
        forward_vector[2] = 0
        norm = np.linalg.norm(forward_vector)
        if norm > 1e-6:
            forward_vector /= norm
        else:
            forward_vector = np.array([1, 0, 0])
        vel[2] = 0
        forward_velocity = np.dot(vel, forward_vector)
        return forward_velocity

    # Sensor readings
    def _get_current_angle(self):
        # Mengembalikan pitch (miring depan/belakang)
        pos, orn = pybullet.getBasePositionAndOrientation(self.robot_id)
        _, pitch, _ = pybullet.getEulerFromQuaternion(orn)
        return pitch

    def _get_current_line_position(self):
        # Proses kamera untuk mendeteksi garis
        camera_link_pose = pybullet.getLinkState(self.robot_id, self.CAMERA_IDX)[0]
        camera_target_link_pose = pybullet.getLinkState(self.robot_id, self.CAMERA_TARGET_IDX)[0]

        mobile_robot_roll, mobile_robot_pitch, mobile_robot_yaw = pybullet.getEulerFromQuaternion(
            pybullet.getLinkState(self.robot_id, self.CAMERA_IDX)[1])
        R = self._Rz(np.deg2rad(90.0) + mobile_robot_yaw) @ self._Ry(mobile_robot_pitch) @ self._Rx(mobile_robot_roll)
        rotate_camera_up_vector = R @ self.camera_up_vector

        view_matrix = pybullet.computeViewMatrix(cameraEyePosition=camera_link_pose,
                                                 cameraTargetPosition=camera_target_link_pose,
                                                 cameraUpVector=rotate_camera_up_vector)

        width, height, rgb_img, depth_img, seg_img = pybullet.getCameraImage(90, 1, view_matrix, self.projection_matrix)
        img = np.reshape(rgb_img, (height, width, 4))
        gray = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGBA2RGB), cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
        moments = cv2.moments(binary)

        cx = None if moments["m00"] == 0 else moments["m10"] / moments["m00"]

        if cx is None:  # Garis tidak terdeteksi
            if self.line_last_position == "left":
                error_steer = -width / 2
            elif self.line_last_position == "right":
                error_steer = width / 2
            else:
                error_steer = self.prev_error_steer
        else:  # Garis terdeteksi
            error_steer = cx - (width / 2)
            if cx < width / 2:
                self.line_last_position = "left"
            elif cx > width / 2:
                self.line_last_position = "right"
        self.prev_error_steer = error_steer
        line_position = error_steer if cx is not None else 45.0
        return line_position

    # Matriks rotasi untuk transformasi
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

    def _get_first_obs(self):
        pitch = self._get_current_angle()
        yaw = self._get_current_line_position()
        forward_velocity = self._get_forward_velocity()
        return np.array([pitch, yaw, forward_velocity])

    def _get_obs(self):
        pitch = self._get_current_angle()
        yaw = self._get_current_line_position()
        forward_velocity = self._get_forward_velocity()
        obs = np.array([pitch, yaw, forward_velocity])
        return self._normalize_obs(obs)

    def _normalize_obs(self, obs):
        return 2 * (obs - self.obs_min) / (self.obs_max - self.obs_min) - 1

    def _denormalize_1d(self, value, min_value, max_value):
        return value * (max_value - min_value) / 2 + (max_value + min_value) / 2

    def _get_reward(self):
        # Reward dihitung berdasarkan kesalahan pitch, yaw (error garis), dan kecepatan maju
        pitch = self._get_current_angle()
        yaw = self._get_current_line_position()
        forward_velocity = self._get_forward_velocity()
        pitch_error = self.target_pitch - pitch
        speed_error = self.target_speed - forward_velocity
        reward = -0.1 * (pitch_error ** 2) - 1.0 * (yaw ** 2) - 0.01 * (speed_error ** 2)
        return reward

    def _get_info(self):
        return {"step_count": self.step_counter}

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

        if self.step_counter >= self.truncation_steps:
            truncated = True
            # Contoh peningkatan batas step jika kecepatan maju melebihi target 80%
            if self._get_forward_velocity() > 0.8 * self.target_speed:
                self.truncation_steps += 500
        else:
            truncated = False

        # Terminasi jika pitch atau yaw melebihi batas
        current_pitch = self._get_current_angle()
        current_yaw = self._get_current_line_position()
        terminated = True if abs(current_pitch) >= 0.25 or abs(current_yaw) >= 0.95 else False

        if truncated:
            info["is_success"] = True
            reward += 25.0
        if terminated:
            info["is_success"] = False
            reward -= 25

        return observation.astype(np.float32), reward, terminated, truncated, info

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
