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
                debug_info = False
                ):

        super(TwsbrEnv, self).__init__()

        # Bottom camera settings
        self.projection_matrix = pybullet.computeProjectionMatrixFOV(fov=120.0, aspect=1.0, nearVal=0.0055, farVal=0.1)
        
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

        # Observation and action space

        # x_dot (velocity), pitch(balance angle), yaw(line angle), motor_left_speed, motor_right_speed
        # +-25 m/s , +- 180 degree(+-pi rad), +-4 units, left wheel, and right wheel ~ in  +-255 rad/s ( +- 40.6 rps or +- 1200 rpm) 
        self.state_limit = np.array([

            np.pi,           # target pitch
            12.0,            # Target linear  speed robot: ±12 m/s
            np.pi,           # Pitch: ±π rad , ±180 Derajat
            45.0,            # Yaw (error posisi garis): ±4 unit
            12.0,            # Kecepatan linear robot: ±12 m/s
            12.0,            # Kecepatan linear roda kiri: ±12 m/s
            12.0,            # Kecepatan linear roda kanan: ±12 m/s

            255.0,           # Kecepatan Putaran Roda Kiri (radps) = Action
            255.0,           # Kecepatan Putaran Roda Kanan (radps) = Action

            100.0,          # Jumlah Putaran roda kiri (revolusi): ±1000
            100.0,          # Jumlah Putaran roda kanan (revolusi): ±1000

            #prev state
            np.pi,           # target pitch
            12.0,            # Target linear  speed robot: ±12 m/s

            np.pi,           # Pitch: ±π rad , ±180 Derajat
            45.0,             # Yaw (error posisi garis): ±4 unit
            12.0,            # Kecepatan linear robot: ±12 m/s
            12.0,            # Kecepatan linear roda kiri: ±12 m/s
            12.0,            # Kecepatan linear roda kanan: ±12 m/s

            255.0,             # Kecepatan Putaran Roda Kiri (radps) = Action
            255.0,             # Kecepatan Putaran Roda Kanan (radps) = Action

            100.0,          # Jumlah Putaran roda kiri (revolusi): ±1000
            100.0,          # Jumlah Putaran roda kanan (revolusi): ±1000

        ])

        self.observation_space = spaces.Box(low=-np.ones(self.state_limit.shape), high=np.ones(self.state_limit.shape)) 
        self.action_space = spaces.Discrete(len(self.discrete_velocity)*2) if action_type == "discrete" else spaces.Box(low=-1.0, high=1.0, shape=(2,)) # normalize version
        
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
        self.prev_error_steer = 0  # Posisi terakhir yang terdeteksi
        self.line_last_position = None  # None, "left", atau "right"
        
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
            #self._bullet_client.resetSimulation()
            self._init_physics_client()

    def _init_physics_client(self):
        
        self._physics_client_id = self._bullet_client._client
        # Set up simulation environment
        self._bullet_client.resetSimulation()
        self._bullet_client.setGravity(0, 0, -9.8)
        self._bullet_client.setTimeStep(1.0 / self.render_fps)
        self._bullet_client.setAdditionalSearchPath(pybullet_data.getDataPath())

        
        #pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Load ground plane and robot
        
        self.texture = os.path.join(os.path.abspath(os.path.dirname(__file__)), "texture", "random_line_trace_ground.png")
        self.plane_id = self._bullet_client.loadURDF("plane.urdf", basePosition=[-3.0, -4.297, 0.0], globalScaling=3.0)
        
        self.tex_uid =  self._bullet_client.loadTexture(self.texture)
        self._bullet_client.changeVisualShape(self.plane_id, -1, textureUniqueId=self.tex_uid)
        
        #self.tex_file_name = os.path.join(os.path.abspath(os.path.dirname(__file__)), "texture", "line_trace_ground.png")
        #self.tex_uid = self._bullet_client.loadTexture(self.tex_file_name)
        #self._bullet_client.changeVisualShape(self.plane_id, -1, textureUniqueId=self.tex_uid)
        
        self.urdf_file_name = os.path.join(os.path.abspath(os.path.dirname(__file__)), "urdf", "twsbr.urdf")
      

        
        #DOMAIN RANDOMIZATION

        if self.render_mode == "human":
            #start_position, start_orientation = [0.0, 0.0, 0.0], pybullet.getQuaternionFromEuler([0, random.uniform(-np.pi/8, np.pi/8), random.uniform(-np.pi/8, np.pi/8)]) 
            #start_position, start_orientation = [0.0, 0.0, 0.0], pybullet.getQuaternionFromEuler([0, 0, np.pi/4]) 
            
            start_position, start_orientation = [0.0, 0.0, 0.0], pybullet.getQuaternionFromEuler([0, 0, 0])

        else:  #for training, give it more challenge         
            #start_position, start_orientation = [0.0, 0.0, 0.0], pybullet.getQuaternionFromEuler([0, random.uniform(-np.pi/8, np.pi/8), random.uniform(-np.pi, np.pi)])
            start_position, start_orientation = [0.0, 0.0, 0.0], pybullet.getQuaternionFromEuler([0, random.uniform(-np.pi/36, np.pi/36), 0]) 
            #start_position, start_orientation = [0.0, 0.0, 0.0], pybullet.getQuaternionFromEuler([0, 0, 0]) 
        
            
            #self.plane_lateral_friction = np.random.uniform(0.7, 1.0)  # lateral friction
            #self.plane_spinning_friction = np.random.uniform(0.5, 1.0)  # lateral friction
            
            #self.robot_mass = np.random.uniform(0.500, 0.550)  # Random mass from 500 to 550 grams
            #self.robot_inertia = np.random.uniform(0.0001, 0.001, size=3)  # Random inertia [Ixx, Iyy, Izz]
            #self.robot_lateral_friction = np.random.uniform(0.7, 1.0)  # lateral friction
            #self.robot_spinning_friction = np.random.uniform(0.5, 1.0)  # lateral friction
            #self.robot_wheels_inertial = np.random.uniform(0.0, 0.001, size=3)  # Random inertia [Ixx, Iyy, Izz]

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
             # Naikkan batas step jika berhasil
            if observation[4] > 0.8 * observation[1]:
                self.truncation_steps += 500
        else:
            truncated=False
        terminated = True if abs(observation[2]) >= 0.25 or abs(observation[3]) >= 0.95  else False #else False #or or or abs(observation[3]) >= 0.95
        
        if truncated==True:
            info["is_success"] = True
            reward += 25.0
        if terminated==True:
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
        #print(obs_denormalized)
        return np.array(obs_denormalized)
    
    # Define rotation matrices used to calculate the cameraUpVector according to the movement of the mobile robot
    def _Rx(self,theta):
        return np.array([[1, 0, 0],
                         [0, np.cos(theta), -np.sin(theta)],
                         [0, np.sin(theta), np.cos(theta)]])

    def _Ry(self,theta):
        return np.array([[np.cos(theta), 0, np.sin(theta)],
                         [0, 1, 0],
                         [-np.sin(theta), 0, np.cos(theta)]])

    def _Rz(self,theta):
        return np.array([[np.cos(theta), -np.sin(theta), 0],
                         [np.sin(theta), np.cos(theta), 0],
                         [0, 0, 1]])

    #sensor readings
    def _get_current_angle(self):
        # Get robot state
        pos, orn = pybullet.getBasePositionAndOrientation(self.robot_id)
        roll, pitch, yaw = pybullet.getEulerFromQuaternion(orn)
        linear_velocity, angular_velocity = pybullet.getBaseVelocity(self.robot_id)
        #roll_deg = np.degrees(roll)
        #pitch_deg = np.degrees(pitch)
        #yaw_deg = np.degrees(yaw)
        #angle_degree = pitch_deg
        angle = pitch
        return angle

    def _get_current_line_position(self):
        # Camera processing for line-following
        camera_link_pose = pybullet.getLinkState(self.robot_id, self.CAMERA_IDX)[0]
        camera_target_link_pose = pybullet.getLinkState(self.robot_id, self.CAMERA_TARGET_IDX)[0]

        mobile_robot_roll, mobile_robot_pitch, mobile_robot_yaw = pybullet.getEulerFromQuaternion(pybullet.getLinkState(self.robot_id, self.CAMERA_IDX)[1])
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
                error_steer = -width / 2  # Ujung kiri
            elif self.line_last_position == "right":
                error_steer = width / 2  # Ujung kanan
            else:
                error_steer = self.prev_error_steer  # Tetap posisi terakhir
                
        else:  # Garis terdeteksi
            error_steer = cx - (width / 2)
            # Perbarui posisi terakhir berdasarkan cx
            if cx < width / 2:
                self.line_last_position = "left"
            elif cx > width / 2:
                self.line_last_position = "right"
        self.prev_error_steer = error_steer
        line_position = error_steer
        if cx is None:
            #print(f" CX is None \r \n {cx} ", end="")
            line_position = 45.0
        #print(f" \r {self.prev_error_steer}", end=" ")
        return line_position

    def initialize_wheel_variables(self):
        global prev_left_angle, prev_right_angle, left_wheel_revolutions, right_wheel_revolutions
        prev_left_angle = 0.0
        prev_right_angle = 0.0
        left_wheel_revolutions = 0
        right_wheel_revolutions = 0

    def _get_wheel_data(self):
        global prev_left_angle, prev_right_angle, left_wheel_revolutions, right_wheel_revolutions

        wheel_radius = 0.045  # 9cm / 2

        # Dapatkan kecepatan dan sudut roda dari PyBullet
        left_wheel_state = pybullet.getJointState(self.robot_id, self.LEFT_WHEEL_JOINT_IDX)
        right_wheel_state = pybullet.getJointState(self.robot_id, self.RIGHT_WHEEL_JOINT_IDX)

        left_wheel_velocity = left_wheel_state[1]
        right_wheel_velocity = right_wheel_state[1]
        left_wheel_angle = left_wheel_state[0]
        right_wheel_angle = right_wheel_state[0]

        # Hitung perubahan sudut roda
        delta_left_angle = left_wheel_angle - prev_left_angle
        delta_right_angle = right_wheel_angle - prev_right_angle

        # Koreksi untuk lompatan sudut di batas -π ke π
        if abs(delta_left_angle) > np.pi:
            delta_left_angle -= np.sign(delta_left_angle) * 2 * np.pi
        if abs(delta_right_angle) > np.pi:
            delta_right_angle -= np.sign(delta_right_angle) * 2 * np.pi

        # Update jumlah putaran roda
        left_wheel_revolutions += delta_left_angle / (2 * np.pi)
        right_wheel_revolutions += delta_right_angle / (2 * np.pi)

        # Simpan sudut saat ini untuk langkah berikutnya
        prev_left_angle = left_wheel_angle
        prev_right_angle = right_wheel_angle

        # Hitung kecepatan linier
        avg_angular_velocity = (left_wheel_velocity + right_wheel_velocity) / 2
        linear_speed = avg_angular_velocity * wheel_radius  # v = ω * r
        linear_left_wheel = left_wheel_velocity * wheel_radius
        linear_right_wheel = right_wheel_velocity * wheel_radius

        return linear_speed, linear_left_wheel, linear_right_wheel, left_wheel_revolutions, right_wheel_revolutions

    def _get_first_obs(self):
        linear_speed, linear_left_wheel, linear_right_wheel, left_wheel_revolutions, right_wheel_revolutions = self._get_wheel_data()
        pitch = self._get_current_angle()
        line = self._get_current_line_position()
        yaw = line 
        obs = np.array([self.target_pitch, self.target_speed, pitch, yaw, linear_speed, linear_left_wheel, linear_right_wheel, self.left_motor_power, self.right_motor_power, left_wheel_revolutions, right_wheel_revolutions])
        return obs

    def _get_obs(self):
        # pitch, yaw_line, x_dot_motor_linear_speed,  motor_left_speed, motor_right_speed
        linear_speed, linear_left_wheel, linear_right_wheel, left_wheel_revolutions, right_wheel_revolutions = self._get_wheel_data()
        pitch = self._get_current_angle()
        line = self._get_current_line_position()
        yaw = line

        obs_temp = np.array([self.target_pitch, self.target_speed, pitch, yaw, linear_speed, linear_left_wheel, linear_right_wheel, self.left_motor_power, self.right_motor_power, left_wheel_revolutions, right_wheel_revolutions])
        obs = np.concatenate((obs_temp, self.previous_state))
        
        self.previous_state = obs_temp
        obs = self._normalize_obs(obs)
        return obs

    def _normalize_obs(self, obs):
        return 2 * (obs - self.obs_min) / (self.obs_max - self.obs_min) - 1

    def _denormalize_1d(self, value, min_value, max_value):
        return value * (max_value - min_value) / 2 + (max_value + min_value) / 2

    def _get_reward(self):
        # Unpack state
        target_pitch, target_speed, pitch, yaw, linear_speed, _, _, left_motor_power, right_motor_power, *_ = self.observation_from_step

        # Base reward
        reward = 0.0  # Reward kecil per step untuk longevity
        pitch_error = target_pitch - pitch
        motor_power = (left_motor_power + right_motor_power)/2
        
        # 1. Pitch Stability (Prioritas Utama)
        reward -= 0.1 * abs(pitch_error **2)  # Range [0,2]
        
        # 2. Yaw Tracking (Prioritas Kedua)
        yaw_error = abs(yaw)
        reward -= 1 * (yaw_error **2) # Range [0,1.5]

        linear_speed_error = abs(target_speed - linear_speed)
        reward -= 0.01 * (linear_speed_error **2) 
        
        #reward += 0.01 * self.step_counter
        #reward += 0.1 if abs(pitch_error) < 0.05 else -0.0
        #reward += 0.1 if abs(motor_power) < 0.25 else -0.0

        return reward

    def _get_info(self):
        return {"step_count": self.step_counter}

    def render(self):
        if self.render_mode == "human" and self._physics_client_id >= 0 and not self.rendered_status:
            #self._bullet_client.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)
            #self._bullet_client.configureDebugVisualizer(pybullet.COV_ENABLE_SHADOWS, 0)
            self._bullet_client.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=45, cameraPitch=-45, cameraTargetPosition=[0, 0, 0])
            self.rendered_status = True
            return None

    def close(self):
        if self._physics_client_id >= 0:
            self._bullet_client.disconnect()
            self._physics_client_id = -1
