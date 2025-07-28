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
    '''
        self-balancing mode = try to balance on the point
        waypoint mode = try to balance and go to target
        line-following mode = try to balance and follow the line
        self-driving mode = try to balance and go    to somewhere by the lidar
    
    Action : Discrete (42| 21 Left, 21 Right), Continuous (2 | Left Right | -255 to 255)
    State(Observation) : (24| prev_roll, prev_pitch, prev_yaw, prev_omega_x, prev_omega_y, prev_omega_z, prev_x, prev_y, prev_z, prev_x_dot, prev_y_dot, prev_z_dot, roll, pitch, yaw, omega_x, omega_y, omega_z, x, y, z, x_dot, y_dot, z_dot)

    '''
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    def __init__(self,
                render_mode=None,
                robot_angle_limit=45,
                action_type="continuous",
                max_velocity=3000,
                truncation_steps=1000,
                debug_info = False
                ):

        super(TwsbrEnv, self).__init__()

        
        # Bottom camera settings
        self.projection_matrix = pybullet.computeProjectionMatrixFOV(fov=160.0, aspect=1.0, nearVal=0.007, farVal=10)
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

        # pitch, yaw, x_dot_motor_linear_speed, motor_left_speed, motor_right_speed
        state_limit = np.array([180.0, 180.0, 100, 100.0, 100.0])
        
        self.observation_space = spaces.Box(low=-np.ones(state_limit.shape), high=np.ones(state_limit.shape)) 
        self.action_space = spaces.Discrete(len(self.discrete_velocity)*2) if action_type == "discrete" else spaces.Box(low=-1.0, high=1.0, shape=(2,)) # normalize version
        
        self.obs_min = np.array([-180.0, -180.0, -100.0, -100.0, -100.0])
        self.obs_max = np.array([180.0, 180.0, 100.0, 100.0, 100.0])
        
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
                self._bullet_client = bullet_client.BulletClient(pybullet.GUI, options="--width=1080 --height=720")
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
        self.plane_id = self._bullet_client.loadURDF("plane.urdf", basePosition=[8.22, 4.5, 0.0], globalScaling=5.0)
        
        self.tex_uid =  self._bullet_client.loadTexture(self.texture)
        self._bullet_client.changeVisualShape(self.plane_id, -1, textureUniqueId=self.tex_uid)
        
        #self.tex_file_name = os.path.join(os.path.abspath(os.path.dirname(__file__)), "texture", "line_trace_ground.png")
        #self.tex_uid = self._bullet_client.loadTexture(self.tex_file_name)
        #self._bullet_client.changeVisualShape(self.plane_id, -1, textureUniqueId=self.tex_uid)
        self._bullet_client.changeDynamics(self.plane_id, -1, lateralFriction=1.0, spinningFriction=0.5)
        self.urdf_file_name = os.path.join(os.path.abspath(os.path.dirname(__file__)), "urdf", "twsbr_v1.urdf")
      
        if self.render_mode == "human":
            start_position, start_orientation = [0.0, 0.0, 0.0], pybullet.getQuaternionFromEuler([0, 0, math.radians(90)]) 
        else:  #for training, give it more challenge         
            #start_position, start_orientation = [0.0, 0.0, 0.0], pybullet.getQuaternionFromEuler([0, random.uniform(-np.pi/8, np.pi/8), random.uniform(-np.pi, np.pi)])
            start_position, start_orientation = [0.0, 0.0, 0.0], pybullet.getQuaternionFromEuler([0, random.uniform(-np.pi/8, np.pi/8), random.uniform(-np.pi, np.pi)]) 

        self.robot_id = self._bullet_client.loadURDF(self.urdf_file_name, basePosition=start_position, baseOrientation=start_orientation)
        self._set_robot_dynamics()

    def _set_robot_dynamics(self):
        for joint in [0, 1]:
            self._bullet_client.changeDynamics(self.robot_id, joint, lateralFriction=1.0, spinningFriction=0.5)
            self._bullet_client.setJointMotorControl2(self.robot_id, joint, pybullet.VELOCITY_CONTROL, force=0)

    def step(self, action):
        left_wheel_velocity, right_wheel_velocity = self._apply_action(action)

        self._bullet_client.setJointMotorControl2(self.robot_id, 0, pybullet.VELOCITY_CONTROL, targetVelocity=left_wheel_velocity)
        self._bullet_client.setJointMotorControl2(self.robot_id, 1, pybullet.VELOCITY_CONTROL, targetVelocity=right_wheel_velocity)
        self._bullet_client.stepSimulation()
        
        self.left_motor_power = left_wheel_velocity
        self.right_motor_power = right_wheel_velocity
        observation, reward, info = self._get_obs(), self._get_reward(), self._get_info()
        self.step_counter += 1
        truncated = True if self.step_counter >= self.truncation_steps else False
        terminated = True if abs(self._denormalize_1d(observation[0], -180, 180)) > self.robot_angle_limit else False
        #terminated = True if abs(observation[1]) > self.robot_angle_limit else False
        
        if truncated==True:
            info["is_success"] = True
            reward += 10
        if terminated==True:
            info["is_success"] = False
            reward -= 10
        return observation.astype(np.float32), reward, terminated, truncated, info

    def _apply_action(self, action):
        if self._action_type == "discrete":
            if action < len(self.discrete_velocity)/2:
                left_velocity = self.discrete_velocity[action]
                right_velocity = 0
            else:
                left_velocity = 0
                right_velocity = self.discrete_velocity[action]
            return left_velocity, right_velocity

        elif self._action_type == "continuous":
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
        roll_deg = np.degrees(roll)
        pitch_deg = np.degrees(pitch)
        yaw_deg = np.degrees(yaw)
        angle_degree = pitch_deg
        angle = np.array([pitch_deg, yaw_deg])
        return angle.astype(np.float32)

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

        width, height, rgb_img, depth_img, seg_img = pybullet.getCameraImage(8, 8, view_matrix, self.projection_matrix)
        img = np.reshape(rgb_img, (height, width, 4))
        gray = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGBA2RGB), cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 64, 255, cv2.THRESH_BINARY_INV)
        moments = cv2.moments(binary)

        cx = None if moments["m00"] == 0 else moments["m10"] / moments["m00"]
        error_steer = 10.0 if cx is None else cx - (width / 2)
        line_position = error_steer
        return line_position

    def _get_current_linear_speed(self):
        wheel_radius = 0.045  # 9cm / 2
        left_wheel_velocity = pybullet.getJointState(self.robot_id, self.LEFT_WHEEL_JOINT_IDX)[1]
        right_wheel_velocity = pybullet.getJointState(self.robot_id, self.RIGHT_WHEEL_JOINT_IDX)[1]
        avg_angular_velocity = (left_wheel_velocity + right_wheel_velocity) / 2
        linear_left_wheel = left_wheel_velocity * wheel_radius
        linear_right_wheel = right_wheel_velocity * wheel_radius
        linear_speed = avg_angular_velocity * wheel_radius  # v = ω * r
        return linear_speed, linear_left_wheel, linear_right_wheel

    def _get_first_obs(self):
        pos, orientation = self._bullet_client.getBasePositionAndOrientation(self.robot_id)
        roll, pitch, yaw = np.degrees(pybullet.getEulerFromQuaternion(orientation))
        vel_linear, vel_angular = self._bullet_client.getBaseVelocity(self.robot_id)
        obs = np.array([roll, pitch, yaw, *vel_angular, *pos, *vel_linear])
        return obs

    def _get_obs(self):
        # pitch, x_dot_motor_linear_speed, yaw_line, motor_left_speed, motor_right_speed
        pitch, yaw = self._get_current_angle()
        #line = self._get_current_line_position()
        speed, left_speed, right_speed = self._get_current_linear_speed()
        obs = np.array([pitch, yaw, speed, left_speed, right_speed])
        return self._normalize_obs(obs)
    
    def _normalize_obs(self, obs):
        return 2 * (obs - self.obs_min) / (self.obs_max - self.obs_min) - 1

    def _denormalize_1d(self, value, min_value, max_value):
        return value * (max_value - min_value) / 2 + (max_value + min_value) / 2

    def _get_reward(self):
        pitch, yaw, speed, left_speed, right_speed = self._get_obs()
        #roll, pitch, yaw, roll_dot, pitch_dot, yaw_dot, x, y, z, x_dot, y_dot, z_dot = self._get_obs()
        
        reward = 0
        # Stabilize and optimal power in zero point
        reward += -abs(pitch)
        reward += -abs(yaw)
        #reward += abs(speed) - 0.01

        reward += 0.25 if abs(pitch) < 0.05 else -0.25
        reward += 0.25 if abs(yaw) < 0.5 else -0.25

        #reward += 0.125 if (left_speed > 0) else -0.05
        #reward += 0.125 if (right_speed > 0) else -0.05

        return reward

    def _get_info(self):
        return {"step_count": self.step_counter}

    def render(self):
        if self.render_mode == "human" and self._physics_client_id >= 0 and not self.rendered_status:
            self._bullet_client.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)
            self._bullet_client.configureDebugVisualizer(pybullet.COV_ENABLE_SHADOWS, 0)
            self._bullet_client.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=0, cameraPitch=-15, cameraTargetPosition=[0, 0, 0])
            self.rendered_status = True
            return None

    def close(self):
        if self._physics_client_id >= 0:
            self._bullet_client.disconnect()
            self._physics_client_id = -1
