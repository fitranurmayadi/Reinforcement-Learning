from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import random
import math
import os
import pybullet
import pybullet_data
from pybullet_utils import bullet_client

class TwsbrEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 100}

    def __init__(self,
                urdf_path="urdf/twsbr.urdf",
                render_mode=None,
                target_position = [0.0, 0.0, 0.0],
                action_type="continuous",
                robot_angle_limit=45,
                max_torque=0.05,
                max_velocity=255,
                truncation_steps=1000):

        super(TwsbrEnv, self).__init__()
        self.robot_angle_limit = robot_angle_limit
        self.truncation_steps = truncation_steps
        self.max_torque = max_torque
        self.max_velocity = max_velocity
        self.render_fps = self.metadata["render_fps"]

        self.discrete_torques = np.linspace(-self.max_torque, self.max_torque, num=21)
        self.discrete_velocity = np.linspace(-self.max_velocity, self.max_velocity, num=21)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        assert action_type in ["discrete", "continuous"]
        self._action_type = action_type

        
        # Observation and action space
        # Observation use two state, last state and current state
        #prev_roll, prev_pitch, prev_yaw, prev_omega_x, prev_omega_y, prev_omega_z, prev_x, prev_y, prev_z, prev_x_dot, prev_y_dot, prev_z_dot, roll, pitch, yaw, omega_x, omega_y, omega_z, x, y, z, x_dot, y_dot, z_dot
        #state_limit = np.array([np.pi, np.pi, np.pi, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0 , np.pi, np.pi, np.pi, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
        
        # Observation by current state
        #roll, pitch, yaw, omega_x, omega_y, omega_z, x, y, z, x_dot, y_dot, z_dot
        state_limit = np.array([np.pi, np.pi, np.pi, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
        
        self.observation_space = spaces.Box(low=-np.ones(state_limit.shape), high=np.ones(state_limit.shape))
        self.action_space = spaces.Discrete(42) if action_type == "discrete" else spaces.Box(low=-1.0, high=1.0, shape=(2,))

        #self.obs_min = np.array([-180, -180, -180, -10, -10, -10, -10, -10, -10, -10, -10, -10, -180, -180, -180, -10, -10, -10, -10, -10, -10, -10, -10, -10])
        #self.obs_max = np.array([180, 180, 180, 10, 10, 10, 10, 10, 10, 10, 10, 10, 180, 180, 180, 10, 10, 10, 10, 10, 10, 10, 10, 10])
        
        self.obs_min = np.array([-180, -180, -180, -10, -10, -10, -10, -10, -10, -10, -10, -10])
        self.obs_max = np.array([180, 180, 180, 10, 10, 10, 10, 10, 10, 10, 10, 10])
        
        self._physics_client_id = -1
        self.load_robot()
        return

    def load_robot(self):
        
        if self._physics_client_id < 0:
            if self.render_mode == "human":
                self._bullet_client = bullet_client.BulletClient(pybullet.GUI, options="--width=960 --height=540")
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
        self.plane_id = self._bullet_client.loadURDF("plane.urdf")
        self._bullet_client.changeDynamics(self.plane_id, -1, lateralFriction=1.0, spinningFriction=1.0)
        
        self.urdf_file_name = os.path.join(os.path.abspath(os.path.dirname(__file__)), "urdf", "twsbr.urdf")
        if self.render_mode == "human":
            start_position, start_orientation = [0.0, 0.0, 0.0], pybullet.getQuaternionFromEuler([0, math.radians(random.uniform(-22.5, 22.5)), 0])
            self.target_position = target_position
        else:
            #for training, give it more challange
            start_position, start_orientation = [0.0, 0.0, 0.0], pybullet.getQuaternionFromEuler([0, math.radians(random.uniform(-44, 44)), 0])
            self.target_position = [round(random.uniform(0, 10), 1) for _ in range(2)] + [0]
        self.robot_id = self._bullet_client.loadURDF(self.urdf_file_name, basePosition=start_position, baseOrientation=start_orientation)
        self._set_robot_dynamics()

    def _set_robot_dynamics(self):
        for joint in [0, 1]:
            self._bullet_client.changeDynamics(self.robot_id, joint, lateralFriction=1.0, spinningFriction=1.0)
            self._bullet_client.setJointMotorControl2(self.robot_id, joint, pybullet.VELOCITY_CONTROL, force=0)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.load_robot()
        self.step_counter = 0
        #self.previous_state = self._get_first_obs().astype(np.float32)
        return self._get_obs().astype(np.float32), self._get_info()

    def step(self, action):
        left_wheel_velocity, right_wheel_velocity = self._apply_action(action)
        self._bullet_client.setJointMotorControl2(self.robot_id, 0, pybullet.VELOCITY_CONTROL, targetVelocity=left_wheel_velocity, force=self.max_torque)
        self._bullet_client.setJointMotorControl2(self.robot_id, 1, pybullet.VELOCITY_CONTROL, targetVelocity=right_wheel_velocity, force=self.max_torque)
        self._bullet_client.stepSimulation()
        
        self.left_motor_power = left_wheel_velocity
        self.right_motor_power = right_wheel_velocity
        observation, reward, info = self._get_obs(), self._get_reward(), self._get_info()
        
        self.step_counter += 1
        truncated = True if self.step_counter >= self.truncation_steps else False
        terminated = True if abs(self._denormalize_1d(observation[1], -180, 180)) > self.robot_angle_limit else False
        if truncated==True:
            info["is_success"] = True
            reward += 1
        if terminated==True:
            info["is_success"] = False
            reward -= 1

        return observation.astype(np.float32), reward, terminated, truncated, info

    def _apply_action(self, action):
        if self._action_type == "discrete" and len(action) == 42:
            left_index, right_index = np.argmax(action[:21]), np.argmax(action[21:])
            return self.discrete_torques[left_index], self.discrete_torques[right_index]
        elif self._action_type == "continuous":
            return (np.clip(action[0] * self.max_velocity, -self.max_velocity, self.max_velocity),
                    np.clip(action[1] * self.max_velocity, -self.max_velocity, self.max_velocity))

    def _get_first_obs(self):
        pos, orientation = self._bullet_client.getBasePositionAndOrientation(self.robot_id)
        roll, pitch, yaw = np.degrees(pybullet.getEulerFromQuaternion(orientation))
        vel_linear, vel_angular = self._bullet_client.getBaseVelocity(self.robot_id)
        obs = np.array([roll, pitch, yaw, *vel_angular, *pos, *vel_linear])
        return obs

    def _get_obs(self):
        pos, orientation = self._bullet_client.getBasePositionAndOrientation(self.robot_id)
        roll, pitch, yaw = np.degrees(pybullet.getEulerFromQuaternion(orientation))
        vel_linear, vel_angular = self._bullet_client.getBaseVelocity(self.robot_id)
        obs = np.array([roll, pitch, yaw, *vel_angular, *pos, *vel_linear])
        #obs_temp = np.array([roll, pitch, yaw, *vel_angular, *pos, *vel_linear])
        #obs = np.concatenate((self.previous_state, obs_temp))
        #self.previous_state = obs_temp
        return self._normalize_obs(obs)

    def _normalize_obs(self, obs):
        return 2 * (obs - self.obs_min) / (self.obs_max - self.obs_min) - 1

    def _denormalize_1d(self, value, min_value, max_value):
        return value * (max_value - min_value) / 2 + (max_value + min_value) / 2

    def _get_reward(self):
        #prev_roll, prev_pitch, prev_yaw, prev_omega_x, prev_omega_y, prev_omega_z, prev_x, prev_y, prev_z, prev_x_dot, prev_y_dot, prev_z_dot, roll, pitch, yaw, omega_x, omega_y, omega_z, x, y, z, x_dot, y_dot, z_dot = self._get_obs()
        roll, pitch, yaw, omega_x, omega_y, omega_z, x, y, z, x_dot, y_dot, z_dot = self._get_obs()

        target_angle = 0
        error_angle = target_angle - abs(pitch)
        error_position_x = abs(self.target_position[0]/10) - abs(x)
        error_position_y = abs(self.target_position[1]/10) - abs(y)
        state_action_power = abs((abs(self.left_motor_power) + abs(self.right_motor_power))/2) / self.max_velocity

        reward = 1
        reward += max(0.1 - abs(pitch) * 10, -2) # Smoother penalty for pitch
        #reward += max(0.1 - abs(yaw) * 100, -2)
        #reward += max(0.1 - abs(x) * 20, -2) # Smoother penalty for x position
        #reward += max(0.1 - abs(y) * 100, -2)
        reward += max(0.1 - abs(error_angle * 20), -2) # Reduced scaling for angle error
        reward += max(1 - abs(error_position_x * 10), -1) # Reduced scaling for position error
        reward += max(1 - abs(error_position_y * 10), -1) # Reduced scaling for position error

        reward += 0.2 if state_action_power < 0.125 else -0.2 # Minor penalty for high power usage
        reward += 0.1 if abs(error_position_x) < 0.01 else -0.1
        reward += 0.1 if abs(error_position_y) < 0.01 else -0.1
        
        #reward += 0.2 if abs(pitch) < abs(prev_pitch) else -0.2
        #reward += 0.2 if abs(omega_y) < abs(prev_omega_y) else -0.2
        #reward += 0.2 if abs(x) < abs(prev_x) else -0.2
        #reward += 0.2 if abs(x_dot) < abs(prev_x_dot) else -0.2
        return reward

    def _get_info(self):
        return {"step_count": self.step_counter}

    def render(self):
        if self.render_mode == "human" and self._physics_client_id >= 0 and not self.rendered_status:
            self._bullet_client.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)
            self._bullet_client.configureDebugVisualizer(pybullet.COV_ENABLE_SHADOWS, 0)
            self._bullet_client.resetDebugVisualizerCamera(cameraDistance=0.5, cameraYaw=45, cameraPitch=-22.50, cameraTargetPosition=[0, 0, 0])
            self.rendered_status = True
            return None

    def close(self):
        if self._physics_client_id >= 0:
            self._bullet_client.disconnect()
            self._physics_client_id = -1
