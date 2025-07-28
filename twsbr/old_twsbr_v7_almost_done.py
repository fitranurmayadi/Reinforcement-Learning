from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
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
                robot_version="RV-1",
                render_mode=None,          
                target_mode="self-balancing",
                observation_type="OB-24",
                robot_angle_limit=75,
                target_position = [0.0, 0.0, 0.0],
                action_type="continuous",
                action_mode="AC-PWM",
                max_pwm=255,
                max_torque=0.05,
                max_velocity=3000,
                truncation_steps=1000,
                debug_info = False
                ):

        super(TwsbrEnv, self).__init__()
        assert robot_version in ["RV-1", "RV-2", "RV-3", "RV4"]
        self.robot_version=robot_version
        
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.render_fps = self.metadata["render_fps"]

        assert target_mode in ["self-balancing","waypoint","line-following","self-driving"]
        self.target_mode = target_mode
        
        self.target_position = target_position

        self.robot_angle_limit = robot_angle_limit
        
        assert action_type in ["discrete", "continuous"]
        self._action_type = action_type

        assert action_mode in ["AC-PWM", "AC-VELOCITY"]
        self.action_mode=action_mode

        
        self.max_pwm = max_pwm
        self.max_torque = max_torque
        self.max_velocity = max_velocity

        self.discrete_pwm = np.linspace(-self.max_pwm, self.max_pwm, num=21)
        self.discrete_torques = np.linspace(-self.max_torque, self.max_torque, num=21)
        self.discrete_velocity = np.linspace(-self.max_velocity, self.max_velocity, num=21)

        self.truncation_steps = truncation_steps
        self.debug_info = debug_info

        # Observation and action space
        # Observation use two state, last state and current state
        #prev_roll, prev_pitch, prev_yaw, prev_omega_x, prev_omega_y, prev_omega_z, prev_x, prev_y, prev_z, prev_x_dot, prev_y_dot, prev_z_dot, roll, pitch, yaw, omega_x, omega_y, omega_z, x, y, z, x_dot, y_dot, z_dot
        state_limit = np.array([180, 180, 180, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0 , 180, 180, 180, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
        
        # Observation by current state
        #roll, pitch, yaw, omega_x, omega_y, omega_z, x, y, z, x_dot, y_dot, z_dot
        #state_limit = np.array([180, 180, 180, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
        
        self.observation_space = spaces.Box(low=-np.ones(state_limit.shape), high=np.ones(state_limit.shape)) 
        self.action_space = spaces.Discrete(len(self.discrete_velocity)*2) if action_type == "discrete" else spaces.Box(low=-1.0, high=1.0, shape=(2,)) # normalize version
        
        self.obs_min = np.array([-180, -180, -180, -10, -10, -10, -10, -10, -10, -10, -10, -10, -180, -180, -180, -10, -10, -10, -10, -10, -10, -10, -10, -10])
        self.obs_max = np.array([180, 180, 180, 10, 10, 10, 10, 10, 10, 10, 10, 10, 180, 180, 180, 10, 10, 10, 10, 10, 10, 10, 10, 10])
        
        #self.obs_min = np.array([-180, -180, -180, -10, -10, -10, -10, -10, -10, -10, -10, -10])
        #self.obs_max = np.array([180, 180, 180, 10, 10, 10, 10, 10, 10, 10, 10, 10])
        
        self._physics_client_id = -1
        self.load_robot()
        return

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.load_robot()
        self.step_counter = 0
        self.previous_state = self._get_first_obs().astype(np.float32)
        return self._get_obs().astype(np.float32), self._get_info()

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
        #self.tex_file_name = os.path.join(os.path.abspath(os.path.dirname(__file__)), "texture", "line_trace_ground.png")
        #self.tex_uid = self._bullet_client.loadTexture(self.tex_file_name)
        #self._bullet_client.changeVisualShape(self.plane_id, -1, textureUniqueId=self.tex_uid)
        self._bullet_client.changeDynamics(self.plane_id, -1, lateralFriction=1.0, spinningFriction=0.5)
        
        
        if self.robot_version == "RV-1":
            self.urdf_file_name = os.path.join(os.path.abspath(os.path.dirname(__file__)), "urdf", "twsbr_v1.urdf")
        elif self.robot_version == "RV-2":
            self.urdf_file_name = os.path.join(os.path.abspath(os.path.dirname(__file__)), "urdf", "twsbr_v2.urdf")
        elif self.robot_version == "RV-3":
            self.urdf_file_name = os.path.join(os.path.abspath(os.path.dirname(__file__)), "urdf", "twsbr_v3.urdf")
        elif self.robot_version == "RV-4":
            self.urdf_file_name = os.path.join(os.path.abspath(os.path.dirname(__file__)), "urdf", "twsbr_v4.urdf")

        if self.render_mode == "human":
            if self.target_mode == "self-balancing":
                start_position, start_orientation = [0.0, 0.0, 0.0], pybullet.getQuaternionFromEuler([0, math.radians(random.uniform(-np.pi/8, np.pi/8)), 0]) #v1
                #self.target_position = [round(random.uniform(0, 10), 1) for _ in range(2)] + [0]
            elif self.target_mode == "waypoint":
                start_position, start_orientation = [0.0, 0.0, 0.0], pybullet.getQuaternionFromEuler([0, math.radians(random.uniform(-np.pi/8, np.pi/8)), math.radians(random.uniform(-np.pi/8, np.pi/8))]) #V2
                #self.target_position = [round(random.uniform(0, 10), 1) for _ in range(2)] + [0]
            elif self.target_mode == "self-driving":
                start_position, start_orientation = [0.0, 0.0, 0.0], pybullet.getQuaternionFromEuler([0, math.radians(random.uniform(-np.pi/8, np.pi/8)), math.radians(random.uniform(-np.pi/8, np.pi/8))]) #V3
                #self.target_position = [round(random.uniform(0, 10), 1) for _ in range(2)] + [0]
            elif self.target_mode == "line-following":
                start_position, start_orientation = [0.0, 0.0, 0.0], pybullet.getQuaternionFromEuler([0, math.radians(random.uniform(-np.pi/8, np.pi/8)), math.radians(random.uniform(-np.pi/8, np.pi/8))]) #V4
                #self.target_position = [round(random.uniform(0, 10), 1) for _ in range(2)] + [0]
        else:  #for training, give it more challenge         
            if self.target_mode == "self-balancing":
                start_position, start_orientation = [0.0, 0.0, 0.0], pybullet.getQuaternionFromEuler([0, math.radians(random.uniform(-np.pi/4, np.pi/4)), 0]) #v1
                #self.target_position = [round(random.uniform(0, 10), 1) for _ in range(2)] + [0]
            elif self.target_mode == "waypoint":
                start_position, start_orientation = [0.0, 0.0, 0.0], pybullet.getQuaternionFromEuler([0, math.radians(random.uniform(-np.pi/6, np.pi/6)), math.radians(random.uniform(-np.pi, np.pi))]) #V2
                #self.target_position = [round(random.uniform(0, 10), 1) for _ in range(2)] + [0]
            elif self.target_mode == "self-driving":
                start_position, start_orientation = [0.0, 0.0, 0.0], pybullet.getQuaternionFromEuler([0, math.radians(random.uniform(-np.pi/8, np.pi/8)), math.radians(random.uniform(-np.pi, np.pi))]) #V3
                #self.target_position = [round(random.uniform(0, 10), 1) for _ in range(2)] + [0]
            elif self.target_mode == "line-following":
                start_position, start_orientation = [0.0, 0.0, 0.0], pybullet.getQuaternionFromEuler([0, math.radians(random.uniform(-np.pi/8, np.pi/8)), math.radians(random.uniform(-np.pi, np.pi))]) #V4
                #self.target_position = [round(random.uniform(0, 10), 1) for _ in range(2)] + [0]

        self.robot_id = self._bullet_client.loadURDF(self.urdf_file_name, basePosition=start_position, baseOrientation=start_orientation)
        self._set_robot_dynamics()

    def _set_robot_dynamics(self):
        for joint in [0, 1]:
            self._bullet_client.changeDynamics(self.robot_id, joint, lateralFriction=1.0, spinningFriction=0.5)
            self._bullet_client.setJointMotorControl2(self.robot_id, joint, pybullet.VELOCITY_CONTROL, force=0)

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
        terminated = True if abs(self._denormalize_1d(observation[13], -180, 180)) > self.robot_angle_limit else False
        #terminated = True if abs(observation[1]) > self.robot_angle_limit else False
        
        if truncated==True:
            info["is_success"] = True
            reward += 50
        if terminated==True:
            info["is_success"] = False
            reward -= 50
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
            if self.action_mode == "AC-PWM":
                # Skala action (-1 to 1) menjadi PWM (-255 to 255)
                pwm_left = np.clip(action[0], -1, 1) * 255
                pwm_right = np.clip(action[1], -1, 1) * 255

                # Konversi PWM ke kecepatan berdasarkan faktor skala
                velocity_scale_factor = self.max_velocity / 255  # Menentukan konversi PWM ke kecepatan
                velocity_left = pwm_left * velocity_scale_factor
                velocity_right = pwm_right * velocity_scale_factor

                # Batasi kecepatan ke rentang max_velocity
                return (np.clip(velocity_left, -self.max_velocity, self.max_velocity),
                        np.clip(velocity_right, -self.max_velocity, self.max_velocity))
                        
            elif self.action_mode == "AC-VELOCITY":
                # Langsung konversi action ke kecepatan
                return (np.clip(action[0] * self.max_velocity, -self.max_velocity, self.max_velocity),
                        np.clip(action[1] * self.max_velocity, -self.max_velocity, self.max_velocity))
           

    def _get_first_obs(self):
        pos, orientation = self._bullet_client.getBasePositionAndOrientation(self.robot_id)
        roll, pitch, yaw = np.degrees(pybullet.getEulerFromQuaternion(orientation))
        vel_linear, vel_angular = self._bullet_client.getBaseVelocity(self.robot_id)
        obs = np.array([roll, pitch, yaw, *vel_angular, *pos, *vel_linear])
        return obs
        
    def observation_denormalize(self):
        obs_normalized = self._get_obs()
        obs_denormalized = [
            self._denormalize_1d(value, min_val, max_val)
            for value, min_val, max_val in zip(obs_normalized, self.obs_min, self.obs_max)
        ]
        print(obs_denormalized)
        return np.array(obs_denormalized)

    def _get_obs(self):
        pos, orientation = self._bullet_client.getBasePositionAndOrientation(self.robot_id)
        roll, pitch, yaw = np.degrees(pybullet.getEulerFromQuaternion(orientation))
        vel_linear, vel_angular = self._bullet_client.getBaseVelocity(self.robot_id)
        obs = np.array([roll, pitch, yaw, *vel_angular, *pos, *vel_linear])
        obs_temp = np.array([roll, pitch, yaw, *vel_angular, *pos, *vel_linear])
        obs = np.concatenate((self.previous_state, obs_temp))
        self.previous_state = obs_temp
        return self._normalize_obs(obs)
    

    def _normalize_obs(self, obs):
        return 2 * (obs - self.obs_min) / (self.obs_max - self.obs_min) - 1

    def _denormalize_1d(self, value, min_value, max_value):
        return value * (max_value - min_value) / 2 + (max_value + min_value) / 2

    def _get_reward(self):
        prev_roll, prev_pitch, prev_yaw, prev_omega_x, prev_omega_y, prev_omega_z, prev_x, prev_y, prev_z, prev_x_dot, prev_y_dot, prev_z_dot, roll, pitch, yaw, omega_x, omega_y, omega_z, x, y, z, x_dot, y_dot, z_dot = self._get_obs()
        #roll, pitch, yaw, roll_dot, pitch_dot, yaw_dot, x, y, z, x_dot, y_dot, z_dot = self._get_obs()

        target_angle_roll = 0
        target_angle_pitch = 0
        target_angle_yaw = 0

        error_angle_pitch = target_angle_pitch - abs(pitch)
        error_angle_yaw = target_angle_yaw - abs(yaw)

        error_position_x = abs(self.target_position[0]/10) - abs(x)
        error_position_y = abs(self.target_position[1]/10) - abs(y)
        state_action_power = abs((abs(self.left_motor_power) + abs(self.right_motor_power))/2) / self.max_velocity
        
        reward = 0
        
        if self.target_mode == "self-balancing":
            # REWARD version 1
            # Stabilize and optimal power in zero point
            reward -= abs(error_position_x)
            reward -= abs(pitch)
            reward -= abs(state_action_power)**2
            reward += 0.5 if abs(abs(error_position_x) < 0.001) else 0.0
            reward += 0.3 if abs(abs(pitch) < 0.005) else 0.0
            reward += 0.2 if abs(abs(state_action_power) < 0.125) else 0.0

        elif self.target_mode == "waypoint":
            #REWARD version 2
            # Stabilize and go to target position
            reward += max(0.1 - (abs(error_angle_pitch))**2, -0.1) # Reduced scaling for angle error
            reward += max(0.1 - (abs(error_angle_yaw)) **2, -0.1) # Reduced scaling for angle error
            reward += max(0.1 - (abs(error_position_x) **2), -0.1) # Reduced scaling for position error
            reward += max(0.1 - (abs(error_position_y) **2), -0.1) # Reduced scaling for position error
            reward += max(0.1 - (abs(state_action_power) **2), -0.1) # Reduce

            reward += 0.1 if abs(error_angle_pitch) < 0.05 else -0.01
            reward += 0.1 if abs(error_angle_yaw) < 0.05 else -0.01
            reward += 0.1 if abs(error_position_x) < 0.01 else -0.01
            reward += 0.1 if abs(error_position_y) < 0.01 else -0.01
            reward += 0.1 if abs(state_action_power) < 0.25 else -0.01 # Minor penalty for high power usage
        elif self.target_mode == "self-driving":
            #REWARD version 3
            #reward += 0.2 if abs(pitch) < abs(prev_pitch) else -0.2
            #reward += 0.2 if abs(omega_y) < abs(prev_omega_y) else -0.2
            #reward += 0.2 if abs(x) < abs(prev_x) else -0.2
            reward += 0.2 if abs(x_dot) < abs(prev_x_dot) else -0.2
        elif self.target_mode == "line-following":
            #REWARD version 4
            #reward += 0.2 if abs(pitch) < abs(prev_pitch) else -0.2
            #reward += 0.2 if abs(omega_y) < abs(prev_omega_y) else -0.2
            #reward += 0.2 if abs(x) < abs(prev_x) else -0.2
            reward += 0.2 if abs(x_dot) < abs(prev_x_dot) else -0.2
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
