import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet
import pybullet_data
import time
import os
import random
import math
from pybullet_utils import bullet_client

class StarshipLanderEnv(gym.Env):
    """
    Environment Precision Landing untuk roket Starship dengan 3 aktuator:
      - thrust utama
      - 2 kontrol gimbal (sumbu Y dan X)
      
    Observasi: 13 dimensi (posisi (3), orientasi Euler (3), kecepatan linier (3),
    kecepatan angular (3), bahan bakar (1))
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 100}

    def __init__(self, render_mode=None, max_thrust=100.0, truncation_steps=5000, planet="earth"):
        super(StarshipLanderEnv, self).__init__()
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.render_fps = self.metadata["render_fps"]
        self.max_thrust = max_thrust
        self.initial_fuel = truncation_steps * 1000  # skala bahan bakar awal
        self.fuel = self.initial_fuel
        self.truncation_steps = truncation_steps
        self.sim_time = 1 / self.render_fps

        # Parameter gravitasi berdasarkan planet
        self.planet = planet.lower()
        if self.planet == "earth":
            self.gravity = -9.8
        elif self.planet == "moon":
            self.gravity = -1.62
        elif self.planet == "mars":
            self.gravity = -3.711
        else:
            raise ValueError("Planet harus 'earth', 'moon', atau 'mars'")

        # Definisi batas observasi (13 dimensi)
        pos_low, pos_high = np.array([-25, -25, -25]), np.array([25, 25, 25])
        orient_low, orient_high = -np.pi * np.ones(3), np.pi * np.ones(3)
        lin_vel_low, lin_vel_high = -10 * np.ones(3), 10 * np.ones(3)
        ang_vel_low, ang_vel_high = -10 * np.ones(3), 10 * np.ones(3)
        fuel_low, fuel_high = np.array([0]), np.array([self.initial_fuel])
        self.obs_min = np.concatenate([pos_low, orient_low, lin_vel_low, ang_vel_low, fuel_low])
        self.obs_max = np.concatenate([pos_high, orient_high, lin_vel_high, ang_vel_high, fuel_high])
        
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(13,), dtype=np.float32)
        # Aksi: 3-dimensi (thrust, gimbal_y, gimbal_x) dengan nilai dalam [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        
        self._bullet_client = None
        self.lander_id = None
        self.thruster_index = None
        self.gimbal_y_index = None
        self.gimbal_x_index = None
        # Indeks untuk kaki dan sensor (sesuai URDF baru)
        self.leg_1_index = None
        self.leg_1_sensor_index = None
        self.leg_2_index = None
        self.leg_2_sensor_index = None
        self.leg_3_index = None
        self.leg_3_sensor_index = None

        self._load_lander()

    def _load_lander(self):
        if self._bullet_client is None:
            if self.render_mode == "human":
                self._bullet_client = bullet_client.BulletClient(pybullet.GUI, options="--width=960 --height=1080")
                self._bullet_client.configureDebugVisualizer(self._bullet_client.COV_ENABLE_GUI, 0)
            else:
                self._bullet_client = bullet_client.BulletClient(pybullet.DIRECT)
        self._init_physics_client()

    def _init_physics_client(self):
        self._bullet_client.resetSimulation()
        self._bullet_client.setGravity(0, 0, self.gravity)
        self._bullet_client.setTimeStep(1.0 / self.render_fps)
        self._bullet_client.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.plane_id = self._bullet_client.loadURDF("plane.urdf")
        
        # Posisi awal: random di atas bidang pendaratan
        base_altitude = 25
        start_x = random.uniform(-25, 25)
        start_y = random.uniform(-25, 25)
        start_z = base_altitude
        start_pos = [start_x, start_y, start_z]
        start_orient = pybullet.getQuaternionFromEuler([0, 0, 0])

        # Asumsikan file URDF berada di folder 'urdf' relatif terhadap file ini
        urdf_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "urdf", "starship.urdf")
        self.lander_id = self._bullet_client.loadURDF(urdf_path,
                                                      basePosition=start_pos,
                                                      baseOrientation=start_orient,
                                                      flags=pybullet.URDF_USE_INERTIA_FROM_FILE)

        num_joints = self._bullet_client.getNumJoints(self.lander_id)
        for i in range(num_joints):
            joint_info = self._bullet_client.getJointInfo(self.lander_id, i)
            joint_name = joint_info[1].decode("utf-8")
            link_name = joint_info[12].decode("utf-8")
            if joint_name == "gimbal_x":
                self.gimbal_x_index = i
            if link_name == "thruster":
                self.thruster_index = i
            if link_name == "gimbal_y_frame":
                self.gimbal_y_index = i
            if joint_name == "leg_1_joint":
                self.leg_1_index = i
            if joint_name == "leg_1_sensor_joint":
                self.leg_1_sensor_index = i
            if joint_name == "leg_2_joint":
                self.leg_2_index = i
            if joint_name == "leg_2_sensor_joint":
                self.leg_2_sensor_index = i
            if joint_name == "leg_3_joint":
                self.leg_3_index = i
            if joint_name == "leg_3_sensor_joint":
                self.leg_3_sensor_index = i

        # Jika ada link yang belum ditemukan, cetak semua joint untuk membantu debug
        missing = []
        if self.thruster_index is None:
            missing.append("thruster")
        if self.gimbal_y_index is None:
            missing.append("gimbal_y_frame")
        if self.gimbal_x_index is None:
            missing.append("gimbal_x")
        if self.leg_1_index is None:
            missing.append("leg_1")
        if self.leg_1_sensor_index is None:
            missing.append("leg_1_sensor")
        if self.leg_2_index is None:
            missing.append("leg_2")
        if self.leg_2_sensor_index is None:
            missing.append("leg_2_sensor")
        if self.leg_3_index is None:
            missing.append("leg_3")
        if self.leg_3_sensor_index is None:
            missing.append("leg_3_sensor")
            
        if missing:
            print("Daftar joint/link yang terdeteksi:")
            for i in range(num_joints):
                info = self._bullet_client.getJointInfo(self.lander_id, i)
                print(f"Joint {i}: joint_name={info[1].decode('utf-8')}, link_name={info[12].decode('utf-8')}")
            raise RuntimeError("Tidak dapat menemukan beberapa link/joint penting pada URDF starship: " + ", ".join(missing))

    def _apply_dynamics(self):
        # Contoh drag sederhana
        lin_vel, ang_vel = self._bullet_client.getBaseVelocity(self.lander_id)
        drag_coeff = 0.5
        drag_force = -drag_coeff * np.array(lin_vel)
        self._bullet_client.applyExternalForce(self.lander_id, -1, drag_force, [0, 0, 0], self._bullet_client.LINK_FRAME)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._load_lander()
        self.step_counter = 0
        self.fuel = self.initial_fuel
        current_obs = self._get_obs()
        return current_obs, self._get_info()

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._apply_thruster_controls(action)
        self._apply_dynamics()
        self._bullet_client.stepSimulation()
        time.sleep(1.0 / self.render_fps)
        
        self.step_counter += 1

        # Update penggunaan bahan bakar (contoh sederhana)
        thrust_usage = abs(action[0]) * 0.1
        self.fuel = max(0, self.fuel - thrust_usage * (self.initial_fuel / 1000.0))
        
        current_obs = self._get_obs()
        reward, terminated, truncated = self._get_reward(current_obs, action)
        info = self._get_info()

        return current_obs, float(reward), terminated, truncated, info

    def _apply_thruster_controls(self, action):
        # Interpretasi aksi:
        # action[0]: thrust utama, diskalakan ke [0, max_thrust]
        # action[1]: kontrol gimbal_y, dikalibrasi ke [-0.5, 0.5] rad
        # action[2]: kontrol gimbal_x, dikalibrasi ke [-0.5, 0.5] rad
        thrust_cmd = max(0, (action[0] + 1) / 2) * self.max_thrust
        desired_gimbal_y = action[1] * 0.5
        desired_gimbal_x = action[2] * 0.5
        
        self._bullet_client.setJointMotorControl2(
            bodyUniqueId=self.lander_id,
            jointIndex=self.gimbal_y_index,
            controlMode=pybullet.POSITION_CONTROL,
            targetPosition=desired_gimbal_y,
            force=10)
        self._bullet_client.setJointMotorControl2(
            bodyUniqueId=self.lander_id,
            jointIndex=self.gimbal_x_index,
            controlMode=pybullet.POSITION_CONTROL,
            targetPosition=desired_gimbal_x,
            force=10)
        
        # Tentukan arah gaya berdasarkan orientasi thruster
        thruster_state = self._bullet_client.getLinkState(self.lander_id, self.thruster_index)
        thruster_world_pos = thruster_state[0]
        thruster_world_orient = thruster_state[1]
        rot_matrix = np.array(self._bullet_client.getMatrixFromQuaternion(thruster_world_orient)).reshape(3, 3)
        thruster_direction = rot_matrix[:, 2]  # sumbu lokal Z
        thrust_force = (thrust_cmd * thruster_direction).tolist()
        self._bullet_client.applyExternalForce(self.lander_id, -1, thrust_force, thruster_world_pos, pybullet.WORLD_FRAME)

    def _get_obs(self):
        pos, orient = self._bullet_client.getBasePositionAndOrientation(self.lander_id)
        lin_vel, ang_vel = self._bullet_client.getBaseVelocity(self.lander_id)
        euler_orient = pybullet.getEulerFromQuaternion(orient)
        
        raw_obs = np.array(list(pos) + list(euler_orient) + list(lin_vel) + list(ang_vel) + [self.fuel], dtype=np.float32)
        normalized_obs = 2 * (raw_obs - self.obs_min) / (self.obs_max - self.obs_min) - 1
        return np.clip(normalized_obs, -1, 1).astype(np.float32)

    def _get_reward(self, obs, action):
        pos = obs[0:3]
        orient = obs[3:6]
        lin_vel = obs[6:9]
        ang_vel = obs[9:12]
        fuel = obs[12]
        
        # Target landing di titik [0, 0, 0]
        target_pos = np.array([0, 0, 0])
        target_vel = np.array([0, 0, 0])
        
        reward = ( 3 - 1 * np.linalg.norm(target_pos - pos)**2
                  + (0.3 - 10 * np.linalg.norm(target_vel - lin_vel)**2)
                  + (0.03 - 100 * np.linalg.norm(orient)**2)
                  + (0.1 - 0.1 * np.linalg.norm(ang_vel)**2))
        
        reward -= 0.3 * ((action[0] + 1) / 2) ** 2
        reward -= 1

        # Cek kontak dengan bidang pendaratan
        contacts = self._bullet_client.getContactPoints(bodyA=self.lander_id, bodyB=self.plane_id)
        allowed_links = {self.leg_1_index, self.leg_1_sensor_index,
                         self.leg_2_index, self.leg_2_sensor_index,
                         self.leg_3_index, self.leg_3_sensor_index}
        crash_detected = False
        safe_contact = False
        for c in contacts:
            if c[3] in allowed_links:
                safe_contact = True
            else:
                crash_detected = True
                break

        if np.linalg.norm(ang_vel) > 0.2:
            crash_detected = True

        if crash_detected:
            #print("Crash detected!")
            reward -= 1000.0
            return reward, True, False

        if safe_contact and np.linalg.norm(lin_vel) < 0.5 and np.linalg.norm(ang_vel) < 0.1:
            #print("Landing successful!")
            reward += 500
            return reward, True, False

        truncated = True if self.fuel <= 0 or self.step_counter >= self.truncation_steps else False
        if truncated and self.step_counter >= self.truncation_steps:
            #print("Episode truncated due to step limit!")
            reward -= 1000.0
            return reward, True, True

        return reward, False, truncated

    def _get_info(self):
        return {"step_count": self.step_counter}
    
    def render(self):
        if self.render_mode == "human":
            lander_pos, _ = self._bullet_client.getBasePositionAndOrientation(self.lander_id)
            self._bullet_client.resetDebugVisualizerCamera(
                cameraDistance=2,
                cameraYaw=45,
                cameraPitch=-45,
                cameraTargetPosition=lander_pos)
            return None

    def close(self):
        if self._bullet_client is not None:
            self._bullet_client.disconnect()
