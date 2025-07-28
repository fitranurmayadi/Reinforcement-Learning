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

class LunarLander3DEnv(gym.Env):
    """
    Environment Lunar Lander 3D dengan 21 aktuator:
      - 1 untuk Main Thruster
      - 20 untuk RCS thrusters (4 grup x 5 nozzle per grup)
      
    Observasi: 34 dimensi (hasil konkatenasi: previous 17-dim + current 17-dim)
    Aksi: continuous, vektor 21-dimensi dalam rentang [-1, 1].

    Mode planet: 'earth', 'moon', dan 'mars' yang akan menyesuaikan parameter dynamics.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 100}

    def __init__(self, render_mode=None, max_thrust=3200.0, truncation_steps=6000, drag_coeff=0.5, wind_force=15, wind_freq=0.1, planet="moon"):
        super(LunarLander3DEnv, self).__init__()
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.render_fps = self.metadata["render_fps"]
        self.max_thrust = max_thrust
        self.initial_fuel = truncation_steps * 1000  # misal, fuel awal = jumlah langkah * skala tertentu
        self.fuel = self.initial_fuel
        self.truncation_steps = truncation_steps
        self.sim_time = 1 / self.render_fps

        # Set parameter dinamik berdasarkan mode planet
        self.planet = planet.lower()
        if self.planet == "earth":
            self.gravity = -9.8
            self.drag_coeff = drag_coeff
            self.wind_force = wind_force
            self.wind_freq = wind_freq
        elif self.planet == "moon":
            self.gravity = -1.62
            self.drag_coeff = 0.0
            self.wind_force = 0.0
            self.wind_freq = 0.0
        elif self.planet == "mars":
            self.gravity = -3.711
            self.drag_coeff = 0.3
            self.wind_force = 10
            self.wind_freq = 0.1
        else:
            raise ValueError("Planet harus 'earth', 'moon', atau 'mars'")

        # Parameter observasi asli (17 dimensi) untuk normalisasi
        pos_low, pos_high = np.array([-100, -100, -100]), np.array([100, 100, 100])
        orient_low, orient_high = -np.pi * np.ones(3), np.pi * np.ones(3)
        lin_vel_low, lin_vel_high = -10 * np.ones(3), 10 * np.ones(3)
        ang_vel_low, ang_vel_high = -10 * np.ones(3), 10 * np.ones(3)
        fuel_low, fuel_high = np.array([0]), np.array([self.initial_fuel])
        contact_low, contact_high = np.zeros(4), 100000 * np.ones(4)
        self.obs_min = np.concatenate([pos_low, orient_low, lin_vel_low, ang_vel_low, fuel_low, contact_low])
        self.obs_max = np.concatenate([pos_high, orient_high, lin_vel_high, ang_vel_high, fuel_high, contact_high])
        
        # Observasi penuh: konkatenasi dari observasi sebelumnya dan saat ini (34 dimensi)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(34,), dtype=np.float32)
        
        # Hanya mode aksi continuous (21-dimensi)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(21,), dtype=np.float32)
        
        # Setup Bullet client
        self._bullet_client = None
        self._physics_client_id = -1
        self.lander_id = None

        # --- Definisi aktuator ---
        self.main_thruster_local_position = np.array([0, 0, -0.5])  # misal, di bawah main body
        self.main_thruster_local_direction = np.array([0, 0, 1])      # mendorong ke atas

        # RCS groups: 4 grup, masing-masing 5 thruster
        self.rcs_thruster_force_scale = 0.1 * self.max_thrust
        self.rcs_groups = {
            'front': {
                'origin': np.array([0.55, 0, 0.6]),
                'offsets': [
                    np.array([0.15, 0, 0]),
                    np.array([0, -0.15, 0]),
                    np.array([0, 0.15, 0]),
                    np.array([0, 0, -0.15]),
                    np.array([0, 0, 0.15])
                ],
                'force_directions': [
                    np.array([-1, 0, 0]),
                    np.array([0, 1, 0]),
                    np.array([0, -1, 0]),
                    np.array([0, 0, 1]),
                    np.array([0, 0, -1])
                ]
            },
            'back': {
                'origin': np.array([-0.55, 0, 0.6]),
                'offsets': [
                    np.array([-0.15, 0, 0]),
                    np.array([0, -0.15, 0]),
                    np.array([0, 0.15, 0]),
                    np.array([0, 0, -0.15]),
                    np.array([0, 0, 0.15])
                ],
                'force_directions': [
                    np.array([1, 0, 0]),
                    np.array([0, 1, 0]),
                    np.array([0, -1, 0]),
                    np.array([0, 0, 1]),
                    np.array([0, 0, -1])
                ]
            },
            'left': {
                'origin': np.array([0, 0.55, 0.6]),
                'offsets': [
                    np.array([-0.15, 0, 0]),
                    np.array([0.15, 0, 0]),
                    np.array([0, 0.15, 0]),
                    np.array([0, 0, -0.15]),
                    np.array([0, 0, 0.15])
                ],
                'force_directions': [
                    np.array([1, 0, 0]),
                    np.array([-1, 0, 0]),
                    np.array([0, -1, 0]),
                    np.array([0, 0, 1]),
                    np.array([0, 0, -1])
                ]
            },
            'right': {
                'origin': np.array([0, -0.55, 0.6]),
                'offsets': [
                    np.array([-0.15, 0, 0]),
                    np.array([0.15, 0, 0]),
                    np.array([0, -0.15, 0]),
                    np.array([0, 0, -0.15]),
                    np.array([0, 0, 0.15])
                ],
                'force_directions': [
                    np.array([1, 0, 0]),
                    np.array([-1, 0, 0]),
                    np.array([0, 1, 0]),
                    np.array([0, 0, 1]),
                    np.array([0, 0, -1])
                ]
            }
        }

        # Contoh tambahan: beberapa thruster untuk debugging (jika diperlukan)
        self.thruster_local_positions = [
            np.array([0.55, 0.55, 0.6]),
            np.array([-0.55, 0.55, 0.6]),
            np.array([0.55, -0.55, 0.6]),
            np.array([-0.55, -0.55, 0.6])
        ]
        self.current_thruster_forces = np.zeros(len(self.thruster_local_positions))
        
        self._load_lander()
        self.prev_obs = np.zeros(17, dtype=np.float32)  # inisialisasi observasi sebelumnya

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
        start_pos = [random.uniform(-100, 100), random.uniform(-100, 100), 100]
        start_orient = pybullet.getQuaternionFromEuler([
            np.radians(random.uniform(-100, 100)),
            np.radians(random.uniform(-100, 100)),
            np.radians(random.uniform(-100, 100))
        ])
        urdf_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "urdf", "lunarlanderv2.urdf")
        self.lander_id = self._bullet_client.loadURDF(urdf_path,
                                                      basePosition=start_pos,
                                                      baseOrientation=start_orient)
        # Inisialisasi sensor kaki sesuai nama pada URDF
        self.leg_sensor_names = ["leg_front_left_sensor", "leg_back_right_sensor", 
                                 "leg_front_right_sensor", "leg_back_left_sensor"]
        self.leg_sensor_indices = {}
        for i in range(self._bullet_client.getNumJoints(self.lander_id)):
            joint_info = self._bullet_client.getJointInfo(self.lander_id, i)
            link_name = joint_info[12].decode("utf-8")
            if link_name in self.leg_sensor_names:
                self.leg_sensor_indices[link_name] = i
        self.prev_shaping = None

    def _apply_dynamics(self):
        # Terapkan drag dan gangguan angin
        lin_vel, ang_vel = self._bullet_client.getBaseVelocity(self.lander_id)
        drag_force = -self.drag_coeff * np.array(lin_vel)
        drag_torque = -self.drag_coeff * 0.5 * np.array(ang_vel)
        self._bullet_client.applyExternalForce(self.lander_id, -1, drag_force, [0, 0, 0], self._bullet_client.LINK_FRAME)
        self._bullet_client.applyExternalTorque(self.lander_id, -1, drag_torque, self._bullet_client.LINK_FRAME)
        noise = np.random.uniform(-20, 20, 3)
        wind_force = self.wind_force * np.sin(2 * np.pi * self.wind_freq * self.sim_time) + noise
        self._bullet_client.applyExternalForce(self.lander_id, -1, wind_force, [0, 0, 0], self._bullet_client.LINK_FRAME)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._load_lander()
        self.step_counter = 0
        self.fuel = self.initial_fuel
        action = np.zeros(21)  # aksi awal nol
        current_obs = self._get_obs(action)
        # Inisialisasi prev_obs dengan current_obs sehingga full observation awal konsisten
        self.prev_obs = current_obs.copy()
        full_obs = np.concatenate([current_obs, self.prev_obs])
        return full_obs, self._get_info()

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._apply_thruster_forces(action)
        self._apply_dynamics()
        self._bullet_client.stepSimulation()
        time.sleep(0.0000001)
        
        self.step_counter += 1

        main_thrust_usage = abs(action[0]) * 0.1
        rcs_thrust_usage = np.sum(np.abs(action)) * 0.01
        self.fuel = max(0, self.fuel - (main_thrust_usage + rcs_thrust_usage) * (self.initial_fuel / 1000.0))
        current_obs = self._get_obs(action)
        full_obs = np.concatenate([current_obs, self.prev_obs])
        self.prev_obs = current_obs.copy()  # perbarui observasi sebelumnya
        reward, terminated, truncated = self._get_reward(full_obs, action)
        info = self._get_info()

        return full_obs, float(reward), terminated, truncated, info

    def _apply_thruster_forces(self, action):
        base_pos, base_orient = self._bullet_client.getBasePositionAndOrientation(self.lander_id)
        rot_matrix = np.array(self._bullet_client.getMatrixFromQuaternion(base_orient)).reshape(3, 3)
        
        # Main thruster (aksi index 0)
        main_force_mag = action[0] * self.max_thrust
        main_thruster_world_pos = np.array(base_pos) + rot_matrix.dot(self.main_thruster_local_position)
        main_thruster_force = (rot_matrix.dot(self.main_thruster_local_direction) * main_force_mag).tolist()
        self._bullet_client.applyExternalForce(self.lander_id, -1, main_thruster_force, main_thruster_world_pos.tolist(), pybullet.WORLD_FRAME)
        
        # RCS thrusters: aksi indices 1..20, per grup (front, back, left, right)
        action_idx = 1
        for group_key in ['front', 'back', 'left', 'right']:
            group = self.rcs_groups[group_key]
            group_origin_world = np.array(base_pos) + rot_matrix.dot(group['origin'])
            for i in range(5):
                nozzle_world_pos = group_origin_world + rot_matrix.dot(group['offsets'][i])
                force_mag = action[action_idx] * self.rcs_thruster_force_scale
                force_dir = rot_matrix.dot(group['force_directions'][i])
                force_vector = (force_dir * force_mag).tolist()
                self._bullet_client.applyExternalForce(self.lander_id, -1, force_vector, nozzle_world_pos.tolist(), pybullet.WORLD_FRAME)
                action_idx += 1

    def _get_contact_forces(self):
        contact_forces = np.zeros(len(self.leg_sensor_names))
        for idx, sensor_name in enumerate(self.leg_sensor_names):
            if sensor_name in self.leg_sensor_indices:
                link_index = self.leg_sensor_indices[sensor_name]
                contacts = self._bullet_client.getContactPoints(self.lander_id, self.plane_id, linkIndexA=link_index)
                total_force = sum(cp[9] for cp in contacts)
                contact_forces[idx] = total_force
        return contact_forces

    def _get_obs(self, action):
        pos, orient = self._bullet_client.getBasePositionAndOrientation(self.lander_id)
        lin_vel, ang_vel = self._bullet_client.getBaseVelocity(self.lander_id)
        euler_orient = pybullet.getEulerFromQuaternion(orient)
        contact_forces = self._get_contact_forces()
        
        fuel_norm = self.fuel
        raw_obs = np.array(list(pos) + list(euler_orient) + list(lin_vel) + list(ang_vel) + [fuel_norm] + list(contact_forces), dtype=np.float32)
        normalized_obs = 2 * (raw_obs - self.obs_min) / (self.obs_max - self.obs_min) - 1
        return np.clip(normalized_obs, -1, 1).astype(np.float32)

    def _unpack_obs(self, obs):
        # Jika obs merupakan full observation (34-dimensi), ambil bagian current (17-dimensi)
        if obs.shape[0] == 34:
            obs = obs[17:]
        pos = obs[0:3]
        orient = obs[3:6]
        lin_vel = obs[6:9]
        ang_vel = obs[9:12]
        fuel = obs[12]
        contact = obs[13:17]
        return pos, orient, lin_vel, ang_vel, fuel, contact

    def _get_reward(self, obs, action):
        # Pisahkan full observation menjadi previous (17-dim) dan current (17-dim)
        current_obs = obs[:17]
        prev_obs = obs[17:]
        

        # Ekstrak komponen-komponen state dari masing-masing observasi
        prev_pos, prev_orient, prev_lin_vel, prev_ang_vel, prev_fuel, prev_contact = self._unpack_obs(prev_obs)
        pos, orient, lin_vel, ang_vel, fuel, contact = self._unpack_obs(current_obs)

        # -------------------------------------------
        # 1) DEFINISIKAN TARGET DAN HITUNG ERROR PER DIMENSI
        # -------------------------------------------
        # Target untuk posisi dan orientasi (ideal: 0)
        target_pos = [0.0, 0.0, 0.0]
        target_orient = [0.0, 0.0, 0.0]

        # Target kecepatan: misal diharapkan meniadakan posisi
        target_vx = -pos[0]
        target_vy = -pos[1]
        target_vz = -pos[2] / 2 if pos[2] >= 0.01 else -0.01
        target_lin_vel = [target_vx, target_vy, target_vz]

        # Untuk observasi sebelumnya, target kecepatan dihitung dengan cara yang sama
        prev_target_vx = -prev_pos[0]
        prev_target_vy = -prev_pos[1]
        prev_target_vz = -prev_pos[2] / 2 if prev_pos[2] >= 0.1 else -0.1
        prev_target_lin_vel = [prev_target_vx, prev_target_vy, prev_target_vz]

        # Hitung error masing-masing dimensi untuk current state
        current_error = [
            pos[0] - target_pos[0],
            pos[1] - target_pos[1],
            pos[2] - target_pos[2],
            orient[0] - target_orient[0],
            orient[1] - target_orient[1],
            orient[2] - target_orient[2],
            lin_vel[0] - target_lin_vel[0],
            lin_vel[1] - target_lin_vel[1],
            lin_vel[2] - target_lin_vel[2],
        ]
        # Hitung error untuk previous state
        prev_error = [
            prev_pos[0] - target_pos[0],
            prev_pos[1] - target_pos[1],
            prev_pos[2] - target_pos[2],
            prev_orient[0] - target_orient[0],
            prev_orient[1] - target_orient[1],
            prev_orient[2] - target_orient[2],
            prev_lin_vel[0] - prev_target_lin_vel[0],
            prev_lin_vel[1] - prev_target_lin_vel[1],
            prev_lin_vel[2] - prev_target_lin_vel[2],
        ]

        # ----------------------------------------------
        # 2) HITUNG PENALTIES DAN REWARD IMPROVEMENT
        # ----------------------------------------------
        # Bobot per dimensi (misalnya, orientasi diberi bobot lebih besar)
        multiply_constant = np.array([1, 1, 1, 20, 20, 20, 10, 10, 10], dtype=float)

        # Penalties: negatif dari nilai absolut error ter-bobot
        scaled_error = np.array(current_error) * multiply_constant
        penalties_array = -np.abs(scaled_error)
        penalties = np.sum(penalties_array)

        # Reward improvement: perbaikan error dibandingkan langkah sebelumnya
        abs_prev_error = np.abs(prev_error)
        abs_current_error = np.abs(current_error)
        improvement = abs_prev_error - abs_current_error
        reward_improvement_array = improvement * multiply_constant
        reward_improvement = np.sum(reward_improvement_array)

        # Gabungkan kedua komponen shaping reward
        shaping = penalties + reward_improvement

        # ----------------------------------------------
        # 3) PENALTI PENGGUNAAN AKSI (thruster penalty)
        # ----------------------------------------------
        thruster_penalty = 0.01 * (abs(action[0]) + np.sum(np.abs(action)))

        # Total reward
        reward = shaping - thruster_penalty

        # ----------------------------------------------
        # 4) KONDISI TERMINASI DAN TRUNCATED
        # ----------------------------------------------
        terminated = False
        truncated = False

        # Misalnya, jika terdapat kontak di minimal 3 kaki (landing), maka truncated dengan bonus landing
        if np.sum(np.array(contact) > -1.0) >= 3:
            truncated = True
            reward = +100  # bonus landing

        # Jika orientasi terlalu miring, fuel habis, atau posisi terlalu rendah, dianggap gagal (terminated)
        if orient[0] > 0.75 or orient[1] > 0.75 or fuel < -1 or pos[2] < 0:
            terminated = True
            reward = -100

        # Simpan current state sebagai previous observation untuk langkah selanjutnya
        self.prev_obs = current_obs.copy()

        return reward, terminated, truncated



    def _get_info(self):
        return {"step_count": self.step_counter}
    
    def render(self):
        if self.render_mode == "human":
            lander_pos, _ = self._bullet_client.getBasePositionAndOrientation(self.lander_id)
            self._bullet_client.resetDebugVisualizerCamera(cameraDistance=10,
                                                             cameraYaw=45,
                                                             cameraPitch=-45,
                                                             cameraTargetPosition=lander_pos)
            return None
        elif self.render_mode == "rgb_array":
            lander_pos, _ = self._bullet_client.getBasePositionAndOrientation(self.lander_id)
            camera_eye = [lander_pos[0] + 5, lander_pos[1] + 5, lander_pos[2] + 5]
            camera_target = lander_pos
            camera_up = [0, 0, 1]
            view_matrix = self._bullet_client.computeViewMatrix(camera_eye, camera_target, camera_up)
            projection_matrix = self._bullet_client.computeProjectionMatrixFOV(fov=60, aspect=1.0, nearVal=0.1, farVal=100)
            width = 1080
            height = 1080
            img_arr = self._bullet_client.getCameraImage(width, height, view_matrix, projection_matrix)
            rgb_array = np.reshape(img_arr[2], (height, width, 4))
            rgb_array = rgb_array[:, :, :3]
            return rgb_array

    def close(self):
        if self._bullet_client is not None:
            self._bullet_client.disconnect()
            self._physics_client_id = -1
