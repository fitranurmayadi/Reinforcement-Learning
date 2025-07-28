import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet
import pybullet_data
import time
import os
import random
from pybullet_utils import bullet_client
import math

class DroneEnv(gym.Env):
    """
    Environment Drone 3D dengan 4 motor (masing-masing terpasang pada ujung link) dan 4 propeler.
    
    Observasi: 13 dimensi:
      - posisi (3)
      - euler orientation (3)
      - kecepatan linear (3)
      - kecepatan angular (3)
    
    Aksi: continuous, vektor 4-dimensi (thrust masing-masing motor) dalam rentang [0, 1].
    
    Fitur: 
      - Perhitungan gaya diberikan berdasarkan posisi motor pada drone.
      - Fungsi get_local_pose() mengkalkulasi pose relatif sehingga yaw di bagian depan dianggap 0.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None, truncation_steps=1000, max_thrust=15.0):
        super(DroneEnv, self).__init__()
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.render_fps = self.metadata["render_fps"]
        self.max_thrust = max_thrust
        self.truncation_steps = truncation_steps
        self.step_counter = 0
        
        # Observasi: posisi, euler orientasi, lin & ang vel (3+3+3+3 = 12); tambahkan altitude (1)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32)
        # Aksi: 4 motor thrust (0: tidak hidup, 1: full thrust)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)

        # Setup Bullet client
        self._bullet_client = None
        self.lander_id = None
        self._load_drone()

    def _load_drone(self):
        if self._bullet_client is None:
            if self.render_mode == "human":
                self._bullet_client = bullet_client.BulletClient(pybullet.GUI, options="--width=960 --height=1080")
                self._bullet_client.configureDebugVisualizer(self._bullet_client.COV_ENABLE_GUI, 0)
            else:
                self._bullet_client = bullet_client.BulletClient(pybullet.DIRECT)
        self._init_physics_client()

    def _init_physics_client(self):
        self._bullet_client.resetSimulation()
        self._bullet_client.setGravity(0, 0, -9.8)
        self._bullet_client.setTimeStep(1.0 / self.render_fps)
        self._bullet_client.setAdditionalSearchPath(pybullet_data.getDataPath())
        # Ground plane
        self.plane_id = self._bullet_client.loadURDF("plane.urdf")

        # Muat URDF drone (pastikan file drone.urdf ada pada folder "urdf")
        urdf_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "urdf", "drone.urdf")
        # Spawn drone pada posisi acak (atau bisa ditentukan)
        #print("Loading URDF from:", urdf_path)
        start_pos = [0, 0, 0.1]
        start_orient = self._bullet_client.getQuaternionFromEuler([0, 0, 0])
        self.drone_id = self._bullet_client.loadURDF(urdf_path,
                                                     basePosition=start_pos,
                                                     baseOrientation=start_orient)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._load_drone()
        self.step_counter = 0
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._apply_motor_forces(action)
        self._bullet_client.stepSimulation()
        time.sleep(1.0 / self.render_fps)
        self.step_counter += 1

        obs = self._get_obs()
        reward = self._compute_reward(obs, action)
        terminated = False
        truncated = self.step_counter >= self.truncation_steps
        info = {"step": self.step_counter, "local_pose": self.get_local_pose()}
        return obs, reward, terminated, truncated, info

    def _apply_motor_forces(self, action):
        """
        Terapkan gaya tiap motor ke drone. Asumsikan bahwa pada URDF,
        masing-masing motor ditempatkan sebagai child dari main_body dengan nama:
          "motor_fl", "motor_fr", "motor_rl", "motor_rr".
        Gaya diberikan ke arah vertikal (dari drone ke atas) dalam frame dunia.
        """
        base_pos, base_orient = self._bullet_client.getBasePositionAndOrientation(self.drone_id)
        rot_matrix = np.array(self._bullet_client.getMatrixFromQuaternion(base_orient)).reshape(3, 3)
        
        # Nama motor sesuai dengan URDF
        motor_names = ["motor_fl", "motor_fr", "motor_rl", "motor_rr"]
        for i, motor_name in enumerate(motor_names):
            # Dapatkan posisi motor (menggunakan getLinkState)
            motor_index = self._get_link_index(motor_name)
            link_state = self._bullet_client.getLinkState(self.drone_id, motor_index)
            motor_pos = np.array(link_state[0])
            # Gaya thrust mengarah ke atas drone, yaitu arah Z lokal main body
            thrust = action[i] * self.max_thrust
            # Transformasikan arah lokal [0,0,1] ke frame dunia
            force_vector = rot_matrix.dot(np.array([0, 0, 1])) * thrust
            self._bullet_client.applyExternalForce(self.drone_id,
                                                     motor_index,
                                                     forceObj=force_vector.tolist(),
                                                     posObj=motor_pos.tolist(),
                                                     flags=pybullet.WORLD_FRAME)

    def _get_link_index(self, link_name):
        num_links = self._bullet_client.getNumJoints(self.drone_id)
        for i in range(num_links):
            info = self._bullet_client.getJointInfo(self.drone_id, i)
            if info[12].decode("utf-8") == link_name:
                return i
        raise ValueError(f"Link {link_name} tidak ditemukan pada drone URDF.")
    
    def _get_obs(self):
        pos, orient = self._bullet_client.getBasePositionAndOrientation(self.drone_id)
        lin_vel, ang_vel = self._bullet_client.getBaseVelocity(self.drone_id)
        euler_orient = self._bullet_client.getEulerFromQuaternion(orient)
        # Obs: posisi (3), euler (3), lin_vel (3), ang_vel (3), total 12; tambahkan altitude (pos[2]) untuk 13
        obs = np.array(list(pos) + list(euler_orient) + list(lin_vel) + list(ang_vel) + [pos[2]], dtype=np.float32)
        return obs

    def _compute_reward(self, obs, action):
        """
        Reward sederhana: penalti untuk jarak dari target (misal: hover di [0,0,2]) 
        dan penalti penggunaan energi.
        """
        pos = obs[0:3]
        target = np.array([0, 0, 2])
        pos_error = np.linalg.norm(np.array(pos) - target)
        energy_penalty = np.sum(action**2)
        reward = - (pos_error + 0.1 * energy_penalty)
        return reward

    def get_local_pose(self):
        """
        Hitung pose drone dalam frame lokal drone sehingga yaw depan = 0.
        Misalnya, jika drone berorientasi global dengan yaw theta, maka dikembalikan:
            pos: posisi global
            yaw_local: yaw - theta, sehingga front drone selalu dianggap 0.
        """
        pos, orient = self._bullet_client.getBasePositionAndOrientation(self.drone_id)
        euler = self._bullet_client.getEulerFromQuaternion(orient)
        # euler[2] adalah yaw global; kita "normalisasi" sehingga front drone dianggap 0
        local_yaw = -euler[2]
        return {"position": pos, "local_yaw": local_yaw, "euler": euler}

    def render(self):
        if self.render_mode == "human":
            pos, _ = self._bullet_client.getBasePositionAndOrientation(self.drone_id)
            self._bullet_client.resetDebugVisualizerCamera(cameraDistance=5,
                                                             cameraYaw=45,
                                                             cameraPitch=-30,
                                                             cameraTargetPosition=pos)
            return None
        elif self.render_mode == "rgb_array":
            pos, _ = self._bullet_client.getBasePositionAndOrientation(self.drone_id)
            camera_eye = [pos[0] + 5, pos[1] + 5, pos[2] + 5]
            camera_target = pos
            camera_up = [0, 0, 1]
            view_matrix = self._bullet_client.computeViewMatrix(camera_eye, camera_target, camera_up)
            projection_matrix = self._bullet_client.computeProjectionMatrixFOV(fov=60, aspect=1.0, nearVal=0.1, farVal=100)
            width = 720
            height = 720
            img_arr = self._bullet_client.getCameraImage(width, height, view_matrix, projection_matrix)
            rgb_array = np.reshape(img_arr[2], (height, width, 4))
            return rgb_array[:, :, :3]

    def close(self):
        if self._bullet_client is not None:
            self._bullet_client.disconnect()

# Contoh penggunaan environment
if __name__ == "__main__":
    env = DroneEnv(render_mode="human")
    obs, _ = env.reset()
    for _ in range(300):
        # Contoh aksi: thrust acak tiap motor
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step: {info['step']}, Reward: {reward}, Local Pose: {info['local_pose']}")
        if terminated or truncated:
            break
    env.close()
