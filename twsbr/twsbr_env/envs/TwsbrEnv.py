import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import os
import pybullet
import pybullet_data
from pybullet_utils import bullet_client

class TwsbrEnv(gym.Env):
    """
    Two-Wheel Self-Balancing Robot Environment dengan keluaran aksi diskrit
    (–100% hingga +100% PWM, step 10%) dan observasi yang disederhanakan
    untuk mengurangi redundansi fitur.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 100}

    def __init__(
        self,
        render_mode=None,
        max_velocity=255,
        truncation_steps=1000,
        debug_info=False
    ):
        super().__init__()

        # --- Simulation & Robot Parameters ---
        self.projection_matrix = pybullet.computeProjectionMatrixFOV(
            fov=120.0, aspect=1.0, nearVal=0.0055, farVal=0.1
        )
        self.prev_error_steer = 0.0
        self.line_last_position = None
        self.CAMERA_IDX = 2
        self.CAMERA_TARGET_IDX = 3
        self.LEFT_WHEEL_JOINT_IDX = 0
        self.RIGHT_WHEEL_JOINT_IDX = 1
        self.camera_up_vector = np.array([0, -1, 0])
        self.wheel_r = 0.045  # radius roda (m)

        assert render_mode in (None, *self.metadata["render_modes"])
        self.render_mode = render_mode
        self.render_fps = self.metadata["render_fps"]
        self.truncation_steps = truncation_steps
        self.debug_info = debug_info

        # --- Parameter Kecepatan Maksimum Linear & Pitch Maksimum ---
        self.max_velocity = max_velocity          # unit PWM maksimum
        self.max_lin_speed = self.max_velocity * self.wheel_r  # m/s
        self.max_pitch_limit = 0.25               # terminasi jika |pitch| ≥ 0.25 rad (~14°)
        self.max_yaw_error = 45.0                 # pixel error maksimum ±45

        # --- Aksi Diskrit (–100% s/d +100% PWM, step 10%) ---
        # array [-1.0, -0.8, …, 0.8, 1.0]
        self.discrete_perc = np.linspace(-1.0, 1.0, num=21, dtype=np.float32)
        # Action space: MultiDiscrete [21, 21] untuk roda kiri & kanan
        self.action_space = spaces.MultiDiscrete([len(self.discrete_perc)] * 2)

        # --- Konfigurasi Sequential Targets & Reward Weights ---
        self.sequential_targets = ["balance", "line_follow", "speed"]
        self.target_dim = len(self.sequential_targets)
        self.sequence_idx = 0
        self.current_target = self.sequential_targets[self.sequence_idx]
        # Bobot reward per sub-task
        self.reward_weights = {
            "balance":     {"pitch": 10.0, "yaw": 0.0,  "speed": 0.0},
            "line_follow": {"pitch": 5.0,  "yaw": 5.0,  "speed": 0.0},
            "speed":       {"pitch": 2.0,  "yaw": 0.5,  "speed": 5.0},
        }

        # --- Observasi yang Disederhanakan ---
        # Fitur: [target_pitch, target_speed, pitch, yaw_err, lin_spd, diff_spd, left_pwm, right_pwm]
        #   + one-hot untuk sequential target (dimensi=3)
        # Batasan observasi berdasarkan terminasi sehingga nilai mentah dapat di-normalisasi ke [-1,1].
        obs_limits = np.array([
            self.max_pitch_limit,       # target_pitch ∈ [–max_pitch_limit, +max_pitch_limit]
            self.max_lin_speed,         # target_speed ∈ [–max_lin_speed, +max_lin_speed]
            self.max_pitch_limit,       # pitch ∈ [–max_pitch_limit, +max_pitch_limit]
            self.max_yaw_error,         # yaw_err ∈ [–max_yaw_error, +max_yaw_error]
            self.max_lin_speed,         # lin_spd ∈ [–max_lin_speed, +max_lin_speed]
            2 * self.max_lin_speed,     # diff_spd ∈ [–2*max_lin_speed, +2*max_lin_speed]
            self.max_velocity,          # left_pwm ∈ [–max_velocity, +max_velocity]
            self.max_velocity           # right_pwm ∈ [–max_velocity, +max_velocity]
        ], dtype=np.float32)

        one_hot_limits = np.ones(self.target_dim, dtype=np.float32)  # one-hot ∈ [0,1]
        self.state_limit = np.concatenate((obs_limits, one_hot_limits))
        self.obs_min = -self.state_limit
        self.obs_max = self.state_limit

        obs_dim = len(self.state_limit)  # 8 + 3 = 11
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        # --- Simulasi Internal ---
        self._physics_client_id = -1
        self.load_robot()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sequence_idx = 0
        self.current_target = self.sequential_targets[self.sequence_idx]
        self.step_counter = 0
        self.prev_error_steer = 0.0
        self.line_last_position = None
        self.left_motor_power = 0.0
        self.right_motor_power = 0.0

        # Target awal
        self.target_speed = 0.0
        self.target_pitch = 0.0

        self.load_robot()
        self.initialize_wheel_variables()

        # Hitung observasi awal (nilai mentah kemudian di-normalisasi)
        pitch_raw = self._get_current_angle()
        yaw_raw = self._get_current_line_position()
        lin_spd, _, _, _, _ = self._get_wheel_data()
        obs = self._get_obs_from_raw(pitch_raw, yaw_raw, lin_spd)
        return obs.astype(np.float32), self._get_info()

    def step(self, action):
        """
        Langkah simulasi:
        1. Peta aksi diskrit ke PWM
        2. Terapkan ke simulasi
        3. Hitung observasi, reward, dan update target
        4. Check termination/truncation
        """
        # --- Pemetaan Aksi Diskrit ke PWM (± max_velocity) ---
        idx_left, idx_right = int(action[0]), int(action[1])
        pwm_left = float(self.discrete_perc[idx_left] * self.max_velocity)
        pwm_right = float(self.discrete_perc[idx_right] * self.max_velocity)
        self.left_motor_power = pwm_left
        self.right_motor_power = pwm_right

        # --- Terapkan ke PyBullet ---
        self._bullet_client.setJointMotorControl2(
            self.robot_id, self.LEFT_WHEEL_JOINT_IDX,
            pybullet.VELOCITY_CONTROL, targetVelocity=pwm_left
        )
        self._bullet_client.setJointMotorControl2(
            self.robot_id, self.RIGHT_WHEEL_JOINT_IDX,
            pybullet.VELOCITY_CONTROL, targetVelocity=pwm_right
        )
        self._bullet_client.stepSimulation()

        # --- Ambil data sensor untuk reward & termination ---
        lin_spd, lin_L, lin_R, _, _ = self._get_wheel_data()
        pitch_raw = self._get_current_angle()
        yaw_raw = self._get_current_line_position()

        # Update target_pitch (hanya untuk mode "speed")
        error_speed = lin_spd - self.target_speed
        threshold = 1.0
        max_deviation = 2.0
        if error_speed < -threshold:
            delta = (self.target_speed - threshold) - lin_spd
            delta = np.clip(delta, 0.0, max_deviation)
            self.target_pitch = (delta / max_deviation) * self.max_pitch_limit
        elif error_speed > threshold:
            delta = lin_spd - (self.target_speed + threshold)
            delta = np.clip(delta, 0.0, max_deviation)
            self.target_pitch = -(delta / max_deviation) * self.max_pitch_limit
        else:
            self.target_pitch = 0.0

        # --- Observasi & Reward ---
        obs = self._get_obs_from_raw(pitch_raw, yaw_raw, lin_spd)
        reward = self._get_reward(pitch_raw, yaw_raw, lin_spd)
        info = self._get_info()
        self.step_counter += 1

        # --- Ganti target berikutnya ---
        self.sequence_idx = (self.sequence_idx + 1) % self.target_dim
        self.current_target = self.sequential_targets[self.sequence_idx]

        # --- Cek termination & truncation ---
        terminated = False
        if abs(pitch_raw) >= self.max_pitch_limit or abs(yaw_raw) >= 0.95 * self.max_yaw_error:
            terminated = True
        truncated = (self.step_counter >= self.truncation_steps)
        if truncated:
            reward += 100.0
            info["is_success"] = True
        if terminated and not truncated:
            reward -= 100.0
            info["is_success"] = False

        return obs.astype(np.float32), reward, terminated, truncated, info

    def _get_obs_from_raw(self, pitch_raw, yaw_raw, lin_spd):
        """
        Bangun vektor observasi (8-dim + one-hot) dari nilai mentah, lalu normalisasi ke [-1,1].
        """
        _, lin_L, lin_R, _, _ = self._get_wheel_data()
        diff_spd = lin_R - lin_L

        obs_curr = np.array([
            np.clip(self.target_pitch, -self.max_pitch_limit, self.max_pitch_limit),
            np.clip(self.target_speed, -self.max_lin_speed, self.max_lin_speed),
            np.clip(pitch_raw, -self.max_pitch_limit, self.max_pitch_limit),
            np.clip(yaw_raw, -self.max_yaw_error, self.max_yaw_error),
            np.clip(lin_spd, -self.max_lin_speed, self.max_lin_speed),
            np.clip(diff_spd, -2 * self.max_lin_speed, 2 * self.max_lin_speed),
            np.clip(self.left_motor_power, -self.max_velocity, self.max_velocity),
            np.clip(self.right_motor_power, -self.max_velocity, self.max_velocity),
        ], dtype=np.float32)

        # One-hot untuk current target
        one_hot = np.zeros(self.target_dim, dtype=np.float32)
        one_hot[self.sequence_idx] = 1.0

        obs_full = np.concatenate((obs_curr, one_hot))
        # Normalisasi: 2 * (val - min)/(max - min) - 1
        obs_norm = 2 * (obs_full - self.obs_min) / (self.obs_max - self.obs_min) - 1
        return np.clip(obs_norm, -1.0, 1.0)

    def _get_wheel_data(self):
        """
        Ambil kecepatan dan posisi sudut roda dari PyBullet:
        - lin_spd: kecepatan linear rata-rata (m/s)
        - lin_L, lin_R: kecepatan linear per roda (m/s)
        - rev_l, rev_r: total revolusi (dipakai jika perlu)
        """
        ls = pybullet.getJointState(self.robot_id, self.LEFT_WHEEL_JOINT_IDX)
        rs = pybullet.getJointState(self.robot_id, self.RIGHT_WHEEL_JOINT_IDX)
        v_l, v_r = ls[1], rs[1]      # kecepatan sudut (rad/s)
        ang_l, ang_r = ls[0], rs[0]  # posisi sudut (rad)

        # Track revolusi kumulatif
        global prev_l, prev_r, rev_l, rev_r
        d_l = ang_l - prev_l
        d_r = ang_r - prev_r
        if abs(d_l) > np.pi: d_l -= np.sign(d_l) * 2 * np.pi
        if abs(d_r) > np.pi: d_r -= np.sign(d_r) * 2 * np.pi
        rev_l += d_l / (2 * np.pi)
        rev_r += d_r / (2 * np.pi)
        prev_l, prev_r = ang_l, ang_r

        # Konversi kecepatan sudut → kecepatan linear (m/s)
        lin_L = v_l * self.wheel_r
        lin_R = v_r * self.wheel_r
        lin_spd = (lin_L + lin_R) / 2.0
        return lin_spd, lin_L, lin_R, rev_l, rev_r

    def _get_current_angle(self):
        """
        Dapatkan pitch (rotasi sumbu-x) robot.
        """
        _, orn = pybullet.getBasePositionAndOrientation(self.robot_id)
        _, pitch, _ = pybullet.getEulerFromQuaternion(orn)
        return pitch

    def _get_current_line_position(self):
        """
        Tangkap citra dari kamera, thresholding untuk ekstraksi garis,
        dan hitung offset (pixel) sebagai yaw_err. Jika garis tidak terdeteksi,
        gunakan posisi terakhir.
        """
        cam_pos = pybullet.getLinkState(self.robot_id, self.CAMERA_IDX)[0]
        tgt_pos = pybullet.getLinkState(self.robot_id, self.CAMERA_TARGET_IDX)[0]
        roll, pitch, yaw = pybullet.getEulerFromQuaternion(
            pybullet.getLinkState(self.robot_id, self.CAMERA_IDX)[1]
        )
        # Rotasi kamera agar orientasi benar
        R = (
            self._Rz(np.deg2rad(90) + yaw)
            @ self._Ry(pitch)
            @ self._Rx(roll)
        )
        view = pybullet.computeViewMatrix(
            cameraEyePosition=cam_pos,
            cameraTargetPosition=tgt_pos,
            cameraUpVector=R @ self.camera_up_vector
        )
        _, _, rgb, _, _ = pybullet.getCameraImage(
            90, 1, view, self.projection_matrix
        )
        img = np.reshape(rgb, (rgb.shape[0], rgb.shape[1], 4))
        gray = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGBA2RGB), cv2.COLOR_RGB2GRAY)
        _, bin_img = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
        m = cv2.moments(bin_img)
        cx = None if m['m00'] == 0 else m['m10'] / m['m00']
        width = gray.shape[1]  # seharusnya 90
        if cx is None:
            if self.line_last_position == 'left':
                err = -width / 2
            elif self.line_last_position == 'right':
                err = width / 2
            else:
                err = self.prev_error_steer
        else:
            err = cx - (width / 2)
            self.line_last_position = 'left' if cx < (width / 2) else 'right'
        self.prev_error_steer = err
        return err

    def _get_reward(self, pitch_raw, yaw_raw, lin_spd):
        """
        Hitung reward sesuai current_target:
        reward = -[w_pitch*(target_pitch - pitch_raw)^2 + w_yaw*(yaw_raw)^2 + w_speed*(target_speed - lin_spd)^2].
        """
        w = self.reward_weights[self.current_target]
        pitch_term = (self.target_pitch - pitch_raw) ** 2
        yaw_term = yaw_raw ** 2
        speed_term = (self.target_speed - lin_spd) ** 2

        reward = -(
            w["pitch"] * pitch_term
            + w["yaw"] * yaw_term
            + w["speed"] * speed_term
        )
        return float(reward)

    def _get_info(self):
        return {"step_count": self.step_counter}

    def _Rx(self, theta):
        return np.array([
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta),  np.cos(theta)]
        ])

    def _Ry(self, theta):
        return np.array([
            [ np.cos(theta), 0, np.sin(theta)],
            [0,              1, 0            ],
            [-np.sin(theta), 0, np.cos(theta)]
        ])

    def _Rz(self, theta):
        return np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0,               0,             1]
        ])

    def initialize_wheel_variables(self):
        global prev_l, prev_r, rev_l, rev_r
        prev_l = prev_r = 0.0
        rev_l = rev_r = 0

    def load_robot(self):
        if self._physics_client_id < 0:
            mode = pybullet.GUI if self.render_mode == 'human' else pybullet.DIRECT
            self._bullet_client = bullet_client.BulletClient(mode, options='--width=1920 --height=1000')
            self._init_physics_client()
        else:
            # Hapus robot lama, lalu muat ulang
            self._bullet_client.removeBody(self.robot_id)
            self._init_physics_client()

    def _init_physics_client(self):
        self._physics_client_id = self._bullet_client._client
        bc = self._bullet_client
        bc.resetSimulation()
        bc.setGravity(0, 0, -9.8)
        bc.setTimeStep(1.0 / self.render_fps)
        bc.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Muat plane dan tekstur jalur
        plane = bc.loadURDF('plane.urdf', [-3, -4.297, 0], globalScaling=3.0)
        tex = bc.loadTexture(os.path.join(os.path.dirname(__file__), 'texture', 'random_line_trace_ground.png'))
        bc.changeVisualShape(plane, -1, textureUniqueId=tex)

        # Muat robot URDF
        self.robot_id = bc.loadURDF(
            os.path.join(os.path.dirname(__file__), 'urdf', 'twsbr.urdf'),
            [0, 0, 0],
            bc.getQuaternionFromEuler([0, 0, 0])
        )
        # Inisialisasi motor roda ke 0
        bc.setJointMotorControl2(self.robot_id, 0, pybullet.VELOCITY_CONTROL, targetVelocity=0)
        bc.setJointMotorControl2(self.robot_id, 1, pybullet.VELOCITY_CONTROL, targetVelocity=0)

    def render(self):
        if self.render_mode == 'human' and self._physics_client_id >= 0:
            self._bullet_client.resetDebugVisualizerCamera(
                cameraDistance=1.5,
                cameraYaw=45,
                cameraPitch=-45,
                cameraTargetPosition=[0, 0, 0]
            )

    def close(self):
        if self._physics_client_id >= 0:
            self._bullet_client.disconnect()
            self._physics_client_id = -1
