import pybullet as p
import pybullet_data
import time
import math

# Inisialisasi PyBullet
p.connect(p.GUI)  # GUI untuk tampilan visual
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # Menyertakan jalur data default

# Set lingkungan
p.setGravity(0, 0, -9.81)  # Mengatur gravitasi
p.loadURDF("plane.urdf")  # Menambahkan tanah

# Muat model robot self_balance
robot_id = p.loadURDF("twsbr_env/envs/urdf/twsbr.urdf")

# --- Fullscreen window and hide default visual elements ---
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=45, cameraPitch=-45, cameraTargetPosition=[0, 0, 0])

# Variabel untuk menyimpan ID garis sebelumnya
orientation_line_ids = {
    'x': None,  # ID untuk garis sumbu X
    'y': None,  # ID untuk garis sumbu Y
    'z': None   # ID untuk garis sumbu Z
}

# Fungsi untuk menggambar sumbu koordinat berdasarkan orientasi
def draw_oriented_axes(base_position, orientation_quaternion, length=0.5):
    global orientation_line_ids

    # Konversi orientasi dari quaternion ke euler angles
    euler = p.getEulerFromQuaternion(orientation_quaternion)

    # Hitung arah sumbu X, Y, dan Z
    x_end = [base_position[0] + length * math.cos(euler[2]),
              base_position[1] + length * math.sin(euler[2]),
              base_position[2]]

    y_end = [base_position[0] - length * math.sin(euler[2]),
              base_position[1] + length * math.cos(euler[2]),
              base_position[2]]

    z_end = [base_position[0], base_position[1], base_position[2] + length]

    # Hapus garis orientasi sebelumnya jika ada
    if orientation_line_ids['x'] is not None:
        p.removeUserDebugItem(orientation_line_ids['x'])
    if orientation_line_ids['y'] is not None:
        p.removeUserDebugItem(orientation_line_ids['y'])
    if orientation_line_ids['z'] is not None:
        p.removeUserDebugItem(orientation_line_ids['z'])

    # Gambar garis baru untuk sumbu X, Y, dan Z
    orientation_line_ids['x'] = p.addUserDebugLine(base_position, x_end, [1, 0, 0], 3)  # Sumbu X (merah)
    orientation_line_ids['y'] = p.addUserDebugLine(base_position, y_end, [0, 1, 0], 3)  # Sumbu Y (hijau)
    orientation_line_ids['z'] = p.addUserDebugLine(base_position, z_end, [0, 0, 1], 3)  # Sumbu Z (biru)

# Simulasi
while True:
    p.stepSimulation()

    # Ambil posisi dan orientasi basis dari robot
    pos, orn = p.getBasePositionAndOrientation(robot_id)

    # Gambar garis orientasi robot
    draw_oriented_axes(pos, orn)

    time.sleep(1./240.)  # Step the simulation at 60 Hz

# Disconnect dari PyBullet
p.disconnect()
