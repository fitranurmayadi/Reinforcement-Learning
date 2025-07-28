import numpy as np
from scipy.linalg import solve_continuous_are, inv
import matplotlib.pyplot as plt

# Parameter dari datasheet dan perhitungan baru
M = 0.498       # Massa chassis
m_motor = 0.110 # Massa motor per buah
m_wheel = 0.015 # Massa roda per buah
b = 0.1        # Koefisien gesekan (N-m-s)
I_motor = 2.75e-5 # Momen inersia motor total
I_wheel = 2.025e-6 # Momen inersia roda total
I_chassis = 7.5e-3 # Momen inersia chassis
I_total = I_chassis + I_wheel + I_motor # Total momen inersia
g = 9.8        # Percepatan gravitasi (m/s^2)
l = 0.04       # Jarak ke pusat massa pendulum

# Denominator untuk matriks A dan B
p = I_total * (M + 2 * m_wheel) + M * 2 * m_wheel * l**2

# Matriks state-space
A = np.array([[0, 1, 0, 0],
              [0, -(I_total + 2 * l**2 * m_wheel) * b / p, (2 * m_wheel**2 * g * l**2) / p, 0],
              [0, 0, 0, 1],
              [0, -(2 * m_wheel * l * b) / p, 2 * m_wheel * g * l * (M + 2 * m_wheel) / p, 0]])

B = np.array([[0],
              [(I_total + 2 * l**2 * m_wheel) / p],
              [0],
              [2 * m_wheel * l / p]])

C = np.array([[1, 0, 0, 0],
              [0, 0, 1, 0]])

D = np.array([[0],
              [0]])

# Matriks LQR
Q = np.array([[1, 0, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 1, 0],
              [0, 0, 0, 0]])

R = np.array([[1]])

# Hitung Riccati
P = solve_continuous_are(A, B, Q, R)

# Hitung LQR Gain
K = inv(R).dot(B.T.dot(P))

# Parameter simulasi
dt = 0.001  # time step
t = np.arange(0, 20, dt)
state = np.array([0.1, 0, 0.1, 0])  # initial state [theta, theta_dot, x, x_dot]
states = []

# Simulasi
for time in t:
    states.append(state)
    state = state + (A.dot(state) + B.dot(-K.dot(state))) * dt
    
    # Cek NaN atau nilai infinite
    if np.any(np.isnan(state)) or np.any(np.isinf(state)):
        print(f"Invalid value encountered at time {time}")
        break

states = np.array(states)

# Plot hasil
plt.figure()
plt.plot(t[:len(states)], states[:, 0], label='Angle (Theta) [rad]')
plt.plot(t[:len(states)], states[:, 2], label='Position (X) [m]')
plt.xlabel('Time [s]')
plt.ylabel('State')
plt.title('Self-Balancing Robot State Over Time')
plt.legend()
plt.show()
