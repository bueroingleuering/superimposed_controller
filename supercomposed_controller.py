import matplotlib.pyplot as plt
import numpy as np
import control as ct


def style_plot(ax, xlabel='', ylabel='', fontsize=20):
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=fontsize)


# Initialize transfer functions
T1, T2, K_1, = [1, 1, 1]
G_S1 = ct.tf(K_1, [T1, 1])
G_S2 = ct.tf(1, [T2, 1])
G_O1 = G_S1 * G_S2

# Compute step response and system dynamics
desired_poles = np.array([(-1 / (0.5 * T1) + 0j), (-1 / (0.5 * T2) - 0j)])
sys_ss = ct.tf2ss(G_O1)
A, B, C, D = sys_ss.A, sys_ss.B, sys_ss.C, sys_ss.D
R = ct.acker(A, B, desired_poles)
sys_sp = ct.ss(A - B.dot(R), B, C, D)
G_O2 = ct.ss2tf(sys_sp)
v = ct.dcgain(G_O1) / ct.dcgain(G_O2)  # prefilter
G_O2 = G_O2 * ct.tf(v, [1])

# Compute responses
t_end = ct.step_response(G_O1)[0][-1]  # get simulation time from original system
t = np.linspace(0, t_end, num=1000)  # built time vector with num steps
dt_random = np.random.random(int(len(t)/10))  # built random vector in length of t / n
dt_random[0] = 0  # initialize first entry with zero
dt_array = np.cumsum(dt_random)  # cumulate the random vector
t_ne = t_end * dt_array / dt_array[-1]  # scale the entries to end with t_end

K_2, T_I = [1, 0.5 * np.diff(t_ne).max()]  # choose integration time, where the greatest difference is

# Step responses in comparison

w = np.ones(len(t))  # setpoint vector
y1 = ct.step_response(G_O1, t)[1]  # step response original system
y2 = ct.step_response(G_O2, t)[1]  # step response tuned feedback system
x = [np.array([[0.0], [0.0]])]  # initialize x
for i in range(1, len(t_ne)):  # solver for the differential equations
    dt = (t_ne[i] - t_ne[i - 1])  # current delta time
    y = C.dot(x[i - 1])  # current output value
    K = K_2 * (1 + (w[i - 1] - y) * dt / T_I)  # current gain factor
    w_dash = K * w[i - 1] + y * (1 - K)  # gained setpoint
    x_dot = A.dot(x[i - 1]) - B.dot(R.dot(x[i - 1])) + B.dot(v * w_dash)  # current dx/dt
    x.append(x[i - 1] + x_dot * dt)  # extend state vector x by integrate dx/dt
y3 = np.dot(C, np.hstack(x)).transpose()  # calculate output vector

# Plotting
label_size = 20

# Step response plots
plt.figure()
plt.plot(t, y1, c='#284b64', linewidth=4, label='original system')
plt.plot(t, y2, c='#3C6E71', linewidth=4, label='controlled system')
plt.plot(t_ne, y3, c='#893636', linewidth=4, label='superimposed system')
style_plot(plt.gcf().axes[0], 'Time [s]', '', label_size)
plt.legend(fontsize=label_size)
plt.title('Step Response', fontsize=label_size)
plt.grid(True)
plt.show()
