import sys
sys.path.append('../')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from matplotlib import cm

from main_module.KrotovV2_utils import *

from sklearn.decomposition import PCA
import numpy.linalg as linalg



N = 2
temp = 1.0/100.0

data_images, data_labels = get_MNIST_train_partitionned(N, get_MNIST_train_images(), get_MNIST_train_labels(), [7, 1])

# Preloaded
data_images = (np.load("data_images.npy")+1.0)/2.00
data_labels = (np.load("data_labels.npy")+1.0)/2.0



n_i = 784
tmax = 100

N_1, N_2, N_3 = 784, 1, 10


# Define E
def E(W_21, W_32, data_images, data_labels):
    E = 0
    for i in range(0, N):
        x = np.expand_dims(data_images[i], axis=-1)
        y = np.expand_dims(data_labels[i], axis=-1)
        dy = y - W_32@W_21@x*temp

        E += (dy.T@dy)**1
    
    return E


# Defining dt
def W_dt(W_21, W_32, x, y):
    dy = y@x.T - W_32@W_21@x@x.T*temp

    dt_W_21 = (W_32.T)@dy
    dt_W_32 = dy @ W_21.T

    return dt_W_21, dt_W_32


# Calculate total dt
def single_time_step(W_21, W_32, data_images, data_labels, rate):
    delta_W_21 = np.zeros((N_2, N_1))
    delta_W_32 = np.zeros((N_3, N_2))
    for i in range(0, N):
        x = np.expand_dims(data_images[i], axis=-1)
        y = np.expand_dims(data_labels[i], axis=-1)
        tmp_delta_W_21, tmp_delta_W_32 = W_dt(W_21, W_32, x, y)
        
        delta_W_21 += tmp_delta_W_21
        delta_W_32 += tmp_delta_W_32

    N_21 = np.max(np.abs(delta_W_21), axis=-1)
    N_21 = np.repeat(np.expand_dims(N_21, axis=-1), N_1, axis=-1)
    
    N_32 = np.max(np.abs(delta_W_32), axis=0)
    N_32 = np.repeat(np.expand_dims(N_32, axis=0), N_3, axis=0)

    # Update 
    W_21 += rate*delta_W_21/N_21
    W_32 += rate*delta_W_32/N_32

"""
data_dir = "data_long_4/"

A, B = data_images[0], data_images[1]

W_21_all_t = np.zeros((tmax, N_2, N_1)) - 0.0000001
W_32_all_t = np.zeros((tmax, N_3, N_2))-0.0000001

for alpha in np.arange(-2, 2, 0.002):
    
    beta = 4*(2*np.random.rand()-1)#(np.random.randint(0, 2)*2-1)*np.sqrt(0.25-alpha*alpha)
    print(alpha, beta)
    W_21_all_t[0, :, :] = alpha*data_images[0, :] + beta*data_images[1, :]
    
    #W_32_all_t[0, 0, 0] = (W_21_all_t[0, 0, :]@A)/( (W_21_all_t[0, 0, :]@A)**2 + (W_21_all_t[0, 0, :]@B)**2 )
    #W_32_all_t[0, 1, 0] = (np.random.randint(0, 2)*2-1)*np.sqrt(1 - (W_32_all_t[0, 0, 0])**2)
    
    W_32_all_t[0, 0, 0] = np.sqrt(0.5)#(np.random.randint(0, 2)*2-1)
    W_32_all_t[0, 1, 0] = np.sqrt(0.5)#(np.random.randint(0, 2)*2-1)
    
    W_21 = W_21_all_t[0, :, :]
    W_32 = W_32_all_t[0, :, :]


    for t in range(0, tmax-2):
        for sub_t in range(0, 50):
            single_time_step(W_21, W_32, data_images, data_labels, 0.0005)
            
        W_21_all_t[t+1, :, :] = np.copy(W_21)
        W_32_all_t[t+1, :, :] = np.copy(W_32)
        
        
    save_file = data_dir + "alpha_"+str(alpha) + ".npz"
    np.savez(save_file, W_21_all_t=W_21_all_t, W_32_all_t=W_32_all_t)
    print(alpha, beta)
    print(A@A, A@B, B@B)


#exit()
"""
####### Analysis / Figure Making ###############


data_dir = "data_long_4/"
A, B = data_images[0], data_images[1]
print(A@A, A@B, B@B)

A_p = (B@B * A - A@B * B)*1.0/(A@A*B@B-(A@B)**2)
B_p = -(A@B * A - A@A * B)*1.0/(A@A*B@B-(A@B)**2)

alpha_range = np.arange(-2, 2, 0.002)
n_alpha = len(alpha_range)

M_alpha = np.zeros((n_alpha, tmax))
M_beta = np.zeros((n_alpha, tmax))
factors = np.zeros((n_alpha, tmax))

for alpha_i, alpha in enumerate(alpha_range):
    print(alpha_i, alpha)
    
    save_file = data_dir + "alpha_"+str(alpha) + ".npz"
    data = np.load(save_file)
    
    W_21_all_t = data['W_21_all_t']
    W_32_all_t = data['W_32_all_t']
    
    factors[alpha_i] = np.sqrt(W_32_all_t[:, 0, 0]**2 + W_32_all_t[:, 1, 0]**2)
    M_alpha[alpha_i] = A_p@W_21_all_t[:, 0, :].T#*factor
    M_beta[alpha_i] = B_p@W_21_all_t[:, 0, :].T#*factor

    
    print(alpha_i)

### Analysis

font = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 24}

matplotlib.rc('font', **font)

    
fig, ax = plt.subplots(1, 2, figsize=(24, 9), sharey=True)

t_alpha = np.arange(-4, 4, 0.001)
t_alpha, t_beta = np.meshgrid(t_alpha, t_alpha)

d_a = (t_alpha * (A@A.T) + t_beta * (A@B.T))*temp
d_b = (t_alpha * (A@B.T) + t_beta * (B@B.T))*temp


ax[0].contour(t_alpha, t_beta, (d_a**2 + d_b**2) - 1, [0], linewidths=5, alpha=0.5, colors="orange", zorder=1)
ax[0].scatter(M_alpha[:, 1]*factors[:, 1], M_beta[:, 1]*factors[:, 1], s=2, color="blue")

ax[1].contour(t_alpha, t_beta, (d_a**2 + d_b**2) - 1, [0], linewidths=5, alpha=0.5, colors="orange", zorder=1)
ax[1].scatter(M_alpha[:, -2]*factors[:, -2], M_beta[:, -2]*factors[:, -2], s=2, color="red")

from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color="orange", marker="None", lw=5, alpha=0.5),
                Line2D([0], [0], color="blue", marker="o", linestyle="", ms=10)]
ax[0].legend(custom_lines, ["Theoretical Fixed Curve", "Simulation Inital States"], loc="upper right")


custom_lines = [Line2D([0], [0], color="orange", marker="None", lw=5, alpha=0.5),
                Line2D([0], [0], color="red", marker="o", linestyle="", ms=10)]
ax[1].legend(custom_lines, ["Theoretical Fixed Curve", "Simulation Final States"], loc="upper right")


plt.title("Dynamical landscape")
ax[0].set_title("A) The inital states distribution"); ax[1].set_title("B) The final states distribution")

ax[0].set_xlabel(r"$\alpha_A$"); ax[1].set_xlabel(r"$\alpha_A$")
ax[0].set_xlim(-2, 2); ax[1].set_xlim(-2, 2)

ax[0].set_ylabel(r"$\alpha_B$")
plt.tight_layout()
plt.savefig("DynamicalLandscape.png")
exit()

#"""


graph, = ax.plot(M_alpha[:, -1].reshape((n_alpha))*factors[:, -1].reshape((n_alpha)), M_beta[:, -1].reshape((n_alpha))*factors[:, -1].reshape((n_alpha)), linestyle="", marker="o", ms=1, zorder=2)

def update_graph(t_):
    t = t_
    print(t)
    graph.set_data(M_alpha[::3, t].reshape((n_alpha))*factors[::3, t].reshape((n_alpha)), M_beta[::3, t].reshape((n_alpha))*factors[::3, t].reshape((n_alpha)))

    return graph,

ani = anim.FuncAnimation(fig, update_graph, tmax//1, interval=1000, blit=True)

plt.show()
