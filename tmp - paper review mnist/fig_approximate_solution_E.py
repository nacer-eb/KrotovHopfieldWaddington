import sys
sys.path.append('../')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim

from main_module.KrotovV2_utils import *

from sklearn.decomposition import PCA
import numpy.linalg as linalg

from PIL import Image

n_i = 784
tmax = 3000

N_1, N_2, N_3 = 784, 2, 10 #*20

# Define E
def E(W_21, W_32, data_images, data_labels):
    E = 0
    for i in range(0, N):
        x = np.expand_dims(data_images[i], axis=-1)
        y = np.expand_dims(data_labels[i], axis=-1)
        dy = y - W_32@W_21@x*temp

        E += (dy.T@dy)**1
    
    return E

def E_W(W, data_images, data_labels):
    E = 0
    for i in range(0, N):
        x = np.expand_dims(data_images[i], axis=-1)
        y = np.expand_dims(data_labels[i], axis=-1)
        dy = y - W@x*temp

        E += (dy.T@dy)**1
    
    return E

# Defining dt
def W_dt(W_21, W_32, x, y, sigma_31, sigma_11):
    dy = sigma_31 - W_32@W_21@sigma_11*temp

    dt_W_21 = (W_32.T)@dy
    dt_W_32 = dy @ W_21.T

    return dt_W_21, dt_W_32


def single_time_step(W_21, W_32, data_images, data_labels, rate, sigma_31, sigma_11):
    delta_W_21 = np.zeros((N_2, N_1)) 
    delta_W_32 = np.zeros((N_3, N_2))
    
    tmp_delta_W_21, tmp_delta_W_32 = W_dt(W_21, W_32, x, y, sigma_31, sigma_11)
    
    delta_W_21 += tmp_delta_W_21
    delta_W_32 += tmp_delta_W_32

    N_21 = np.max(np.abs(delta_W_21), axis=-1)
    N_21 = np.repeat(np.expand_dims(N_21, axis=-1), N_1, axis=-1)
    
    N_32 = np.max(np.abs(delta_W_32), axis=0)
    N_32 = np.repeat(np.expand_dims(N_32, axis=0), N_3, axis=0)

    # Update

    delta_W_21 = delta_W_21/N_21
    delta_W_32 = delta_W_32/N_32
    
    W_21 += rate*delta_W_21
    W_32 += rate*delta_W_32

    return delta_W_21, delta_W_32


"""
# First load
N_range = np.arange(100, 700, 20)
E_predict, E_train = np.zeros((2, len(N_range)))
print(len(N_range))
for i in range(0, len(N_range)):
    N = N_range[i]
    temp = 1.0/700.0
    
    data_images, data_labels = get_MNIST_train_partitionned(N, get_MNIST_train_images(), get_MNIST_train_labels(), selected_digits=[1, 7])

    data_images = (data_images + 1.0)/2.0
    data_labels = (data_labels + 1.0)/2.0
    
    sigma_31 = np.zeros((N_3, N_1))
    sigma_11 = np.zeros((N_1, N_1))
    
    
    for n in range(0, N):
        x = np.expand_dims(data_images[n], axis=-1)
        y = np.expand_dims(data_labels[n], axis=-1)
        sigma_31 += y@x.T
        sigma_11 += x@x.T
        
        
    U, S, V = np.linalg.svd(sigma_31)
    V = V.T
    
    S_mat = np.zeros((N_3, N_1))
    mask = np.eye(N_3, N_1, dtype=np.int32)
    S_mat[mask==1] = S
    
    D = V.T @ sigma_11 @ V

    rcond = 1E-20
    rank = 10
    stack_rank = 10
    W = 0
    while rank>2 or stack_rank!=0:
        rcond *= 10
        W = U @ (S_mat @ np.linalg.pinv(D, rcond=rcond)) @ V.T
        rank = np.linalg.matrix_rank(W)
        
        stack_rank = np.linalg.matrix_rank(np.vstack([data_images, W])) - np.linalg.matrix_rank(data_images) 
        print(rank, stack_rank)
        
    print(E_W(W/temp, data_images, data_labels)[0][0])

    E_predict[i] = E_W(W/temp, data_images, data_labels)[0][0]
    
    
    R = 1E-7 # 0.001
    W_21 = np.zeros(( N_2, N_1)) - 0.0000001 - 0.0001*np.random.randn(N_2, N_1)
    W_32 = np.zeros((N_3, N_2)) - 0.0000001 - 0.0001*np.random.randn(N_3, N_2)

    W_32[:, :] = 0
    W_32[1, 0] = 1
    W_32[7, 1] = 1
  
    E_old = 0
    E_new = 0
    
    W_32[:, :] = 0
    W_32[1, 0] = 1
    W_32[7, 1] = 1

    print(np.shape(W[[1, 7], :]))
    print(np.shape( np.linalg.pinv(data_images) ))
    print(np.shape(data_images))

    noisyness = np.eye(N, N)
    W_21 = ( W[[1, 7], :] @ np.linalg.pinv(data_images) ) @  (noisyness)  @ data_images / temp
    
    W_21_goal = np.linalg.pinv(W_32)@W[:, :]@np.linalg.pinv(data_images)@data_images
    for t in range(0, tmax-1):

        if (t%10)==0:
            E_new = E(W_21, W_32, data_images, data_labels)    
            print(t, R, E_new[0][0])

        
        E_old = E_new

        if (t%100)==0:
            print(np.sum(np.abs(W/temp - W_32@W_21)))
        

        delta_W_21, delta_W_32 = single_time_step(W_21, W_32, data_images, data_labels, R, sigma_31, sigma_11)
        for sub_t in range(0, 5):
            E_new = E(W_21, W_32, data_images, data_labels)
                
            if E_new >= E_old:
                W_21 -= R*delta_W_21
                W_32 -= R*delta_W_32
                R /= 3
                break


            
            R *= 2
            W_21 += R*delta_W_21
            W_32 += R*delta_W_32
            
            E_old = E_new
            

        if R <= 1e-50:
            print(R)
            break
            
    E_train[i] = E(W_21, W_32, data_images, data_labels)[0][0]
    print(np.shape(E_train))
            
np.savez("tmp_save_2.npz", E_predict=E_predict, E_train=E_train)
"""

### Analysis
N_range = np.arange(100, 700, 20)
data = np.load("tmp_save_2.npz")
E_predict=data['E_predict']
E_train=data['E_train']

font = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 24}

matplotlib.rc('font', **font)

fig, ax = plt.subplots(1, 1, figsize=(16, 9))

ax.set_yscale("log")

ax.scatter(N_range, E_predict, marker="o", color="black", s=20, label="Rank Controlled Inverse method")
ax.scatter(N_range, E_train, marker="v", color="orange", s=20, label="Trained network")

from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color="black", marker="o", linestyle="", lw=4, ms=10),
                Line2D([0], [0], color="orange", marker="v", linestyle="", lw=4, ms=10)]

ax.legend(custom_lines, ["Rank Controlled Inverse method", "Trained network"], loc="upper left")

ax.set_ylabel("Energy")
ax.set_xlabel("Number of training examples")

ax.set_title("System Energy and Task Complexity")

plt.tight_layout()

plt.savefig("NetworkEnergyVTrainingSetN.png")
