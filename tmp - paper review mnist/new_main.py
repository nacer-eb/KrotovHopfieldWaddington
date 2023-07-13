import sys
sys.path.append('../')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim

from main_module.KrotovV2_utils import *

from sklearn.decomposition import PCA
import numpy.linalg as linalg

N = 2
temp = 1.0/100.0

# First load

data_images, data_labels = get_MNIST_train_partitionned(N, get_MNIST_train_images(), get_MNIST_train_labels(), [7, 1])

"""
np.save("data_images2.npy", data_images)
np.save("data_labels2.npy", data_labels)
data_images, data_labels = get_MNIST_train_partitionned(N, get_MNIST_train_images(), get_MNIST_train_labels(), [7, 1])

data_images = (data_images+0.0)/1.0
data_labels = (data_labels+0.0)/1.0
0
np.save("data_images.npy", data_images)
np.save("data_labels.npy", data_labels)
"""

# Preloaded
data_images = (np.load("data_images.npy")+1.0)/2.00
data_labels = (np.load("data_labels.npy")+1.0)/2.0


print(data_images@data_images.T)

n_i = 784
tmax = 1000

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


data_dir = "data_long_2/"

A, B = data_images[0], data_images[1]
W_21_all_t = np.zeros((tmax, N_2, N_1)) - 0.0000001
W_32_all_t = np.zeros((tmax, N_3, N_2))-0.0000001
for alpha in np.arange(-2, 2, 0.02)[20: 21]:
    print(alpha)
    beta = (np.random.randint(0, 2)*2-1)*np.sqrt(4-alpha*alpha)
    
    W_21_all_t[0, :, :] = alpha*data_images[0, :] + beta*data_images[1, :]
    
    W_32_all_t[0, 0, 0] = (W_21_all_t[0, 0, :]@A)/( (W_21_all_t[0, 0, :]@A)**2 + (W_21_all_t[0, 0, :]@B)**2 )
    W_32_all_t[0, 1, 0] = (np.random.randint(0, 2)*2-1)*np.sqrt(1 - (W_32_all_t[0, 0, 0])**2)
    
    W_32_all_t[0, 0, 0] = (np.random.randint(0, 2)*2-1)
    W_32_all_t[0, 1, 0] = (np.random.randint(0, 2)*2-1)
    
    W_21 = W_21_all_t[0, :, :]
    W_32 = W_32_all_t[0, :, :]
    
    for t in range(0, tmax-1):
        for sub_t in range(0, 10):
            single_time_step(W_21, W_32, data_images, data_labels, 0.00021)
            
        W_21_all_t[t+1, :, :] = np.copy(W_21)
        W_32_all_t[t+1, :, :] = np.copy(W_32)
        
        
    save_file = data_dir + "alpha_"+str(alpha) + ".npz"
    np.savez(save_file, W_21_all_t=W_21_all_t, W_32_all_t=W_32_all_t)
    print(alpha, beta)

exit()

A, B = data_images[0], data_images[1]

print(A@A, A@B, B@B)

A_p = (B@B * A - A@B * B)*1.0/(A@A*B@B-(A@B)**2)
B_p = -(A@B * A - A@A * B)*1.0/(A@A*B@B-(A@B)**2)

factor = W_32_all_t[-1, 1, 0]**2 + W_32_all_t[-1, 7, 0]**2

print(W_32_all_t[-1, 1, 0]/factor)
print(W_32_all_t[-1, 7, 0]/factor)

print("================")

print(A_p@W_21_all_t[-1, 0, :].T*factor)
print(B_p@W_21_all_t[-1, 0, :].T*factor)




    


