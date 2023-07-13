import sys
sys.path.append('../')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim

from main_module.KrotovV2_utils import *

from sklearn.decomposition import PCA
import numpy.linalg as linalg

N = 2
data_images, data_labels = get_MNIST_train_partitionned(N, get_MNIST_train_images(), get_MNIST_train_labels(), [0, 1])

temp = 1.0/100.0

# Init network objects


data_images = (data_images+1.0)/2.0
data_labels = (data_labels+1.0)/2.0



n_i = 784

sigma_11 = np.zeros((n_i, n_i))
sigma_31 = np.zeros((10, n_i))
for i in range(0, N):
    if i % 100 == 0:
        print(i)
    x = np.expand_dims(data_images[i], -1)
    y = np.expand_dims(data_labels[i], -1)
    sigma_11 += x@x.T
    sigma_31 += y@x.T
u, s, v = linalg.svd(sigma_31)




N_1, N_2, N_3 = 784, 2, 10

W_21 = np.zeros((N_2, N_1)) - 0.03 + 0.01*np.random.rand(N_2, N_1)
W_32 = np.zeros((N_3, N_2)) - 0.01#- 0.03 + 0.01*np.random.rand(N_3, N_2)


# Define E
def E(W_21, W_32, data_images, data_labels):
    E = 0
    for i in range(0, N):
        x = np.expand_dims(data_images[i], axis=-1)
        y = np.expand_dims(data_labels[i], axis=-1)
        dy = y - W_32@W_21@x*temp

        E += dy.T@dy
    
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



tmax = 10000

for i in range(0, tmax):
    if i % 100 == 0:
        print(i)

    single_time_step(W_21, W_32, data_images, data_labels, 0.003)

    #W_21 = np.clip(W_21, -1, 1)
    #W_32 = np.clip(W_32, -1, 1)



    
print(W_21@data_images.T)
print("==="*5)
print(W_21)


"""

fig, ax = plt.subplots(1, 1, figsize=(16, 9))
im = ax.imshow((W_21@v)[0, :].reshape(28, 28), cmap="bwr", vmin=-1, vmax=1)
plt.show()


fig, ax = plt.subplots(1, 1, figsize=(16, 9))
im = ax.imshow(np.expand_dims((W_21@v)[:, 0], -1), cmap="bwr", vmin=-1, vmax=1)
plt.show()


fig, ax = plt.subplots(1, 1, figsize=(16, 9))
im = ax.imshow(merge_data(W_21@v, 2, 1), cmap="bwr", vmin=-1, vmax=1)
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(16, 9))
im = ax.imshow(merge_data(W_21, 2, 1), cmap="bwr", vmin=-1, vmax=1)
plt.show()


"""

fig, ax = plt.subplots(1, 1, figsize=(16, 9))
im = ax.imshow((u.T@W_32)@(u.T@W_32).T, cmap="bwr", vmin=-1, vmax=1)
plt.show()


fig, ax = plt.subplots(1, 1, figsize=(16, 9))
im = ax.imshow(W_32, cmap="bwr", vmin=-1, vmax=1)
plt.show()


    
print(W_32)
print(E(W_21, W_32, data_images, data_labels))




    


