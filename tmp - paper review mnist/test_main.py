import sys
sys.path.append('../')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim

from main_module.KrotovV2_utils import *

from sklearn.decomposition import PCA
import numpy.linalg as linalg

from PIL import Image

N = 700
temp = 1.0/100.0


# First load
data_images, data_labels = get_MNIST_train_partitionned(N, get_MNIST_train_images(), get_MNIST_train_labels(), selected_digits=[1, 7])


n_i = 784
tmax = 1000

N_1, N_2, N_3 = 784, 4, 10 #*20

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
    W_21 += rate*delta_W_21/N_21
    W_32 += rate*delta_W_32/N_32




W_21 = np.zeros(( N_2, N_1)) - 0.0000001 - 0.0001*np.random.randn(N_2, N_1)
W_32 = np.zeros((N_3, N_2)) - 0.0000001 - 0.0001*np.random.randn(N_3, N_2)


a, b = [3, -0.1], [-0.1, 3]
E_old = 10000
for d in range(0, 2):    
        
    R = 0.001 # 0.001

    sigma_31 = np.zeros((N_3, N_1))
    sigma_11 = np.zeros((N_1, N_1))

    
    for n in range(0, N):
        x = np.expand_dims(data_images[n], axis=-1)
        y = np.expand_dims(data_labels[n], axis=-1)
        sigma_31 += y@x.T
        sigma_11 += x@x.T


    U, S, V = np.linalg.svd(sigma_31)

    S_mat = np.zeros((N_3, N_1))
    mask = np.eye(N_3, N_1, dtype=np.int32)

    S_mat[mask==1] = S

    
    V = V.T

    D = V.T @ sigma_11 @ V
    W = U @ (S_mat @ np.linalg.pinv(D)) @ V.T

    
    
    print(E_W(W/temp, data_images, data_labels))
    
    plt.imshow(W[1].reshape(28, 28), cmap="bwr")
    plt.show()

        

    
    exit()
    #W_21[0, :] = data_images[0, :] - data_images[2, :]
    #W_21[1, :] = data_images[1, :] - data_images[3, :]
    
    for t in range(0, tmax-1):
       
        if (t%100)==0:
            E_new = E(W_21, W_32, data_images, data_labels)
            
            print(t, R, E_new)
            
       
            
        
        for sub_t in range(0, 10):
            single_time_step(W_21, W_32, data_images, data_labels, R, sigma_31, sigma_11)
            
            

    
    print(W_32[[1, 7], :])
    print(W_21@data_images[0], W_21@data_images[1])
    
    print(W_21@data_images[2], W_21@data_images[3])

    print("==="*10)
    print(E(W_21, W_32, data_images, data_labels))
    
    
    
    fig, ax = plt.subplots(1, N_2, figsize=(16, 9))
    
    for i in range(0, N_2):
        ax[i].imshow(W_21[i].reshape(28, 28), cmap="bwr", vmin=-1, vmax=1)
    plt.show()
    
    
