import sys
sys.path.append('../')

import numpy as np
import sys

from main_module.KrotovV2 import *


data_dir = "data_49_mean/"

temp_range = np.arange(400, 900, 20)
n_range = np.arange(2, 31, 1)

print(np.shape(temp_range))
print(np.shape(n_range))

data_Ms = np.zeros((len(temp_range), len(n_range), 2, 784))

data_T = np.load(data_dir+"n"+str(n_range[0])+"_T"+str(temp_range[0])+".npz")["miniBatchs_images"][0]
data_T_inv = np.linalg.pinv(data_T)


first_run = False

if first_run:

    for i, temp in enumerate(temp_range):
        for j, n in enumerate(n_range):
            saving_dir=data_dir+"n"+str(n)+"_T"+str(temp)+".npz"
            
            data = np.load(saving_dir)
            data_Ms[i, j] = data['M']
    
        print(i)
    np.save(data_dir + "data_Ms.npy", data_Ms)




    
if not first_run:
    data_Ms = np.load(data_dir + "data_Ms.npy")
    data_coefs = data_Ms @ data_T_inv
    
    
    aspect = (np.min(n_range) - np.max(n_range))/(np.min(temp_range) - np.max(temp_range))
    extent = [np.min(n_range), np.max(n_range), np.max(temp_range), np.min(temp_range)]

    digit = 0
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 9))

    t_i = -7
    plt.title("T="+str(temp_range[t_i]))
    im = ax.plot(data_coefs[t_i, :, digit, 1-digit])
    ax.set_ylabel(r"$\alpha$")
    ax.set_xlabel("n-power")
    plt.show()