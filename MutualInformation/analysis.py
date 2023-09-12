import sys
sys.path.append('../')

import numpy as np

from main_module.KrotovV2 import *

data_dir = "data/"

isFirstRun = True

selected_digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]#
prefix = str(selected_digits)+"/" # I used main,and momentum #"main"#

tmax = 5000
N_mem = 100


n_range = np.arange(1, 41, 2)[0:4]
Nn = len(n_range)

data_Ms = np.zeros((Nn, N_mem, 784))
data_Ls = np.zeros((Nn, N_mem, 10))

data_T = np.load(data_dir + "miniBatchs_images.npy")[0]
data_T_inv = np.linalg.pinv(data_T)

if isFirstRun:
    for temp in [800]:
        for i, n in enumerate(n_range):
            print(n, temp)
            
            saving_dir = data_dir+prefix+"trained_net_n"+str(n)+"_T"+str(temp)+".npz"

            data = np.load(saving_dir)
            data_Ms[i] = data['M'][-1]
            data_Ls[i] = data['L'][-1]

        data_coefs = (data_Ms@data_T_inv).reshape(Nn, N_mem, 10, 20)

        np.save(data_dir + prefix + "data_Ms.npy", data_Ms)
        np.save(data_dir + prefix + "data_Ls.npy", data_Ls)
        np.save(data_dir + prefix + "data_Coefs.npy", data_coefs)

data_Ms = np.load(data_dir + prefix + "data_Ms.npy")
data_Ls = np.load(data_dir + prefix + "data_Ls.npy")
data_coefs = np.load(data_dir + prefix + "data_Coefs.npy")

data_coefs_per_class = np.sum(np.abs(data_coefs), axis=-1)
data_coefs_per_class = data_coefs_per_class / np.repeat(np.expand_dims(np.sum(data_coefs_per_class, axis=-1), axis=-1), 10, axis=-1)

plt.scatter(n_range, np.mean(np.max(data_coefs_per_class, axis=-1), axis=-1))
plt.show()
