import sys
sys.path.append('../')

import numpy as np
import sys

from main_module.KrotovV2 import *

import matplotlib
font = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 36}
matplotlib.rc('font', **font)


data_dir = "data/"
selected_digits = [4, 9]#
prefix = str(selected_digits)+"/" # I used main,and momentum #"main"#

temp_range = np.arange(500, 900, 20)
n_range = np.arange(2, 32, 2)

data_Ms_saddles = np.zeros((len(temp_range), len(n_range), 2, 784))

isFirstRun = False
if isFirstRun:
    for i, temp in enumerate(temp_range):
        for j, n in enumerate(n_range):
            for k in range(2):
                
                saving_dir=data_dir+prefix+"net_saddle_n"+str(n)+"_T"+str(temp)+"ic"+str(selected_digits[k])+".npz"
                data_Ms_saddles[i, j, k] = np.load(saving_dir)['M'][0, :]
                                
        print(temp)

    np.save(data_dir+prefix+"data_Ms_saddles.npy", data_Ms_saddles)


data_Ms_saddles = np.load(data_dir+prefix+"data_Ms_saddles.npy")

data_T = np.load(data_dir + "miniBatchs_images.npy")[0]
data_T_inv = np.linalg.pinv(data_T)

data_Ms_saddles_coefs = data_Ms_saddles@data_T_inv

fig, ax = plt.subplots(1, 2, figsize=(16, 9))

ax[0].imshow(data_Ms_saddles_coefs[:, :, 0, 1], cmap="bwr", vmin=-1, vmax=1)
ax[1].imshow(data_Ms_saddles_coefs[:, :, 1, 0], cmap="bwr", vmin=-1, vmax=1)
plt.show()

plt.plot(data_Ms_saddles_coefs[-1, :, 0, 0], n_range, marker=".", color="purple")
plt.plot(data_Ms_saddles_coefs[-1, :, 1, 0], n_range, marker=".", color="cyan")
plt.show()



