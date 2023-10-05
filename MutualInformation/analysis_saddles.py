import sys
sys.path.append('../')

import numpy as np
from multiprocessing import Pool

from main_module.KrotovV2 import *

import matplotlib
import matplotlib.pyplot as plt



data_dir = "data/"

# Makes sure the data_dir exits else creates it.
if not path.exists(data_dir):
    print(data_dir, "Does not exist. It will be created ...")
    os.mkdir(data_dir)
    print(data_dir, "Created!")

isFirstRun = False

selected_digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]#
prefix = str(selected_digits)+"_saddles/" # I used main,and momentum #"main"#
n_range = np.arange(1, 41, 1)


n_range = np.arange(1, 61, 1)
temp_range = np.asarray([550, 650, 750])


cmap_tab10 = matplotlib.cm.tab10
norm = matplotlib.colors.Normalize(vmin=0, vmax=10)

N_samples = 6

fig, ax = plt.subplots(1, 3, figsize=(16, 9))
color=["blue", "orange", "red"]
for t_i, temp in enumerate(temp_range):
    print(t_i)
    for n_i, n in enumerate(n_range):
        for ic in [1, 7]:
            saving_dir = data_dir+prefix+"trained_net_end_n"+str(n)+"_T"+str(temp)+"_ic"+str(ic)+".npz"
            
            data = np.load(saving_dir)
            data_M = data['M']
            
            data_T = data['miniBatchs_images'][0]
            data_T_inv = np.linalg.pinv(data_T)
            
            data_M_coefs = (data_M@data_T_inv)[0, 0, :].reshape(10, 20)
            data_M_coefs_d = np.sum(data_M_coefs, axis=-1)



            ax[t_i].scatter(data_M_coefs_d[4], n, color=cmap_tab10(norm(ic)), s=2)#
plt.show()



for t_i, temp in enumerate(temp_range):
    fig, ax = plt.subplots(2, 30, figsize=(16, 9))
    for n_i, n in enumerate(n_range[::len(n_range)//30]):
        for ic_i, ic in enumerate([1, 7]):
            saving_dir = data_dir+prefix+"trained_net_end_n"+str(n)+"_T"+str(temp)+"_ic"+str(ic)+".npz"
            
            data = np.load(saving_dir)
            data_M = data['M']
            
            ax[ic_i, n_i].imshow(data_M.reshape(28, 28), cmap="bwr", vmin=-1, vmax=1)

    plt.show()




