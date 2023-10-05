import sys
sys.path.append('../')

import numpy as np

from main_module.KrotovV2 import *
from main_module.KrotovV2_utils import *

import matplotlib
import matplotlib.animation as anim

data_dir = "data/"

isFirstRun = False

selected_digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]#
prefix = str(selected_digits)+"/" # I used main,and momentum #"main"#

tmax = 10000
N_mem = 100

n_range = np.arange(1, 61, 1)
Nn = len(n_range)

temp_range = [550, 650, 750]
Nt = len(temp_range)

data_Ms = np.zeros((Nt, Nn, N_mem, 784))
data_Ls = np.zeros((Nt, Nn, N_mem, 10))

# I assume everything was pre-loaded
data_Ms = np.load(data_dir + prefix + "data_Ms_T.npy")
data_Ls = np.load(data_dir + prefix + "data_Ls_T.npy")
data_coefs = np.load(data_dir + prefix + "data_Coefs_T.npy")
data_coefs_flat = data_coefs.reshape(Nt, Nn, N_mem, 200)

# Average maximum (absolute) alpha per n

data_coefs_abs_max = np.max(np.abs(data_coefs_flat), axis=-1)

n_i = 0

fig, ax = plt.subplots(1, 3, figsize=(16, 6))
for h in range(0, 3):
    ax[h].hist(data_coefs_abs_max[h, n_i], bins=np.linspace(0, 1, 20))
    ax[h].set_title("Temp = " + str(temp_range[h]) + ", n = "+str(n_range[n_i]))
    ax[h].set_xlabel(r"Maximum " + r'$ | \alpha | $ ')
    ax[h].set_ylabel("Number of memories")

def update(n_i):
    print(n_i, n_range[n_i])
    for h in range(0, 3):
        ax[h].cla()
        ax[h].hist(data_coefs_abs_max[h, n_i], bins=np.linspace(0, 1, 20))
        ax[h].set_title("Temp = " + str(temp_range[h]) + ", n = "+str(n_range[n_i]))
        ax[h].set_xlabel(r"Maximum " + r'$ | \alpha | $ ')
        ax[h].set_ylabel("Number of memories")
    
    return ax[0], ax[1], ax[2]

ani = anim.FuncAnimation(fig, update, frames=len(n_range), interval=100, blit=False, save_count=10) #len(n_range)
ani.save('alpha_max_movie.mp4', writer="ffmpeg")
fig.show()

