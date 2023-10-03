import sys
sys.path.append('../')

import numpy as np

from main_module.KrotovV2 import *
from main_module.KrotovV2_utils import *

import matplotlib
import matplotlib.animation as anim

import umap
import pickle

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

data_T = np.load(data_dir + "miniBatchs_images.npy")[0]

keys = np.zeros((len(data_T)))
for d in range(0, 10):
    keys[d*len(data_T)//10:(d+1)*len(data_T)//10] = d

mapper = pickle.load((open(data_dir+"/umap_model_correlation.sav", 'rb')))
embedding = mapper.transform(data_T)



for h in range(0, 3):
    fig, ax = plt.subplots(1, 1, figsize=(16, 9))

    ax.scatter(embedding[:, 0], embedding[:, 1], c=keys, cmap="tab10", s=10, marker="*") # Plotting the UMAP training data


    M_embedding = mapper.transform(data_Ms[h, 0, :])
    im, = ax.plot(M_embedding[:, 0], M_embedding[:, 1], color="k", marker="o", ms=10, alpha=0.5, linestyle="")

    def update(n_i):
        print(n_i)
        M_embedding = mapper.transform(data_Ms[h, n_i, :])
        im.set_data(M_embedding[:, 0], M_embedding[:, 1])
        ax.set_title("Temperature="+str(temp_range[h])+", n="+str(n_range[n_i]))

        return im, ax,

    ani = anim.FuncAnimation(fig, update, frames=len(n_range), interval=200, blit=False)
    ani.save('UMAP_T'+str(temp_range[h])+'.mp4', writer="ffmpeg")
    

