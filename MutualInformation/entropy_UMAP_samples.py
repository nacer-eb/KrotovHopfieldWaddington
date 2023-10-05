import sys
sys.path.append('../')

import numpy as np

from main_module.KrotovV2 import *
from main_module.KrotovV2_utils import *


import matplotlib
font = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 16}
matplotlib.rc('font', **font)

import matplotlib.pyplot as plt


data_dir = "data/"
subdir = "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]/"

tmax = 10000
N_mem = 100

n_range = np.arange(1, 61, 1)
Nn = len(n_range)

temp_range = [550, 650, 750]
Nt = len(temp_range)


#8*4, 64-36+1
fig = plt.figure(figsize=((8*4+3)*0.5, (57-36+1+4)*0.5), dpi=200)

axs = fig.subplot_mosaic("""
aaaaaaaa.bbbbbbbb.cccccccc.dddddddd
AAAAAAAA.BBBBBBBB.CCCCCCCC.DDDDDDDD
AAAAAAAA.BBBBBBBB.CCCCCCCC.DDDDDDDD
AAAAAAAA.BBBBBBBB.CCCCCCCC.DDDDDDDD
AAAAAAAA.BBBBBBBB.CCCCCCCC.DDDDDDDD
AAAAAAAA.BBBBBBBB.CCCCCCCC.DDDDDDDD
...................................
...................................
eeeeeeee.ffffffff.gggggggg.hhhhhhhh
EEEEEEEE.FFFFFFFF.GGGGGGGG.HHHHHHHH
EEEEEEEE.FFFFFFFF.GGGGGGGG.HHHHHHHH
EEEEEEEE.FFFFFFFF.GGGGGGGG.HHHHHHHH
EEEEEEEE.FFFFFFFF.GGGGGGGG.HHHHHHHH
EEEEEEEE.FFFFFFFF.GGGGGGGG.HHHHHHHH
...................................
...................................
iiiiiiii.jjjjjjjj.kkkkkkkk.llllllll
IIIIIIII.JJJJJJJJ.KKKKKKKK.LLLLLLLL
IIIIIIII.JJJJJJJJ.KKKKKKKK.LLLLLLLL
IIIIIIII.JJJJJJJJ.KKKKKKKK.LLLLLLLL
IIIIIIII.JJJJJJJJ.KKKKKKKK.LLLLLLLL
IIIIIIII.JJJJJJJJ.KKKKKKKK.LLLLLLLL
""")


ax = np.asarray([[axs['A'], axs['B'], axs['C'], axs['D']],
                 [axs['E'], axs['F'], axs['G'], axs['H']],
                 [axs['I'], axs['J'], axs['K'], axs['L']]])
ax_mem = np.asarray([[axs['a'], axs['b'], axs['c'], axs['d']],
                     [axs['e'], axs['f'], axs['g'], axs['h']],
                     [axs['i'], axs['j'], axs['k'], axs['l']]])

mem_size = 8
index = np.random.randint(0, 100, size=mem_size)

import pickle
import umap


data_T = np.load(data_dir+"miniBatchs_images.npy")[0]
M = len(data_T)
keys = np.zeros((M))
for d in range(0, 10):
    keys[d*M//10:(d+1)*M//10] = d

mapper = pickle.load((open(data_dir+"/umap_model_correlation.sav", 'rb')))
embedding = mapper.transform(data_T)

props = dict(boxstyle='round', facecolor='whitesmoke', alpha=0.5)


for t_i, temp in enumerate(temp_range):
    for n_i, n in enumerate([10, 21, 35, 50]):
        print(n_i, n)
        ax[t_i, n_i].scatter(embedding[:, 0], embedding[:, 1], c=keys, cmap="tab10", s=10, marker="*") # Plotting the UMAP training data

        saving_dir = data_dir + subdir + "trained_net_end_n" + str(n) + "_T"+str(temp)+".npz"

        data_M = np.load(saving_dir)['M'][0]

        print(np.shape(data_M))
        
        M_embedding = mapper.transform(data_M)
                
        ax[t_i, n_i].plot(M_embedding[:, 0], M_embedding[:, 1], marker="o",
                               linestyle="", alpha=1, markeredgecolor="k", markerfacecolor="white", markeredgewidth=1, markersize=4)


        ax_mem[t_i, n_i].imshow(merge_data(data_M[index], 8, 1), cmap="bwr", vmin=-1, vmax=1)
        ax_mem[t_i, n_i].set_xticks([]); ax_mem[t_i, n_i].set_yticks([])
        
        #ax[t_i, n_i].set_xlabel("UMAP 1"); ax[t_i, n_i].set_ylabel("UMAP 2")

        if n_i > 0:
            ax[t_i, n_i].set_yticks([])
            
        #if t_i < 2: ax[t_i, n_i].set_xticks([])

        ax[t_i, 0].set_ylabel("UMAP 2")
        #ax[-1, n_i].set_xlabel("UMAP 1")
        ax[t_i, n_i].set_xlabel("UMAP 1")

        if t_i == 0:
            ax_mem[0, n_i].set_title(r"$n="+str(n_range[n]) + r"$", pad=30, fontsize=16, bbox=props)

#plt.subplots_adjust(hspace=0.3, wspace=0.2)
plt.savefig("UMAP_samples.png")
