import sys
sys.path.append('../')

import numpy as np

import matplotlib
font = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 18}
matplotlib.rc('font', **font)

import matplotlib.pyplot as plt
import matplotlib

import pickle
import umap

from main_module.KrotovV2_utils import *



n, temp = 30, 800
data_dir = "../paper - figure 1/data/"
subdir = "[1, 7, 9]/"
selected_digits = [1, 7, 9]
saving_dir = data_dir+subdir+"trained_net_n"+str(n)+"_T"+str(temp)+".npz"


# Loading data - will improve dir struct soon..

data_M = np.load(saving_dir)['M']
data_L = np.load(saving_dir)['L']
data_T = np.load(data_dir+"miniBatchs_images.npy")[0]



mapper = pickle.load((open(data_dir+"/umap_model_correlation.sav", 'rb')))

embedding = mapper.transform(data_T)
M_embedding = np.load(data_dir+subdir+"/memory_umap_embed_correlation_n"+str(n)+"_T"+str(temp)+".npy")
    
digits = [1, 7, 9]
indices = np.zeros(5*3)
for i in range(len(indices)):
    strictness = 0.99
    all_indices = np.argwhere(data_L[-1, :, digits[i//5]] >= strictness )
    while len(all_indices) == 0:
        strictness -= 0.1
        all_indices = np.argwhere(data_L[-1, :, digits[i//5]] >= strictness )
        print(digits[i//5], len(all_indices), strictness)
        
        
    
    indices[i] =  all_indices[np.random.randint(len(all_indices))] # -> Pick randomly when Label is mostly # digit class i//2 $
indices = np.asarray(indices, dtype=int)

# Use the time stamps file or manually get timestamps from UMAP movies.
# Handpicked, notice this is /10 because of the UMAP timesteps
t_s = [20, 50, 60, 70, 344] 
t_samples = np.asarray(t_s)*10


props = dict(boxstyle='round', facecolor='whitesmoke', alpha=0.5)


def memory_sample_plot(ax):
    tmax = len(data_M)
            
    for t_i, t in enumerate(t_samples):
        im = ax[t_i].imshow(merge_data(data_M[t, indices], 5, 3), cmap="bwr", vmin=-1, vmax=1, aspect=5.0/4.0) # Plotting the selected memory samples 
        ax[t_i].set_title(r"$t=$"+str(t), pad=15, fontsize=16, bbox=props) # Time stamps / Cosmetics
        ax[t_i].axis('off')



def UMAP_plot(ax):
    M = len(data_T)
    keys = np.zeros((M))
    for d in range(0, 10):
        keys[d*M//10:(d+1)*M//10] = d
    
    tmax, N_mem = np.shape(M_embedding)[0], np.shape(M_embedding)[1]
        
    for t_i in range(0, len(t_s)-1):
        im = ax[t_i].scatter(embedding[:, 0], embedding[:, 1], c=keys, cmap="tab10", s=10, marker="*") # Plotting the UMAP training data
    
        for i in range(0, N_mem-1):
            data_pnts = M_embedding[t_s[0]:t_s[t_i+1], i, :]
            ax[t_i].plot(data_pnts[:, 0], data_pnts[:, 1], linewidth=1, alpha=0.3, color="k") # Plotting the trajectories of the memories on UMAP


        # Time stamps / Cosmetics
        ax[t_i].text(0.95, 0.95, r"$t=$"+str(t_s[t_i+1]*10), transform=ax[t_i].transAxes, fontsize=16, verticalalignment='top', ha='right', bbox=props)

        # Plotting the memories as white points for each time point.
        ax[t_i].plot(M_embedding[t_s[t_i+1], :, 0], M_embedding[t_s[t_i+1], :, 1], marker="o",
                               linestyle="", alpha=1, markeredgecolor="k", markerfacecolor="white", markeredgewidth=1, markersize=4)

        # Cosmetics
        if t_i != 0 and t_i != 2:
            ax[t_i].set_yticks([])
            
        if t_i < 2:
            ax[t_i].set_xticks([])

        # Labels / cosmetics
        ax[0].set_ylabel("UMAP 2"); ax[2].set_ylabel("UMAP 2"); ax[2].set_xlabel("UMAP 1"); ax[3].set_xlabel("UMAP 1")



def split_plot(ax, t_start, t_stop):
    tmax, N_max, tmp = np.shape(data_M)

    # Invert using 1, 7, 9 only 
    train_mask = np.zeros(200)
    for d in selected_digits:
        train_mask[d*20: (d+1)*20] = 1
    train_mask = np.asarray(train_mask, dtype=bool)

    data_coefs = np.sum((data_M@np.linalg.pinv(data_T[train_mask])).reshape(tmax, N_max, len(selected_digits), 20), axis=-1)

    
    tab10_cmap = matplotlib.cm.tab10
    tab10_norm = matplotlib.colors.Normalize(vmin=0, vmax=10)
    
    for n in range(N_max):
        ax[0].plot(data_coefs[t_start:t_stop, n, 0], data_coefs[t_start:t_stop, n, 1], c=tab10_cmap(tab10_norm(np.argmax(data_L[-1, n]))), lw=2)
        ax[0].scatter(data_coefs[t_start, n, 0], data_coefs[t_start, n, 1], marker=".", color="k", s=400)
                

        ax[1].plot(data_coefs[t_start:t_stop, n, 0], data_coefs[t_start:t_stop, n, 2], c=tab10_cmap(tab10_norm(np.argmax(data_L[-1, n]))), lw=2)
        ax[1].scatter(data_coefs[t_start, n, 0], data_coefs[t_start, n, 2], marker=".", color="k", s=400)        
        
        
    ax[1].yaxis.set_label_position('right')
    ax[1].yaxis.tick_right()

    ax[0].set_xlabel(r"$\alpha_1$"); ax[1].set_xlabel(r"$\alpha_1$");
    ax[0].set_ylabel(r"$\alpha_7$"); ax[1].set_ylabel(r"$\alpha_9$");

        
fig = plt.figure(figsize=(25, 15.7))
axs = fig.subplot_mosaic("""

AAAAAAAABBBBBBBB..1111112222220
AAAAAAAABBBBBBBB..1111112222220
AAAAAAAABBBBBBBB..1111112222220
AAAAAAAABBBBBBBB..1111112222220
AAAAAAAABBBBBBBB..3333334444440
AAAAAAAABBBBBBBB..3333334444440
AAAAAAAABBBBBBBB..3333334444440
AAAAAAAABBBBBBBB..3333334444440
...............................
aaaaaabbbbbbccccccddddddeeeeeex
aaaaaabbbbbbccccccddddddeeeeeex
aaaaaabbbbbbccccccddddddeeeeeex

""")


ax_mem_1 = np.asarray([axs['a'], axs['b'], axs['c'], axs['d'], axs['e']])
memory_sample_plot(ax_mem_1)

ax_UMAP_1 = np.asarray([axs['1'], axs['2'], axs['3'], axs['4']])
UMAP_plot(ax_UMAP_1)

ax_split_1 = np.asarray([axs['A'], axs['B']])
split_plot(ax_split_1, 200, 2000)

ax_cb_UMAP = axs['0']
tab10_cmap = matplotlib.cm.tab10
tab10_norm = matplotlib.colors.Normalize(vmin=0, vmax=10)
cb_UMAP = matplotlib.colorbar.ColorbarBase(ax_cb_UMAP, cmap=tab10_cmap, norm=tab10_norm, orientation='vertical')
cb_UMAP.set_ticks(np.arange(0, 10, 1) + 0.5) # Finally found how to center these things 
cb_UMAP.set_ticklabels(np.arange(0, 10, 1))
cb_UMAP.set_label("Digit class")


ax_cb_mem = axs['x']
bwr_cmap = matplotlib.cm.bwr
bwr_norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
cb_mem = matplotlib.colorbar.ColorbarBase(ax_cb_mem, cmap=bwr_cmap, norm=bwr_norm, orientation='vertical')
cb_mem.set_label("Pixel value")

plt.subplots_adjust(wspace=1.2, hspace=1.2)
plt.savefig("tmp.png")
