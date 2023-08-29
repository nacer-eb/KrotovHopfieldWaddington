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



n, temp = 30, 670
data_dir = "data/"
subdir = "momentum/"
saving_dir = data_dir+subdir+"trained_net_n"+str(n)+"_T"+str(temp)+".npz"

# Loading data - will improve dir struct soon..
data_M = np.load(saving_dir)['M']
data_L = np.load(saving_dir)['L']
data_T = np.load(data_dir+"miniBatchs_images.npy")[0]
mapper = pickle.load((open(data_dir+"/umap_model_correlation.sav", 'rb')))

embedding = mapper.transform(data_T)
M_embedding = np.load(data_dir+subdir+"/memory_umap_embed_correlation_n"+str(n)+"_T"+str(temp)+".npy")
    

indices = np.zeros(20)
for i in range(len(indices)):
    strictness = 0.99
    all_indices = np.argwhere(data_L[-1, :, i//2] >= strictness )
    while len(all_indices) == 0:
        strictness -= 0.1
        all_indices = np.argwhere(data_L[-1, :, i//2] >= strictness )
        print(i//2, len(all_indices), strictness)
        
        
    
    indices[i] =  all_indices[np.random.randint(len(all_indices))] # -> Pick randomly when Label is mostly # digit class i//2 $
indices = np.asarray(indices, dtype=int)

# Use the time stamps file or manually get timestamps from UMAP movies.
# Handpicked, notice this is /10 because of the UMAP timesteps
#t_s = [20, 27, 37, 51, 62, 90, 344] #n=3
#t_s = [20, 30, 55, 70, 150, 200, 344] #n=15
#t_s = [20, 62, 100, 132, 191, 270, 344] #n=25
t_s = [20, 90, 125, 153, 180, 280, 344] #n=30
#t_s = [20, 62, 110, 161, 187, 233, 344] #n=40
t_samples = np.asarray(t_s)*10


props = dict(boxstyle='round', facecolor='whitesmoke', alpha=0.5)

def memory_sample_plot(ax):
    tmax = len(data_M)
            
    for t_i, t in enumerate(t_samples):
        im = ax[t_i].imshow(merge_data(data_M[t, indices], 4, 5), cmap="bwr", vmin=-1, vmax=1, aspect=5.0/4.0) # Plotting the selected memory samples 
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
        if t_i != 0 and t_i != 3:
            ax[t_i].set_yticks([])
            
        if t_i < 3:
            ax[t_i].set_xticks([])

        # Labels / cosmetics
        ax[0].set_ylabel("UMAP 2"); ax[3].set_ylabel("UMAP 2"); ax[3].set_xlabel("UMAP 1"); ax[4].set_xlabel("UMAP 1"); ax[5].set_xlabel("UMAP 1")                 

        
fig = plt.figure(figsize=(17, 14.5))
axs = fig.subplot_mosaic("""

11111111111111111111111111112222222222222222222222222222333333333333333333333333333300
11111111111111111111111111112222222222222222222222222222333333333333333333333333333300
11111111111111111111111111112222222222222222222222222222333333333333333333333333333300
11111111111111111111111111112222222222222222222222222222333333333333333333333333333300
11111111111111111111111111112222222222222222222222222222333333333333333333333333333300
11111111111111111111111111112222222222222222222222222222333333333333333333333333333300
11111111111111111111111111112222222222222222222222222222333333333333333333333333333300
11111111111111111111111111112222222222222222222222222222333333333333333333333333333300
11111111111111111111111111112222222222222222222222222222333333333333333333333333333300
11111111111111111111111111112222222222222222222222222222333333333333333333333333333300
11111111111111111111111111112222222222222222222222222222333333333333333333333333333300
11111111111111111111111111112222222222222222222222222222333333333333333333333333333300
11111111111111111111111111112222222222222222222222222222333333333333333333333333333300
11111111111111111111111111112222222222222222222222222222333333333333333333333333333300
11111111111111111111111111112222222222222222222222222222333333333333333333333333333300
11111111111111111111111111112222222222222222222222222222333333333333333333333333333300
11111111111111111111111111112222222222222222222222222222333333333333333333333333333300
11111111111111111111111111112222222222222222222222222222333333333333333333333333333300
11111111111111111111111111112222222222222222222222222222333333333333333333333333333300
11111111111111111111111111112222222222222222222222222222333333333333333333333333333300
44444444444444444444444444445555555555555555555555555555666666666666666666666666666600
44444444444444444444444444445555555555555555555555555555666666666666666666666666666600
44444444444444444444444444445555555555555555555555555555666666666666666666666666666600
44444444444444444444444444445555555555555555555555555555666666666666666666666666666600
44444444444444444444444444445555555555555555555555555555666666666666666666666666666600
44444444444444444444444444445555555555555555555555555555666666666666666666666666666600
44444444444444444444444444445555555555555555555555555555666666666666666666666666666600
44444444444444444444444444445555555555555555555555555555666666666666666666666666666600
44444444444444444444444444445555555555555555555555555555666666666666666666666666666600
44444444444444444444444444445555555555555555555555555555666666666666666666666666666600
44444444444444444444444444445555555555555555555555555555666666666666666666666666666600
44444444444444444444444444445555555555555555555555555555666666666666666666666666666600
44444444444444444444444444445555555555555555555555555555666666666666666666666666666600
44444444444444444444444444445555555555555555555555555555666666666666666666666666666600
44444444444444444444444444445555555555555555555555555555666666666666666666666666666600
44444444444444444444444444445555555555555555555555555555666666666666666666666666666600
44444444444444444444444444445555555555555555555555555555666666666666666666666666666600
44444444444444444444444444445555555555555555555555555555666666666666666666666666666600
44444444444444444444444444445555555555555555555555555555666666666666666666666666666600
44444444444444444444444444445555555555555555555555555555666666666666666666666666666600
......................................................................................
......................................................................................
......................................................................................
......................................................................................
......................................................................................
......................................................................................
......................................................................................
......................................................................................
AAAAAAAAAAAABBBBBBBBBBBBCCCCCCCCCCCCDDDDDDDDDDDDEEEEEEEEEEEEFFFFFFFFFFFFGGGGGGGGGGGGXX
AAAAAAAAAAAABBBBBBBBBBBBCCCCCCCCCCCCDDDDDDDDDDDDEEEEEEEEEEEEFFFFFFFFFFFFGGGGGGGGGGGGXX
AAAAAAAAAAAABBBBBBBBBBBBCCCCCCCCCCCCDDDDDDDDDDDDEEEEEEEEEEEEFFFFFFFFFFFFGGGGGGGGGGGGXX
AAAAAAAAAAAABBBBBBBBBBBBCCCCCCCCCCCCDDDDDDDDDDDDEEEEEEEEEEEEFFFFFFFFFFFFGGGGGGGGGGGGXX
AAAAAAAAAAAABBBBBBBBBBBBCCCCCCCCCCCCDDDDDDDDDDDDEEEEEEEEEEEEFFFFFFFFFFFFGGGGGGGGGGGGXX
AAAAAAAAAAAABBBBBBBBBBBBCCCCCCCCCCCCDDDDDDDDDDDDEEEEEEEEEEEEFFFFFFFFFFFFGGGGGGGGGGGGXX
AAAAAAAAAAAABBBBBBBBBBBBCCCCCCCCCCCCDDDDDDDDDDDDEEEEEEEEEEEEFFFFFFFFFFFFGGGGGGGGGGGGXX
AAAAAAAAAAAABBBBBBBBBBBBCCCCCCCCCCCCDDDDDDDDDDDDEEEEEEEEEEEEFFFFFFFFFFFFGGGGGGGGGGGGXX
AAAAAAAAAAAABBBBBBBBBBBBCCCCCCCCCCCCDDDDDDDDDDDDEEEEEEEEEEEEFFFFFFFFFFFFGGGGGGGGGGGGXX
AAAAAAAAAAAABBBBBBBBBBBBCCCCCCCCCCCCDDDDDDDDDDDDEEEEEEEEEEEEFFFFFFFFFFFFGGGGGGGGGGGGXX
AAAAAAAAAAAABBBBBBBBBBBBCCCCCCCCCCCCDDDDDDDDDDDDEEEEEEEEEEEEFFFFFFFFFFFFGGGGGGGGGGGGXX
AAAAAAAAAAAABBBBBBBBBBBBCCCCCCCCCCCCDDDDDDDDDDDDEEEEEEEEEEEEFFFFFFFFFFFFGGGGGGGGGGGGXX
AAAAAAAAAAAABBBBBBBBBBBBCCCCCCCCCCCCDDDDDDDDDDDDEEEEEEEEEEEEFFFFFFFFFFFFGGGGGGGGGGGGXX
AAAAAAAAAAAABBBBBBBBBBBBCCCCCCCCCCCCDDDDDDDDDDDDEEEEEEEEEEEEFFFFFFFFFFFFGGGGGGGGGGGGXX
AAAAAAAAAAAABBBBBBBBBBBBCCCCCCCCCCCCDDDDDDDDDDDDEEEEEEEEEEEEFFFFFFFFFFFFGGGGGGGGGGGGXX

""")


ax_UMAPs = [axs['1'], axs['2'], axs['3'], axs['4'], axs['5'], axs['6']]
ax_UMAPs = np.asarray(ax_UMAPs)          
UMAP_plot(ax_UMAPs)


ax_mem_sample = [axs['A'], axs['B'], axs['C'], axs['D'], axs['E'], axs['F'], axs['G']]
ax_mem_sample = np.asarray(ax_mem_sample)
memory_sample_plot(ax_mem_sample)



ax_cb_UMAP = axs['0']
tab10_cmap = matplotlib.cm.tab10
tab10_norm = matplotlib.colors.Normalize(vmin=0, vmax=10)
cb_UMAP = matplotlib.colorbar.ColorbarBase(ax_cb_UMAP, cmap=tab10_cmap, norm=tab10_norm, orientation='vertical')
cb_UMAP.set_ticks(np.arange(0, 10, 1) + 0.5) # Finally found how to center these things 
cb_UMAP.set_ticklabels(np.arange(0, 10, 1))
cb_UMAP.set_label("Digit class")


ax_cb_mem = axs['X']
bwr_cmap = matplotlib.cm.bwr
bwr_norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
cb_mem = matplotlib.colorbar.ColorbarBase(ax_cb_mem, cmap=bwr_cmap, norm=bwr_norm, orientation='vertical')
cb_mem.set_label("Pixel value")


from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color=plt.cm.tab10(0.9), marker="*", linestyle="", ms=13),
                Line2D([0], [0], marker="o", markeredgecolor="k", markerfacecolor="white", markeredgewidth=4, linestyle="", ms=13),
                Line2D([0], [0], color="k", lw=2, marker="", linestyle="-")]

fig.legend(custom_lines, ['Training Data', 'Memory (Visible layer)', 'Memory trajectory/trail'], loc='upper center', ncol=4)


plt.subplots_adjust(top=0.92, wspace=1.4, hspace=1.4)
plt.savefig(data_dir + subdir + "tmp_fig_n"+str(n)+".png")

