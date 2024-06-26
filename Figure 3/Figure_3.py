import argparse
import sys
sys.path.append('../')

import numpy as np

import matplotlib.pyplot as plt
import matplotlib

from main_module.KrotovV2 import *
from main_module.KrotovV2_utils import *

import matplotlib
fontsize = 34
font = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : fontsize}
matplotlib.rc('font', **font)



# set n when running script
parser = argparse.ArgumentParser(description="This program generates Figure 3 and its supplements.")
parser.add_argument('--n', help="The n-power for the figure. [DEFAULT=30]", default=30, type=int)
parse_args = parser.parse_args()
n = parse_args.n

temp = 670
data_dir = "data/"

selected_digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
subdir = str(selected_digits)+"/"
saving_dir = data_dir+subdir+"trained_net_n"+str(n)+"_T"+str(temp)+".npz"


# Loading data - will improve dir struct soon..
data_M = np.load(saving_dir)['M']
data_L = np.load(saving_dir)['L']

umap_model_path = "../defaults/umap_model_correlation.sav"
dataset = "../defaults/miniBatchs_images.npy"
data_T = np.load(dataset)[0]


import pickle
import umap

mapper = pickle.load((open(umap_model_path, 'rb')))

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

    
# This selects handpicked memory (slight improvement for n=30)
# set useHandpicked to False if you're using a different dataset or want randomly picked memories
useHandpicked = True
if n == 30 and useHandpicked: 
    indices = [2, 56, 3, 39, 20, 87, 7, 41, 4, 72, 12, 48, 10, 31, 13, 24, 71, 82, 86, 97] # [2, 11, 0, 29, 51, 95, 42, 99, 19, 71, 8, 34, 6, 78, 7, 56, 14, 16, 70, 49] 
    
indices = np.asarray(indices, dtype=int)

# Use the time stamps file or manually get timestamps from UMAP movies.
# Handpicked, notice this is /10 because of the UMAP timesteps

if n == 3:
    t_s = [10, 30, 40, 62, 90, 120, 344] #[10, 40, 51, 62, 90, 120, 344] #n=3 #[10, 27, 37, 51, 62, 90, 344] #n=3
if n == 15:
    t_s = [20, 30, 55, 70, 150, 200, 344] #n=15
if n == 25:
    t_s = [20, 62, 100, 132, 191, 270, 344] #n=25
if n == 30:
    t_s = [40, 90, 125, 153, 180, 280, 344] #n=30 [40, 90, 125, 153, 180, 280, 344] #n=30
if n == 40:
    t_s = [20, 62, 110, 161, 187, 233, 344] #n=40

        
t_samples = np.asarray(t_s)*10


props = dict(boxstyle='round', facecolor='whitesmoke', alpha=0.5) #facecolor='cyan', alpha=0.1) 

def memory_sample_plot(ax):
    tmax = len(data_M)
            
    for t_i, t in enumerate(t_samples):
        im = ax[t_i].imshow(merge_data(data_M[t, indices], 2, 2*5), cmap="bwr", vmin=-1, vmax=1, aspect='auto') # Plotting the selected memory samples 
        ax[t_i].set_title(str(t), pad=17, fontsize=30) #bbox=props # Time stamps / Cosmetics
        ax[t_i].axis('off')

    

        
def UMAP_plot(ax):
    M = len(data_T)
    keys = np.zeros((M))
    for d in range(0, 10):
        keys[d*M//10:(d+1)*M//10] = d
    
    tmax, N_mem = np.shape(M_embedding)[0], np.shape(M_embedding)[1]
        
    for t_i in range(0, len(t_s)-1):
        im = ax[t_i].scatter(embedding[:, 0], embedding[:, 1], c=keys, cmap="tab10", s=40, marker="*") # Plotting the UMAP training data

        
        for i in range(0, N_mem-1):
            data_pnts = M_embedding[t_s[0]:t_s[t_i], i, :]

            if t_i > 0 and n!=3:# and t_i != 3 and t_i != 4 :
                data_pnts = M_embedding[t_s[0]:t_s[t_i], i, :]
                ax[t_i].plot(data_pnts[:, 0], data_pnts[:, 1], linewidth=3, alpha=0.3, color="k") # Plotting the trajectories of the memories on UMAP

            if n==3:
                data_pnts = M_embedding[t_s[0]:t_s[t_i], i, :]
                ax[t_i].plot(data_pnts[:, 0], data_pnts[:, 1], linewidth=3, alpha=0.3, color="k") # Plotting the trajectories of the memories on UMAP


        # Time stamps / Cosmetics
        #ax[t_i].text(0.5, 0.95, r"Epoch: "+str(t_s[t_i+1]*10), transform=ax[t_i].transAxes, fontsize=34, verticalalignment='top', ha='center', bbox=props)
        ax[t_i].text(0.95, 0.97, str(t_i+1), transform=ax[t_i].transAxes, fontsize=34, verticalalignment='top', ha='right', style='italic')

        # Plotting the memories as white points for each time point.
        ax[t_i].plot(M_embedding[t_s[t_i], :, 0], M_embedding[t_s[t_i], :, 1], marker="o",
                               linestyle="", alpha=1, markeredgecolor="k", markerfacecolor="white", markeredgewidth=3, markersize=10)


        ymin, ymax = ax[t_i].get_ylim()
        ax[t_i].set_ylim(ymin, ymax*1.1)
        #ax[t_i].set_ylim(-5, 20)
        #ax[t_i].set_xlim(-5, 20)
        # Cosmetics
        if t_i != 0 and t_i != 3:
            ax[t_i].set_yticks([])
            
        if t_i < 3:
            ax[t_i].set_xticks([])

        ax[t_i].set_yticks([])
        ax[t_i].set_xticks([])

        # Labels / cosmetics
        ax[0].set_ylabel("UMAP 2", labelpad=20); ax[3].set_ylabel("UMAP 2", labelpad=20)
        ax[3].set_xlabel("UMAP 1", labelpad=20); ax[4].set_xlabel("UMAP 1", labelpad=20); ax[5].set_xlabel("UMAP 1", labelpad=20)                 

        
mosaic_layout =  ("........11111111111111111111111111112222222222222222222222222222333333333333333333333333333300........\n"*27 +
                  "........44444444444444444444444444445555555555555555555555555555666666666666666666666666666600........\n"*27 +
                  "......................................................................................................\n"*19 +
                  "........AAAAAAAAAAAABBBBBBBBBBBBCCCCCCCCCCCCDDDDDDDDDDDDEEEEEEEEEEEEFFFFFFFFFFFFGGGGGGGGGGGGXX........\n"*60 +
                  "......................................................................................................\n"*8 +
                  "........SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSOO........\n"*27)

fig = plt.figure(figsize=((12*6+2+4*4)*0.2, (297-135+5)*0.2), dpi=300)
axs = fig.subplot_mosaic(mosaic_layout)


ax_UMAPs = [axs['1'], axs['2'], axs['3'], axs['4'], axs['5'], axs['6']]
ax_UMAPs = np.asarray(ax_UMAPs)          
UMAP_plot(ax_UMAPs)


ax_mem_sample = [axs['A'], axs['B'], axs['C'], axs['D'], axs['E'], axs['F'], axs['G']]
ax_mem_sample = np.asarray(ax_mem_sample)
memory_sample_plot(ax_mem_sample)



# The stack plot
M = len(data_T)
    


max_epoch = 3500

epoch_range = np.arange(1, max_epoch, 10)
classification_score = np.zeros((len(epoch_range), 10))

tab10_cmap = matplotlib.cm.tab10
tab10_norm = matplotlib.colors.Normalize(vmin=0, vmax=10)

net = KrotovNet(M=M, nbMiniBatchs=1) # The rest will be filled in by the next line load-net
net.load_net(saving_dir, epoch=0)
for t_i, epoch in enumerate(epoch_range):
    print(epoch)
    net.visibleDetectors = data_M[epoch]
    net.hiddenDetectors = data_L[epoch]
    
    for i in range(M):
        output = net.compute(data_T[i])
        
        if np.max(output) > -1.0:
            classification_score[t_i, i//20] += np.argmax(output)==(i//20) # 20 examples per class, if it's right add one else don't
           
axs['S'].stackplot(epoch_range, classification_score.T/200.0, colors=tab10_cmap(tab10_norm(np.arange(0, 10, 1))), alpha=0.7)

#axs['S'].set_xticks(np.asarray(t_s)*10, np.asarray(t_s)*10)
#if n == 3:
#    axs['S'].set_xticks([370, 620, 900, 3440], [370, 620, 900, 3440])

axs['S'].set_xlim(1, max(t_s)*10)
    
axs['S'].set_xlabel("Training epochs")
axs['S'].set_ylabel("Classification accuracy", labelpad=25)
axs['S'].set_yticks([0, 0.5, 1.0])
axs['S'].set_yticklabels([0, '50%', '100%'])







ax_cb_UMAP = axs['0']
tab10_cmap = matplotlib.cm.tab10
tab10_norm = matplotlib.colors.Normalize(vmin=0, vmax=10)
cb_UMAP = matplotlib.colorbar.ColorbarBase(ax_cb_UMAP, cmap=tab10_cmap, norm=tab10_norm, orientation='vertical')
cb_UMAP.set_ticks(np.arange(0, 10, 1) + 0.5) # Finally found how to center these things 
cb_UMAP.set_ticklabels(np.arange(0, 10, 1))
cb_UMAP.set_label("Digit class", labelpad=10)


ax_cb_UMAP = axs['O']
tab10_cmap = matplotlib.cm.tab10
tab10_norm = matplotlib.colors.Normalize(vmin=0, vmax=10)
cb_UMAP = matplotlib.colorbar.ColorbarBase(ax_cb_UMAP, cmap=tab10_cmap, norm=tab10_norm, orientation='vertical')
cb_UMAP.set_ticks(np.arange(0, 10, 1) + 0.5) # Finally found how to center these things 
cb_UMAP.set_ticklabels(np.arange(0, 10, 1))
cb_UMAP.set_label("Digit class", labelpad=10)


ax_cb_mem = axs['X']
bwr_cmap = matplotlib.cm.bwr
bwr_norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
cb_mem = matplotlib.colorbar.ColorbarBase(ax_cb_mem, cmap=bwr_cmap, norm=bwr_norm, orientation='vertical')
cb_mem.set_ticks([-1, 0, 1])
cb_mem.set_ticklabels([-1, 0, 1])
cb_mem.set_label("Pixel value", labelpad=10)


from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color=plt.cm.tab10(0.9), marker="*", linestyle="", ms=13),
                Line2D([0], [0], marker="o", markeredgecolor="k", markerfacecolor="white", markeredgewidth=4, linestyle="", ms=13),
                Line2D([0], [0], color="k", lw=2, marker="", linestyle="-")]

fig.legend(custom_lines, ['Training Data', 'Memory', 'Memory trajectory/trail'], loc='upper center', ncol=4)


plt.subplots_adjust(top=0.92, wspace=1.4, hspace=1.4)
print(n)
plt.savefig("Figure_3_n"+str(n)+"_v2.png")

