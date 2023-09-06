import sys
sys.path.append('../')

import numpy as np
import sys

from main_module.KrotovV2 import *

data_dir = "../paper - figure 1/data/main/"

fig = plt.figure(figsize=( (8*4+1)/1.5 , (21*2+6)/1.5))

axs = fig.subplot_mosaic("""

AAAAAAAABBBBBBBBCCCCCCCCDDDDDDDDx
AAAAAAAABBBBBBBBCCCCCCCCDDDDDDDDx
AAAAAAAABBBBBBBBCCCCCCCCDDDDDDDDx
AAAAAAAABBBBBBBBCCCCCCCCDDDDDDDDx
AAAAAAAABBBBBBBBCCCCCCCCDDDDDDDDx
AAAAAAAABBBBBBBBCCCCCCCCDDDDDDDDx
AAAAAAAABBBBBBBBCCCCCCCCDDDDDDDDx
AAAAAAAABBBBBBBBCCCCCCCCDDDDDDDDx
.................................
11111111111111....22222222222222X
11111111111111....22222222222222X
11111111111111....22222222222222X
11111111111111....22222222222222X
11111111111111....22222222222222X
11111111111111....22222222222222X
11111111111111....22222222222222X
11111111111111....22222222222222X
11111111111111....22222222222222X
11111111111111....22222222222222X
11111111111111....22222222222222X
11111111111111....22222222222222X
.................................
.................................
.................................
.................................
.................................
.................................
aaaaaaaabbbbbbbbccccccccddddddddo
aaaaaaaabbbbbbbbccccccccddddddddo
aaaaaaaabbbbbbbbccccccccddddddddo
aaaaaaaabbbbbbbbccccccccddddddddo
aaaaaaaabbbbbbbbccccccccddddddddo
aaaaaaaabbbbbbbbccccccccddddddddo
aaaaaaaabbbbbbbbccccccccddddddddo
aaaaaaaabbbbbbbbccccccccddddddddo
.................................
33333333333333....44444444444444O
33333333333333....44444444444444O
33333333333333....44444444444444O
33333333333333....44444444444444O
33333333333333....44444444444444O
33333333333333....44444444444444O
33333333333333....44444444444444O
33333333333333....44444444444444O
33333333333333....44444444444444O
33333333333333....44444444444444O
33333333333333....44444444444444O
33333333333333....44444444444444O

""")


ax_r = np.asarray([[axs['A'], axs['B'], axs['C'], axs['D']],
                   [axs['a'], axs['b'], axs['c'], axs['d']]]) # Reconstruction axis

ax_stats = np.asarray([axs['1'], axs['3']])
ax_stats_2 = np.asarray([axs['2'], axs['4']])

bwr_axs = np.asarray([axs['x'], axs['o']])
tab10_axs = np.asarray([axs['X'], axs['O']])

for ax in ax_r.ravel():
    ax.set_xticks([]); ax.set_yticks([])



memory_pick = [6, 4]
picked_points = np.asarray([[50, 120, 200], [5, 20, 50]])
mem_ranges = np.asarray([np.arange(1, 201, 1), np.arange(1, 51, 1)])

props = dict(boxstyle='round', facecolor='whitesmoke', alpha=0.5)
for i, n in enumerate([3, 30]):
    data = np.load(data_dir + "trained_net_n"+str(n)+"_T670.npz")
    data_M = data["M"]
    data_T = data["miniBatchs_images"][0]
    data_T_inv = np.linalg.pinv(data_T)

    coefs = data_M[-1, memory_pick[i]]@data_T_inv

    coefs_i_sort = np.argsort(np.abs(coefs))[::-1] #reverse order



    for j, k in enumerate(picked_points[i]):
        coefs_selector = np.eye(len(coefs))
        coefs_selector[k:, :] = 0
        M_deg_r = coefs_selector@coefs[coefs_i_sort]@data_T[coefs_i_sort] # First n coefficient reconstruct
        ax_r[i, j].imshow(M_deg_r.reshape(28, 28), cmap="bwr", vmin=-1, vmax=1)

        # Cosmetics
        ax_r[i, j].set_title(str(k)+" TEs", pad=20, fontsize=18, bbox=props);
        
    ax_r[i, -1].imshow(data_M[-1, memory_pick[i]].reshape(28, 28), cmap="bwr", vmin=-1, vmax=1)
    ax_r[i, -1].set_title("Original Memory", pad=20, fontsize=18, bbox=props);


    ax_r[i, 1].text(1.03, 1.25, r"$n=$"+str(n), transform=ax_r[i, 1].transAxes, fontsize=25, verticalalignment='bottom', ha='center', bbox=props)
    
    mem_range = mem_ranges[i]
    ax_stats[i].scatter(mem_range, coefs[coefs_i_sort[:len(mem_range)]], c=(coefs_i_sort[:len(mem_range)]//20), cmap="tab10", marker=".", s=90, alpha=0.9)

    for n_mem in mem_range:
        coefs_selector = np.eye(len(coefs))
        coefs_selector[n_mem:, :] = 0
        M_deg_r = coefs_selector@coefs[coefs_i_sort]@data_T[coefs_i_sort] # First n coefficient reconstruct
        
        diff = np.sum(np.abs(M_deg_r - data_M[-1, memory_pick[i]]))
        if n_mem in picked_points[i]:
            s=200
            ec = "black"
        else:
            s=90
            ec = 'None'
        ax_stats_2[i].scatter(n_mem, diff, c=(coefs_i_sort[n_mem-1]//20), cmap="tab10", vmin=0, vmax=10, marker=".", s=s, ec=ec)
    
    #Cosmetics
    ax_stats[i].hlines(0, 0, np.max(mem_range), lw=1, color="red")

    ax_stats[i].set_xlabel("Training Example #"); ax_stats_2[i].set_xlabel("Training Examples included in reconstruction")
    ax_stats[i].set_ylabel(r"$\alpha$ contribution"); ax_stats_2[i].set_ylabel("Reconstruction error", labelpad=20)


    bwr_cmap = matplotlib.cm.bwr
    bwr_norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
    cb_mem = matplotlib.colorbar.ColorbarBase(bwr_axs[i], cmap=bwr_cmap, norm=bwr_norm, orientation='vertical')
    cb_mem.set_label("Pixel value")

    tab10_cmap = matplotlib.cm.tab10
    tab10_norm = matplotlib.colors.Normalize(vmin=0, vmax=10)
    cb_stats = matplotlib.colorbar.ColorbarBase(tab10_axs[i], cmap=tab10_cmap, norm=tab10_norm, orientation='vertical')
    cb_stats.set_ticks(np.arange(0, 10, 1) + 0.5) # Finally found how to center these things 
    cb_stats.set_ticklabels(np.arange(0, 10, 1))
    cb_stats.set_label("Training Example Class", labelpad=20)


plt.savefig("Figure_MP_supp_memory.png")

