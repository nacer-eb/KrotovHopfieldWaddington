import sys
sys.path.append('../')

from main_module.KrotovV2 import *

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import matplotlib.animation as animation

import pickle
import umap


data_dir = "data/"

# The digit classes to include in the training
selected_digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] 
prefix = str(selected_digits)+"/"

# Loading dataset and umap model
umap_model_path = "../defaults/umap_model_correlation.sav"
mapper = pickle.load((open(umap_model_path, 'rb')))


n, temp = 30, 670

fig = plt.figure(layout="constrained", figsize=(18, 10))

grid = GridSpec(10, 18, figure=fig)

axs = [[0 for i in range(3)] for i in range(2)]

epoch_indicators = [0 for index in range(6)]
ims = [0 for index in range(6)]
M_embeddings = [0 for index in range(6)]

tmax = 350

for i in range(2):
    for j in range(3):
        index = i*3 + j # flat index

        # Loading memories
        M_embeddings[index] = np.load(data_dir+prefix+"memory_umap_embed_correlation_n"+str(n)+"_T"+str(temp)+"_run"+str(index)+".npy")
        tmax, N_mem, umap_dims = M_embeddings[index].shape
        
        data_T = np.load(data_dir+prefix+"trained_net_n"+str(n)+"_T"+str(temp)+"_run"+str(index)+".npz")['miniBatchs_images'][0]
        embedding = mapper.transform(data_T)
        
        axs[i][j] = fig.add_subplot(grid[1+4*i:1+4*(i+1), 1+5*j:1+5*(j+1)])

        axs[i][j].scatter(embedding[:, 0], embedding[:, 1], c=np.arange(0, 10, 1).repeat(20), cmap="tab10", s=80, marker="*")
        ims[index], = axs[i][j].plot(M_embeddings[index][0, :, 0], M_embeddings[index][0, :, 1], linestyle="", marker="o",
                                     markeredgecolor="k", markerfacecolor="white", markeredgewidth=4, markersize=11)

        # Plotting axis label
        axs[i][j].text(0.05, 0.95, chr(ord('A')+index), transform=axs[i][j].transAxes, fontsize=30, verticalalignment='top', ha='left')

        # Plotting the epoch of training
        props = dict(boxstyle='round', facecolor='whitesmoke', alpha=0.5)
        epoch_indicators[index] = axs[i][j].text(0.95, 0.95, r"epoch "+str(0), transform=axs[i][j].transAxes, fontsize=20, verticalalignment='top', ha='right', bbox=props)

        axs[i][j].set_xticks([])
        axs[i][j].set_yticks([])





# Creating an populating colorbar
ax_cb = fig.add_axes([16.5/18, 1.03/10, 0.02, 7.95/10])

tab10_cmap = matplotlib.cm.tab10
tab10_norm = matplotlib.colors.Normalize(vmin=0, vmax=10)
cb_UMAP = matplotlib.colorbar.ColorbarBase(ax_cb, cmap=tab10_cmap, norm=tab10_norm, orientation='vertical')
cb_UMAP.set_ticks(np.arange(0, 10, 1) + 0.5) # Finally found how to center these things 
cb_UMAP.set_ticklabels(np.arange(0, 10, 1))
cb_UMAP.set_label("Digit class", labelpad=20, fontsize=20)



# custom lines
cls = [Line2D([0], [0], color=tab10_cmap(0.), linestyle="", marker="*", markersize=11),
                Line2D([0], [0], color="k", linestyle="", marker="o",
                                     markeredgecolor="k", markerfacecolor="white", markeredgewidth=4, markersize=11)]


ax_legend = fig.add_axes([1/18, 9.3/10, 16/18, 0.5/18])
ax_legend.legend(cls, ['Training data', 'Memories'], loc='center', ncols=2, handler_map = {tuple: matplotlib.legend_handler.HandlerTuple(None)})
ax_legend.axis('off')

def update(t):
    # Update Labels
    print(t)
    
    # Update umap, adding some in-between frames for cosmetics
    # To be able to track points with your eyes 
    t_u = t//10
    dt = t_u%10
    
    
    for index in range(6):
        M_embedding = M_embeddings[index]
        epoch_indicator = epoch_indicators[index]

        p1 = M_embedding[t_u, :]
        p2 = M_embedding[t_u+1, :]
        p_li = p2 * (dt/10.0) + p1 * (10-dt)/10.0
        
        ims[index].set_data(p_li[:, 0], p_li[:, 1])

        epoch_indicator.set_text(r"epoch "+str(t)) # Display epoch
    
    return *ims, *epoch_indicators


ani = animation.FuncAnimation(fig, update, frames=(tmax-1)*10, interval=100, blit=True)
ani.save("learning_movie_many_tsets_n"+str(n)+".mp4", writer="ffmpeg", fps=60)

#plt.savefig("layout.pdf")
