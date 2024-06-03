import sys
sys.path.append('../')

import numpy as np

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation as anim

from main_module.KrotovV2 import *
from main_module.KrotovV2_utils import *

import matplotlib
fontsize = 40
font = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : fontsize}
matplotlib.rc('font', **font)


n, temp = 30, 670

# Loading data from Figure 3
data_dir = "../Figure 3/data/"
subdir = "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]_900_2000/"
saving_dir = data_dir+subdir+"trained_net_n"+str(n)+"_T"+str(temp)+".npz"
data_M = np.load(saving_dir)['M']
data_L = np.load(saving_dir)['L']

K = 900
Kx = 30

# Loading dataset and umap model
umap_model_path = "../defaults/umap_model_correlation.sav"
dataset = "../defaults/miniBatchs_images.npy"
data_T = np.load(dataset)[0]

# Setting up the axes geometry
fig = plt.figure(figsize=(8+2+20+4+20+1, 20))
axs = fig.subplot_mosaic("""

00000000..MMMMMMMMMMMMMMMMMMMM....UUUUUUUUUUUUUUUUUUUU!
00000000..MMMMMMMMMMMMMMMMMMMM....UUUUUUUUUUUUUUUUUUUU!
11111111..MMMMMMMMMMMMMMMMMMMM....UUUUUUUUUUUUUUUUUUUU!
11111111..MMMMMMMMMMMMMMMMMMMM....UUUUUUUUUUUUUUUUUUUU!
22222222..MMMMMMMMMMMMMMMMMMMM....UUUUUUUUUUUUUUUUUUUU!
22222222..MMMMMMMMMMMMMMMMMMMM....UUUUUUUUUUUUUUUUUUUU!
33333333..MMMMMMMMMMMMMMMMMMMM....UUUUUUUUUUUUUUUUUUUU!
33333333..MMMMMMMMMMMMMMMMMMMM....UUUUUUUUUUUUUUUUUUUU!
44444444..MMMMMMMMMMMMMMMMMMMM....UUUUUUUUUUUUUUUUUUUU!
44444444..MMMMMMMMMMMMMMMMMMMM....UUUUUUUUUUUUUUUUUUUU!
55555555..MMMMMMMMMMMMMMMMMMMM....UUUUUUUUUUUUUUUUUUUU!
55555555..MMMMMMMMMMMMMMMMMMMM....UUUUUUUUUUUUUUUUUUUU!
66666666..MMMMMMMMMMMMMMMMMMMM....UUUUUUUUUUUUUUUUUUUU!
66666666..MMMMMMMMMMMMMMMMMMMM....UUUUUUUUUUUUUUUUUUUU!
77777777..MMMMMMMMMMMMMMMMMMMM....UUUUUUUUUUUUUUUUUUUU!
77777777..MMMMMMMMMMMMMMMMMMMM....UUUUUUUUUUUUUUUUUUUU!
88888888..MMMMMMMMMMMMMMMMMMMM....UUUUUUUUUUUUUUUUUUUU!
88888888..MMMMMMMMMMMMMMMMMMMM....UUUUUUUUUUUUUUUUUUUU!
99999999..MMMMMMMMMMMMMMMMMMMM....UUUUUUUUUUUUUUUUUUUU!
99999999..MMMMMMMMMMMMMMMMMMMM....UUUUUUUUUUUUUUUUUUUU!

""")


# Setting the colorbar
ax_cb_UMAP = axs['!']
tab10_cmap = matplotlib.cm.tab10
tab10_norm = matplotlib.colors.Normalize(vmin=0, vmax=10)
cb_UMAP = matplotlib.colorbar.ColorbarBase(ax_cb_UMAP, cmap=tab10_cmap, norm=tab10_norm, orientation='vertical')
cb_UMAP.set_ticks(np.arange(0, 10, 1) + 0.5) # Finally found how to center these things 
cb_UMAP.set_ticklabels(np.arange(0, 10, 1))
cb_UMAP.set_label("Digit class", labelpad=20, fontsize=40)

# Sorting memories by the labels at the end (for visual purposes)
order_sort = np.argsort(np.argmax(data_L[-1], axis=-1))

# Preparing the label axes
t=-1
for d in range(0, 9):
    axs[str(d)].set_xticks([])
    axs[str(d)].set_yticks([100])
    axs[str(d)].set_yticklabels([100])
        
    axs[str(d)].set_ylabel(r"$\mu$", rotation=0, labelpad=20)
    axs[str(d)].yaxis.set_label_coords(-0.2, 0.32)

axs['9'].set_xlim(-1.05, 1.05)
axs['9'].set_ylabel(r"$\mu$", rotation=0, labelpad=20)
axs['9'].set_xlabel(r"$l^{\mu}_{d}$", labelpad=10)
axs['9'].yaxis.set_label_coords(-0.2, 0.32)

# Plotting the label and saving the line object
im_d = [0]*10
for d in range(10):
    im_d[d], = axs[str(d)].plot(data_L[t, :, d], range(K), color=tab10_cmap(tab10_norm(d)), linestyle="", marker=".", ms=10)

# Plotting the memory and saving the im object
im_M = axs['M'].imshow(merge_data(data_M[t, order_sort, :], Kx, Kx), cmap="bwr", vmin=-1, vmax=1)

# Cosmetics
axs['M'].set_xticks([]); axs['M'].set_yticks([])

import pickle
import umap

# Loading umap model defaults and pre-existing memory embedding from Figure3
mapper = pickle.load((open(umap_model_path, 'rb')))
embedding = mapper.transform(data_T)
M_embedding = np.load(data_dir+subdir+"/memory_umap_embed_correlation_n"+str(n)+"_T"+str(temp)+".npy")

# Creating a color key for the training data
M = len(data_T)
keys = np.zeros((M))
for d in range(0, 10):
    keys[d*M//10:(d+1)*M//10] = d
        
tmax, N_mem = np.shape(M_embedding)[0], np.shape(M_embedding)[1]
t_u = 0


axs['U'].scatter(embedding[:, 0], embedding[:, 1], c=keys, cmap="tab10", s=150, marker="*") # Plotting the UMAP training data
im_UM, = axs['U'].plot(M_embedding[t_u, :, 0], M_embedding[t_u, :, 1], marker="o",
                               linestyle="", alpha=1, markeredgecolor="k", markerfacecolor="white", markeredgewidth=4, markersize=15) # Plotting UMAP Memories

# Cosmetics
axs['U'].set_xlabel("UMAP 1")
axs['U'].set_ylabel("UMAP 2")


# Plotting the epoch of training
props = dict(boxstyle='round', facecolor='whitesmoke', alpha=0.5)
time_indicator = axs['U'].text(0.95, 0.95, r"epoch "+str(t), transform=axs['U'].transAxes, fontsize=40, verticalalignment='top', ha='right', bbox=props)

# Add title for ref
axs['M'].set_title(str(K)+"-memory system with n="+str(n) +" and rescaled temperature "+'{0:.2f}'.format(temp/784), fontsize=50, verticalalignment='bottom', ha='center', pad=30, bbox=props)

def update(t):
    # Update Labels
    print(t)
    for d in range(10):
        im_d[d].set_data(data_L[t, :, d], range(K))

    # Update memory plot
    im_M.set_data(merge_data(data_M[t, order_sort, :], Kx, Kx))

    # Update umap, adding some in-between frames for cosmetics
    # To be able to track points with your eyes 
    t_u = t//10
    dt = t_u%10

    p1 = M_embedding[t_u, :]
    p2 = M_embedding[t_u+1, :]

    p_li = p2 * (dt/10.0) + p1 * (10-dt)/10.0

    im_UM.set_data(p_li[:, 0], p_li[:, 1])

    time_indicator.set_text(r"epoch "+str(t)) # Display epoch
    
    return *im_d, im_M, im_UM, time_indicator

ani = anim.FuncAnimation(fig, update, frames=(tmax-1)*10, interval=100, blit=True)
ani.save("learning_movie_"+str(K)+"_n"+str(n)+".mov", writer="ffmpeg", fps=60)
#plt.show()



