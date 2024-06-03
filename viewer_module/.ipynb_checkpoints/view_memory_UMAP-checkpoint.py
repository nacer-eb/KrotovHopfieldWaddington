import sys
sys.path.append('../')

import phate
import umap

from main_module.KrotovV2_utils import *

import matplotlib.pyplot as plt
import matplotlib.animation as anim

def view_memory_PHATE(skip, prefix, network_dir, DIGIT_CLASSES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]):

    tab_cmap, tab_norm = get_tab10() #matplotlib.cm.Set3

    data = np.load(network_dir)
    data_M = data['M'][skip:3400:10]
    data_L = data['L'][skip:3400:10]
    data_T = data['miniBatchs_images'][0]
    data_T_inv = np.linalg.pinv(data_T)


    alphas = data_M@data_T_inv

    print(np.shape(alphas))

    tmax, K, N = np.shape(alphas)
    M = len(data_T)
    alphas_flat = alphas.reshape(tmax*K, N)
    
    
    reducer = umap.UMAP(n_neighbors=(M//10)*2, min_dist=0.0, low_memory=False, verbose=True)
    
    embedding_flat = reducer.fit_transform(alphas_flat)
    embedding = embedding_flat.reshape(tmax, K, 2)
    
    embedding_labels = np.argmax(data_L[:], axis=-1) 
    embedding_labels_flat = embedding_labels.reshape(tmax*K)


    fig = plt.figure(figsize=(2*(11+3), 2*10))

    mosaic_str = ""
    for i in range(10):
        mosaic_str += "U"*10 + "!"*1 + "."*3  +  "\n"
    
    axs = fig.subplot_mosaic(mosaic_str)

    ax_U = axs['U']
    ax_cb = axs['!']

    tab_cmap, tab_norm = get_tab10() #matplotlib.cm.Set3    
    im = ax_U.scatter(embedding_flat[:, 0], embedding_flat[:, 1], c=tab_cmap(tab_norm(embedding_labels_flat)), marker=".", s=15) # Training genes all
    
    
    cb_mem = matplotlib.colorbar.ColorbarBase(ax_cb, cmap=tab_cmap, norm=tab_norm, orientation='vertical')
    ticks = np.linspace(0, len(DIGIT_CLASSES)-1, len(DIGIT_CLASSES))
    cb_mem.set_ticks(ticks + ticks[1]/2.0)
    cb_mem.set_ticklabels(DIGIT_CLASSES, fontsize=50)
    cb_mem.set_label("Digit Classes", labelpad=10)

    ax_U.set_xticks([])
    ax_U.set_yticks([])

    fig.savefig(prefix+"UMAP_PLOT.png")
    


for n in [30, 15, 3]:
    view_memory_PHATE(20, "figures/n_"+str(n)+"_900_", "../Figure 3/data/[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]_900_2000/trained_net_n"+str(n)+"_T670.npz") # old is 40.0

exit()

for n in [30, 15, 3]:
    view_memory_PHATE(20, "figures/n_"+str(n)+"_400_", "../Figure 3/data/[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]_400_1000/trained_net_n"+str(n)+"_T670.npz") # old is 40.0

for n in [30, 15, 3]:
    view_memory_PHATE(20, "figures/n_"+str(n)+"_100_", "../Figure 3/data/[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]/trained_net_n"+str(n)+"_T670.npz") # old is 40.0
