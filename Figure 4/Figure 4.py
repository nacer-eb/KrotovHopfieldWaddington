import sys
sys.path.append('../')

from main_module.KrotovV2 import *

import matplotlib.pyplot as plt

import umap

import scipy.interpolate as interpolate

import matplotlib.animation as anim



selected_digits = np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
data_dir = "../Figure 3/data/[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]_400_1000/"

T = 670
for n in [3, 30]:

    savefile = data_dir+"trained_net_n" + str(n) + "_T" + str(T) + ".npz"
    
    data = np.load(savefile)
    
    if n == 3:
        data_M = data['M'][100:1000:10]
        data_L = np.argmax(data['L'][100:1000:10], axis=-1)
        
    if n == 30:
        data_M = data['M'][400:2300:5]
        data_L = np.argmax(data['L'][400:2300:5], axis=-1)

    data_T = data['miniBatchs_images']
    data_T_inv = np.linalg.pinv(data_T)


    alphas = data_M@data_T_inv
    tmax, K, N = np.shape(alphas)
    

    alphas_flat = alphas.reshape(tmax*K, N)
    data_L_flat = data_L.reshape(tmax*K)

    if n == 3:
        reducer = umap.UMAP(n_components=2, n_neighbors=1000, verbose=True, low_memory=False) #800 for high n

    if n == 30:
        reducer = umap.UMAP(n_components=2, n_neighbors=1000, verbose=True, low_memory=False) #800 for high n
    embedding = reducer.fit_transform(alphas_flat)

    
    
    fig = plt.figure(figsize=(16, 16))
    ax = fig.add_subplot()
    tab, norm = get_tab10()

    im = ax.scatter(embedding[:, 0], embedding[:, 1], cmap=tab, norm=norm, c=data_L_flat, s=1)
    cbar = plt.colorbar(im)
    
    cbar.ax.get_yaxis().set_ticks(np.arange(0, len(selected_digits), 1)+0.5)
    cbar.ax.get_yaxis().set_ticklabels(selected_digits)

    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")

    fig.savefig("umap_2d_figure_n" + str(n) + "_T" + str(T) + "_final.png")


    # Make movie

    embedding_unflat = embedding.reshape(tmax, K, 2)

    fig = plt.figure(figsize=(16, 16))
    ax = fig.add_subplot()
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    tab, norm = get_tab10()
    
    im = ax.scatter(embedding[:, 0], embedding[:, 1], cmap=tab, norm=norm, c=data_L_flat, s=3)
    cbar = plt.colorbar(im)
    
    cbar.ax.get_yaxis().set_ticks(np.arange(0, len(selected_digits), 1)+0.5)
    cbar.ax.get_yaxis().set_ticklabels(selected_digits)
    cbar.ax.set_ylabel("Digit Class")

    props = dict(boxstyle='round', facecolor='whitesmoke', alpha=0.5)
    time_indicator = ax.text(0.9, 0.95, r"epoch "+str(0), transform=ax.transAxes, fontsize=20, verticalalignment='top', ha='right', bbox=props)

    ax.set_title(r"$n = $" + str(n) + " , " + "$T_r = $" + '{0:.2f}'.format(T/784))

    def update(t):
        print(t)
        im.set_offsets(embedding_unflat[:t].reshape(t*K, 2))

        data_L_flat_tmp = data_L[:t].reshape(t*K)
        colors = tab(norm(data_L_flat_tmp))
        im.set_facecolors(colors)
        
        if n == 30:
            time_indicator.set_text(r"epoch "+str(t*5+400))

        if n == 3:
            time_indicator.set_text(r"epoch "+str(t*10+100))

        return im, time_indicator,

    ani = anim.FuncAnimation(fig, update, frames=tmax-1, blit=True)
    ani.save("umap_2d_figure_n" + str(n) + "_T" + str(T) + "_final.mp4", fps=30, extra_args=['-vcodec', 'libx264'])
