import sys
sys.path.append('../')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim

import pickle

import umap

from main_module.KrotovV2_utils import *



dataset = "../defaults/miniBatchs_images.npy"
umap_model_path = "../defaults/umap_model_correlation.sav"

def generate_umap_embedding(data_dir, prefix, n, temp, isUMAPObjAlreadyGenerated=True, verbose=False):

    data_T = np.load(dataset)[0]
    file_mapper = open(umap_model_path, 'rb') # not rb on linux test
    mapper = pickle.load(file_mapper)
        
    # Shows you the UMAP of the training set first. (if verbose is True)
    if verbose:
        M = len(data_T)
        keys = np.zeros((M))
        for d in range(0, 10):
            keys[d*M//10:(d+1)*M//10] = d
        embedding = mapper.transform(data_T)


        fig, ax = plt.subplots(1, 1, figsize=(16, 9))
        plt.scatter(embedding[:, 0], embedding[:, 1], c=keys, cmap="tab10", s=10, marker="*")
        plt.colorbar()



        
    TIME_SKIP = 10 # Time step size 
    saving_dir = data_dir+prefix+"trained_net_n"+str(n)+"_T"+str(temp)+".npz"
    data_M = np.load(saving_dir)['M'][::TIME_SKIP]
    tmax, N_mem = np.shape(data_M)[0], np.shape(data_M)[1]


    # Generate the umap embedding and save it
    t = 0; t_slice_size = 10
    M_embedding = mapper.transform(data_M[t*t_slice_size:(t+1)*t_slice_size].reshape(t_slice_size*N_mem, 784)).reshape(t_slice_size, N_mem, 2)
    
    for t in range(1, tmax//t_slice_size):
        tmp_embed = mapper.transform(data_M[t*t_slice_size:(t+1)*t_slice_size].reshape(t_slice_size*N_mem, 784)).reshape(t_slice_size, N_mem, 2)
        M_embedding = np.concatenate((M_embedding, tmp_embed), axis=0)
        
        if verbose:
            print(t, "out of", tmax//t_slice_size)
        
    np.save(data_dir+prefix+"memory_umap_embed_correlation_n"+str(n)+"_T"+str(temp), M_embedding)


    if not verbose:
       return;

    # else
    M_embedding = np.load(data_dir+prefix+"memory_umap_embed_correlation_n"+str(n)+"_T"+str(temp)+".npy")
    t = 0 
    pts, = ax.plot(M_embedding[t, :, 0], M_embedding[t, :, 1], marker=".", markersize=5, linestyle="", color="k")
    text = ax.text(0.05, 0.95, "t=", transform=ax.transAxes, fontsize=14, verticalalignment='top')
        
    # Plot the UMAP movie for reference
    def update(t):    
        pts.set_data(M_embedding[t+1, :, 0], M_embedding[t+1, :, 1])
        text.set_text("t="+str(t)) 
        return pts, text,

    ani = anim.FuncAnimation(fig, update, frames=len(M_embedding)-2, interval=100, blit=True, repeat=False) # This crashes sometimes not sure why...
    ani.save(data_dir+prefix+"UMAP_movie_n"+str(n)+"_T"+str(temp)+".mp4", writer="ffmpeg")

