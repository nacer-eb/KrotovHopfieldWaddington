import sys
sys.path.append('../')

import numpy as np

import matplotlib
font = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 18}
matplotlib.rc('font', **font)

import matplotlib.pyplot as plt
import matplotlib.animation as anim
from mpl_toolkits.axes_grid1 import make_axes_locatable

import pickle

import umap
import umap.plot

from main_module.KrotovV2 import *
from main_module.KrotovV2_utils import *

data_dir = "data_100_10_200/"

noise_r = 8
temp = 670
for n in [30, 3]:
    
    r = 1.0/10**(noise_r)
    net = KrotovNet(Kx=10, Ky=10, n_deg=n, m_deg=n, M=200, nbMiniBatchs=1, momentum=0*0.6, rate=0.005, temp=temp, rand_init_mean=-0.001, rand_init_std=r)


    data_T = np.load(data_dir+"miniBatchs_images.npy")[0]
    mapper = pickle.load((open(data_dir+"/umap_model_correlation.sav", 'rb')))
    

    M = len(data_T)
    keys = np.zeros((M))
    for d in range(0, 10):
        keys[d*M//10:(d+1)*M//10] = d
        
    embedding = mapper.transform(data_T)
    M_embedding = np.load(data_dir+"/memory_umap_embed_correlation_n30.npy") # Used only for the background training data

    tmax, N_mem = np.shape(M_embedding)[0], np.shape(M_embedding)[1]

    
    t_s = [20, 90, 125, 153, 180, 280, 344] # high n

    if n == 3:
        t_s = [20, 27, 37, 51, 62, 90, 344] # low n

    fig, ax = plt.subplots(2, 3, figsize=(18, 10), sharey=True, sharex=True)
    for t_i in range(0, len(t_s)-1):
        
        net.load_net(data_dir + "run_0_n"+str(n)+"_T670.npz", epoch=t_s[t_i]*10)

        C = [0]*200
        for i in range(0, 200):
            C[i] = np.argmax(net.compute(net.miniBatchs_images[0, i]), axis=-1)
            print(i)
            
            
        im = ax[t_i//3, t_i%3].scatter(embedding[:, 0], embedding[:, 1], c=C, cmap="tab10", s=10, alpha=0.1, marker=".")
        im = ax[t_i//3, t_i%3].scatter(embedding[:, 0], embedding[:, 1], c=keys, cmap="tab10", s=70, alpha=(C==keys), marker="*")
    

        props = dict(boxstyle='round', facecolor='whitesmoke', alpha=0.5)
        ax[t_i//3, t_i%3].text(0.83, 0.97, r"$t=$"+str(t_s[t_i+1]*10), transform=ax[t_i//3, t_i%3].transAxes, fontsize=14, verticalalignment='top', bbox=props)
                           
        


    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color=plt.cm.tab10(0.9), marker="*", linestyle="", ms=25),
                    Line2D([0], [0], color=plt.cm.tab10(0.9), marker=".", linestyle="", ms=14, alpha=0.2)]


    if n == 30:
        fig.legend(custom_lines, ['Correctly classified data', 'Incorrectly classified data'], loc='upper center', ncol=4)

    plt.subplots_adjust(top=0.92, bottom=0.09, left=0.052, right=0.92, hspace=0.02, wspace=0.02)

    cbar_ax = fig.add_axes([0.925, 0.09, 0.02, 0.83])
    cb = fig.colorbar(im, cax=cbar_ax)
    cb.ax.set_ylabel("Network classification")
    
    plt.show()



    
