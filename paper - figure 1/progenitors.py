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

from main_module.KrotovV2_utils import *

n, temp = 30, 670
data_dir = "data/"
subdir = "main/"
saving_dir = data_dir+subdir+"trained_net_n"+str(n)+"_T"+str(temp)+".npz"

data_M = np.load(saving_dir)['M']
data_L = np.load(saving_dir)['L']
data_T = np.load(data_dir+"miniBatchs_images.npy")[0]



t_range = np.arange(1000, 2800, 100)

for t in t_range:

    data_M_unique = np.zeros((1, 784))
    
    clone_of = np.zeros((len(data_M[t])))
    
    clone_of[0] = -1
    data_M_unique[0] = data_M[t, 0]


    for i in range(1, len(data_M[t])):
        print(i)
        isClone = False
        for j in range(0, len(data_M_unique)):
            diff = np.sum(np.abs(data_M_unique[j] - data_M[t, i]))
            
            tol = 0.15*784*np.mean(np.abs(data_M_unique[j]))
            
            if diff < tol:
                isClone = True
                clone_of[i] = j
                break
            
        if not isClone:
            clone_of[i] = -1
            data_M_unique = np.concatenate( [data_M_unique, [data_M[t, i]]] , axis=0)
        
    

    
    tab10_cmap = matplotlib.colormaps["tab10"]
    tab10_norm = matplotlib.colors.Normalize(0, 10)

    fig, ax = plt.subplots(2, len(data_M_unique)+1, figsize=(16, 9))
    
    progenitors = clone_of==-1
    for j in range(0, len(data_M_unique)):

        offspring = np.argmax(data_L[-1, clone_of==j, :], axis=-1)
        offspring = np.concatenate([offspring, [np.argmax(data_L[-1, progenitors, :][j], axis=-1)]]) # add offspring of curr progenitor j
        
        offspring_prop = np.zeros((10))
        for d in range(10):
            offspring_prop[d] = np.sum(offspring == d)/len(offspring)

    
        print(j, " is the progenitor of ", offspring_prop)
    
        ax[0, j].pie(offspring_prop, colors=tab10_cmap(tab10_norm(np.arange(0, 10, 1))))
        ax[1, j].imshow(data_M_unique[j].reshape(28, 28), cmap="bwr", vmin=-1, vmax=1)
        
        ax[0, j].set_title(str(len(offspring)))
    
    plt.show()

