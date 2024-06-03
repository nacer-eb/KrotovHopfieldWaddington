import argparse
import sys
sys.path.append('../')

import numpy as np
from multiprocessing import Process

from main_module.KrotovV2 import *
from generate_umap_embedding import *


data_dir = "data/"

dataset = "../defaults/miniBatchs_images.npy"

# The digit classes to include in the training
selected_digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] 
prefix = str(selected_digits)+"/" 


n, temp = 30, 670
initial_noise = 1.0/10**3



# This defines a function to simulate many runs in parralel
def single_n(init_M, init_L):
    
    net = KrotovNet(Kx=10, Ky=10, n_deg=n, m_deg=n, M=20*len(selected_digits), nbMiniBatchs=1, momentum=0*0.6, rate=0.002, temp=temp, rand_init_mean=0, rand_init_std=0, selected_digits=selected_digits)
            
        
    # Taking only the digits we need from the full training set
    data_T = np.load(dataset)[0]
    train_mask = np.zeros(200)
    for d in selected_digits:
        train_mask[d*20: (d+1)*20] = 1
    train_mask = np.asarray(train_mask, dtype=bool)


    net.visibleDetectors = init_M
    net.hiddenDetectors = init_L

    # First no noise version
    net.train_plot_update(1000, isPlotting=False, isSaving=True, saving_dir=data_dir+prefix+"trained_net_n"+str(n)+"_T"+str(temp)+".npz", testFreq=500, noiseStd=0.0, pulsePeriod=10000)
    


# This defines a function to simulate many runs in parralel
def single_n_noisy(init_M, init_L):
    
    net2 = KrotovNet(Kx=10, Ky=10, n_deg=n, m_deg=n, M=20*len(selected_digits), nbMiniBatchs=1, momentum=0*0.6, rate=0.002, temp=temp, rand_init_mean=0, rand_init_std=0, selected_digits=selected_digits)
            
        
    # Taking only the digits we need from the full training set
    data_T = np.load(dataset)[0]
    train_mask = np.zeros(200)
    for d in selected_digits:
        train_mask[d*20: (d+1)*20] = 1
    train_mask = np.asarray(train_mask, dtype=bool)


    net2.visibleDetectors = init_M
    net2.hiddenDetectors = init_L

    # First no noise version
    net2.train_plot_update(1000, isPlotting=False, isSaving=True, saving_dir=data_dir+prefix+"trained_net_noisy_n"+str(n)+"_T"+str(temp)+".npz", testFreq=500, noiseStd=0.0, pulsePeriod=10000)





init_M = np.random.normal(-0.003, initial_noise, (100, 784)).round(5)
init_L = np.random.normal(-0.003, initial_noise, (100, 10)).round(5)


single_n_noisy(init_M, init_L)
single_n(init_M, init_L)
