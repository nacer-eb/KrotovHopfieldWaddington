import sys
sys.path.append('../')

import os

import numpy as np
from multiprocessing import Pool

from main_module.KrotovV2 import *


data_dir = "data/"


isFirstRun = False

selected_digits = [1, 7, 6]#[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]#[4, 6]# [1, 4] cool [1, 7] too# WARNING REMOVED 1
prefix = str(selected_digits)+"_saddles/" # I used main,and momentum #"main"#
n_range = np.arange(1, 41, 1)

noise_r = 5

subsample=1
def single_n(nT_merge):
    
    ic_range = [subsample]
    for ic_i, ic in enumerate(ic_range):
        n, temp = nT_merge
        print(n, temp)


        saving_dir=data_dir+prefix+"trained_net_end_n"+str(n)+"_T"+str(temp)+"_ic"+str(ic)+".npz"

        if os.path.isfile(saving_dir):
            print("File already exists..")
            continue
        
        
        r = 1.0/10**(noise_r)
        net = KrotovNet(Kx=1, Ky=1, n_deg=n, m_deg=n, M=20*len(selected_digits), nbMiniBatchs=1, momentum=0*0.6, rate=0.01,
                        temp=temp/((10.0*len(selected_digits))**(1.0/n)), rand_init_mean=-0.003, rand_init_std=r, selected_digits=selected_digits)
            
            
            
        # First run
        if isFirstRun:
            np.save(data_dir+"miniBatchs_images.npy", net.miniBatchs_images); exit()
                
        # Taking only the digits we need from the full training set
        data_T = np.load(data_dir+"miniBatchs_images.npy")[0]
        train_mask = np.zeros(200)
        for d in selected_digits:
            train_mask[d*20: (d+1)*20] = 1
        train_mask = np.asarray(train_mask, dtype=bool)
            

        net.miniBatchs_images[0] = data_T[train_mask]

        
        """
        means = np.zeros((10, 784))
        for d in range(10):
            means[d] = np.mean(data_T[20*d:20*(d+1)], axis=0)

        plt.imshow(means@means.T, cmap="bwr")
        plt.colorbar()
        plt.show()
        
        exit()
        
        """
        
        net.hiddenDetectors[:, :] = -0.99
        net.hiddenDetectors[:, ic//20] = 0.99

        net.visibleDetectors[:, :] += (0.98)*data_T[ic]

        net.train_plot_update(1000, isPlotting=False, isSaving=False, saving_dir=saving_dir, testFreq=500)
        net.train_plot_update(1, isPlotting=False, isSaving=True, saving_dir=saving_dir, testFreq=500)




if __name__ == '__main__':

    # Makes sure the data_dir exits else creates it.
    if not path.exists(data_dir):
        print(data_dir, "Does not exist. It will be created ...")
        os.mkdir(data_dir)
        print(data_dir, "Created!")
    
    # Makes sure the data_dir exits else creates it.
    if not path.exists(data_dir+prefix):
        print(data_dir+prefix, "Does not exist. It will be created ...")
        os.mkdir(data_dir+prefix)
        print(data_dir, "Created!")
        
           
    
    n_range = np.arange(2, 61, 1)
    temp_range = np.asarray([500, 700, 800]) #np.asarray([550, 650, 750])
    
    n, T = np.meshgrid(n_range, temp_range)
    nT_merge = np.asarray([n.flatten(), T.flatten()]).T

    for sub_i in range(0, 5):
        for i in selected_digits:
            subsample = 20*i+sub_i
            
            with Pool(61) as p:
                p.map(single_n, nT_merge)
        

