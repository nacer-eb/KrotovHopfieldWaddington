import sys
sys.path.append('../')

import os

import numpy as np
from multiprocessing import Pool

from main_module.KrotovV2 import *




data_dir = "data/"
        
selected_digits = [1, 7, 8]#
prefix = str(selected_digits)+"_long_3/" # I used main,and momentum #"main"#

prefix_run = "end_states_0/"

global_learning_rate = 0.01 

noise_r = 2 # lowered from 3
def single_run(nT):
    n, temp = nT
    
    for k in range(3):
        print(n, temp)
        
        r = 1.0/10**(noise_r)
        
        net = KrotovNet(Kx=10, Ky=10, n_deg=n, m_deg=n, M=len(selected_digits)*20, nbMiniBatchs=1, momentum=0*0.6,
                        rate=0.01, temp=temp, rand_init_mean=-0.03, rand_init_std=r, selected_digits=selected_digits)
        

        
        saving_dir=data_dir+prefix+prefix_run+"trained_net_end_n"+str(n)+"_T"+str(temp)+"ic"+str(selected_digits[k])+".npz"
        

        # Init conditions
        data_T = np.load(data_dir+prefix+"miniBatchs_images.npy")[0]
        train_mask = np.zeros(200)
        for d in selected_digits:
            train_mask[d*20: (d+1)*20] = 1
        train_mask = np.asarray(train_mask, dtype=bool)
            

        net.miniBatchs_images[0] = data_T[train_mask]

        A, B, C = np.mean(net.miniBatchs_images[0, :net.M//3], axis=0), np.mean(net.miniBatchs_images[0, net.M//3:2*net.M//3], axis=0), np.mean(net.miniBatchs_images[0, 2*net.M//3:], axis=0)
                
        net.hiddenDetectors[:, :] += -0.6
        net.hiddenDetectors[:, net.selected_digits[k]] += 2*0.6 # Silly mistake this should be 0.6 not 0.3 i.e. 2x

        
        if k == 0:
            net.visibleDetectors[:, :] = 0.8*A 
        if k == 1:
            net.visibleDetectors[:, :] = 0.8*B
        if k == 2:
            net.visibleDetectors[:, :] = 0.8*C
        

        """
        # non g noise
        # adding non gaussian noise
        coef = np.linspace(0, 1, net.K)
        for i in range(net.K):
            net.visibleDetectors[i, :] += 1*0.01*(  coef[i]*A + (1-coef[i])*B )
        """
        
        net.train_plot_update(10000, isPlotting=False, isSaving=False, # This needs  to be False False
                              saving_dir=data_dir+prefix+prefix_run+"trained_net_n"+str(n)+"_T"+str(temp)+"ic"+str(selected_digits[k])+".npz", testFreq=400)
        net.train_plot_update(1, isPlotting=False, isSaving=True,
                              saving_dir=saving_dir, testFreq=400) 
        
        
        
        
        ####### Saddles ########

        saving_dir=data_dir+prefix+"saddles/net_saddle_n"+str(n)+"_T"+str(temp)+"ic"+str(selected_digits[k])+".npz"

        if not os.path.isfile(saving_dir):
            print("From Scratch!")
            net = KrotovNet(Kx=1, Ky=1, n_deg=n, m_deg=n, M=20*len(selected_digits), nbMiniBatchs=1, momentum=0*0.6,
                            rate=0.01, temp=temp/(100.0**(1.0/n)), rand_init_mean=-0.003, rand_init_std=r, selected_digits=selected_digits)
        
            # Init conditions
            data_T = np.load(data_dir+prefix+"miniBatchs_images.npy")[0]
            train_mask = np.zeros(200)
            for d in selected_digits:
                train_mask[d*20: (d+1)*20] = 1
            train_mask = np.asarray(train_mask, dtype=bool)
            

            net.miniBatchs_images[0] = data_T[train_mask]

            
            A, B, C = np.mean(net.miniBatchs_images[0, :net.M//3], axis=0), np.mean(net.miniBatchs_images[0, net.M//3:2*net.M//3], axis=0), np.mean(net.miniBatchs_images[0, 2*net.M//3:], axis=0)
            
            net.hiddenDetectors[0, :] += -0.6
            net.hiddenDetectors[0, net.selected_digits[k]] += 0.6*2
            
            if k == 0:
                net.visibleDetectors[:, :] = 0.8*A 
            if k == 1:
                net.visibleDetectors[:, :] = 0.8*B
            if k == 2:
                net.visibleDetectors[:, :] = 0.8*C
            
            net.train_plot_update(4000, isPlotting=False, isSaving=False,
                                  saving_dir=saving_dir, testFreq=400)
            net.train_plot_update(1, isPlotting=False, isSaving=True,
                                  saving_dir=saving_dir, testFreq=400)
            


if __name__ == "__main__":
    
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
        
    # Makes sure the data_dir exits else creates it.
    if not path.exists(data_dir+prefix+"saddles/"):
        print(data_dir+prefix+"saddles/", "Does not exist. It will be created ...")
        os.mkdir(data_dir+prefix+"saddles/")
        print(data_dir, "Created!")
    
    
    for run in range(10):
        prefix_run = "end_states_ric_g_" +str(run) + "/"
        
        # Makes sure the data_dir exits else creates it.
        if not path.exists(data_dir+prefix+prefix_run):
            print(data_dir+prefix+prefix_run, "Does not exist. It will be created ...")
            os.mkdir(data_dir+prefix+prefix_run)
            print(data_dir, "Created!")


        
        n_range = np.arange(2, 60, 1)    
        temp_range = np.asarray([550, 750, 800]) #np.arange(700, 900, 20)[::2] #np.arange(600, 900, 20)[::2]
        
        n, T = np.meshgrid(n_range, temp_range)
        nT_merge = np.asarray([n.flatten(), T.flatten()]).T
        
        with Pool(61) as p:
            p.map(single_run, nT_merge)
                
        
    
    
