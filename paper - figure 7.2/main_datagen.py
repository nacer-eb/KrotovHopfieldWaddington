import sys
sys.path.append('../')

import os

import numpy as np
from multiprocessing import Pool

from main_module.KrotovV2 import *




data_dir = "data/"
        
selected_digits = [1, 7]#
prefix = str(selected_digits)+"/" # I used main,and momentum #"main"#

prefix_run = "end_states_0/"

global_learning_rate = 0.01 

noise_r = 2 # lowered from 3
def single_run(nT):
    n, temp = nT
    
    for k in range(2):
        print(n, temp)
        
        r = 1.0/10**(noise_r)
        net = KrotovNet(Kx=10, Ky=10, n_deg=n, m_deg=n, M=2, nbMiniBatchs=1, momentum=0*0.6,
                        rate=0.01, temp=temp, rand_init_mean=-0.03, rand_init_std=r, selected_digits=selected_digits)
        


        """
        #data fetching
        print(net.miniBatchs_images[0]@net.miniBatchs_images[0].T)
        net.training_diversification(630, 650, width_difference=1)
        net.show_minibatchs()
        np.save(data_dir + prefix + "miniBatchs_images.npy", net.miniBatchs_images)
        """
        

        saving_dir=data_dir+prefix+prefix_run+"trained_net_end_n"+str(n)+"_T"+str(temp)+"ic"+str(selected_digits[k])+".npz"

        # Init conditions
        net.miniBatchs_images = np.load(data_dir+prefix+"miniBatchs_images.npy")

        A, B = net.miniBatchs_images[0]
        
        
        net.hiddenDetectors[:, :] += -0.6
        net.visibleDetectors[:, :] += 0.6*net.miniBatchs_images[0, k]
        net.hiddenDetectors[:, net.selected_digits[k]] += 2*0.6 # Silly mistake this should be 0.6 not 0.3 i.e.


        # adding non gaussian noise
        coef = np.linspace(0, 1, net.K)
        for i in range(net.K):
            net.visibleDetectors[i, :] += 0.01*(  coef[i]*A + (1-coef[i])*B )
        
        
        net.train_plot_update(10000, isPlotting=False, isSaving=False,
                              saving_dir=data_dir+prefix+prefix_run+"trained_net_n"+str(n)+"_T"+str(temp)+"ic"+str(selected_digits[k])+".npz", testFreq=400)
        net.train_plot_update(1, isPlotting=False, isSaving=True,
                              saving_dir=saving_dir, testFreq=400)
                
    
        
        
        ####### Saddles ########

        saving_dir=data_dir+prefix+"saddles/net_saddle_n"+str(n)+"_T"+str(temp)+"ic"+str(selected_digits[k])+".npz"

        if not os.path.isfile(saving_dir):
            print("From Scratch!")
            net = KrotovNet(Kx=1, Ky=1, n_deg=n, m_deg=n, M=2, nbMiniBatchs=1, momentum=0*0.6,
                            rate=0.01, temp=temp/(100.0**(1.0/n)), rand_init_mean=-0.003, rand_init_std=r, selected_digits=selected_digits)
        
            # Init conditions
            net.miniBatchs_images = np.load(data_dir+prefix+"miniBatchs_images.npy")
            
            net.hiddenDetectors[0, :] += -0.6
            net.visibleDetectors[0, :] += 0.6*net.miniBatchs_images[0, k]
            net.hiddenDetectors[0, net.selected_digits[k]] += 0.6*2
            
            net.train_plot_update(20000, isPlotting=False, isSaving=False,
                                  saving_dir=saving_dir, testFreq=400)
            net.train_plot_update(1, isPlotting=False, isSaving=True,
                                  saving_dir=saving_dir, testFreq=400)
            


if __name__ == "__main__":
    
    # Makes sure the data_dir exits else creates it.
    if not path.exists(data_dir+prefix+"saddles/"):
        print(data_dir, "Does not exist. It will be created ...")
        os.mkdir(data_dir+prefix+prefix_run)
        print(data_dir, "Created!")
    
    for run in range(0, 100):
        prefix_run = "end_states_" +str(run) + "/"
        
        # Makes sure the data_dir exits else creates it.
        if not path.exists(data_dir+prefix+prefix_run):
            print(data_dir, "Does not exist. It will be created ...")
            os.mkdir(data_dir+prefix+prefix_run)
            print(data_dir, "Created!")

       
        n_range = np.arange(2, 62, 1)[::1]    
        temp_range = np.arange(700, 900, 20)[::1] #np.arange(600, 900, 20)[::2]
    
        n, T = np.meshgrid(n_range, temp_range)
        nT_merge = np.asarray([n.flatten(), T.flatten()]).T
    
        with Pool(61) as p:
            p.map(single_run, nT_merge)
        
            
    
    
