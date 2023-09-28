import sys
sys.path.append('../')

import numpy as np
from multiprocessing import Pool

from main_module.KrotovV2 import *




data_dir = "data/"
        
selected_digits = [4, 9]#
prefix = str(selected_digits)+"/" # I used main,and momentum #"main"#

noise_r = 2
def single_run(nT):
    n, temp = nT
    
    for k in range(2):
        print(n, temp)
        
        r = 1.0/10**(noise_r)
        net = KrotovNet(Kx=10, Ky=10, n_deg=n, m_deg=n, M=2, nbMiniBatchs=1, momentum=0*0.6,
                        rate=0.005, temp=temp, rand_init_mean=-0.003, rand_init_std=r, selected_digits=selected_digits)
        

        
        # Init conditions
        net.miniBatchs_images = np.load(data_dir+"miniBatchs_images.npy")
        
        net.hiddenDetectors[:, :] += -0.9
        net.visibleDetectors[:, :] += 0.9*net.miniBatchs_images[0, k]
        net.hiddenDetectors[:, net.selected_digits[k]] += 0.9*2 # Silly mistake this should be 0.6 not 0.3
        
        net.train_plot_update(5000, isPlotting=False, isSaving=False,
                              saving_dir=data_dir+prefix+"trained_net_n"+str(n)+"_T"+str(temp)+"ic"+str(selected_digits[k])+".npz", testFreq=400)
        net.train_plot_update(1, isPlotting=False, isSaving=True,
                              saving_dir=data_dir+prefix+"trained_net_end_n"+str(n)+"_T"+str(temp)+"ic"+str(selected_digits[k])+".npz", testFreq=400)
                
    
        
        
        ####### Saddles ########
        
        net = KrotovNet(Kx=1, Ky=1, n_deg=n, m_deg=n, M=2, nbMiniBatchs=1, momentum=0*0.6,
                        rate=0.005, temp=temp/(100.0**(1.0/n)), rand_init_mean=-0.003, rand_init_std=r, selected_digits=selected_digits)
        
        # Init conditions
        net.miniBatchs_images = np.load(data_dir+"miniBatchs_images.npy")
        
        net.hiddenDetectors[0, :] += -0.9
        net.visibleDetectors[0, :] += 0.9*net.miniBatchs_images[0, k]
        net.hiddenDetectors[0, net.selected_digits[k]] += 0.9
        
        net.train_plot_update(5000, isPlotting=False, isSaving=False,
                              saving_dir=data_dir+prefix+"net_saddle_n"+str(n)+"_T"+str(temp)+"ic"+str(selected_digits[k])+".npz", testFreq=400)
        net.train_plot_update(1, isPlotting=False, isSaving=True,
                              saving_dir=data_dir+prefix+"net_saddle_n"+str(n)+"_T"+str(temp)+"ic"+str(selected_digits[k])+".npz", testFreq=400)


if __name__ == "__main__":
    
    n_range = np.arange(2, 32, 1)    
    temp_range = np.arange(500, 900, 20)

    n, T = np.meshgrid(n_range, temp_range)
    nT_merge = np.asarray([n.flatten(), T.flatten()]).T

    with Pool(61) as p:
        p.map(single_run, nT_merge)
    
    
