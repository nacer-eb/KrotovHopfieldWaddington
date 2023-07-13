import sys
sys.path.append('../')

import numpy as np
import sys

from main_module.KrotovV2 import *


data_dir = "data_14/"

# Makes sure the data_dir exits else creates it.
if not path.exists(data_dir):
    print(data_dir, "Does not exist. It will be created ...")
    os.mkdir(data_dir)
    print(data_dir, "Created!")




for i, n in enumerate([3]):
    
    noise_r = [10, 4][i]
    r = 1.0/10**(noise_r)
    temp = 700

    net = KrotovNet(Kx=1, Ky=1, n_deg=n, m_deg=n, M=2, nbMiniBatchs=1, momentum=0*0.6, rate=0.001, temp=temp/(2**(1.0/n)), rand_init_mean=-0.001, rand_init_std=r, selected_digits=[1, 4])
    net.miniBatchs_images = np.load("miniBatchs_images.npy")[:, [21, 41], :]
    
    net.train_plot_update(10000, isPlotting=False, isSaving=True, saving_dir=data_dir+"1m_l_n"+str(n)+"_T"+str(temp)+".npz", testFreq=100, l_condition=False)

            
            
            

            
 

