import sys
sys.path.append('../')

import numpy as np

from main_module.KrotovV2 import *

data_dir = "data/"

dataset = "../defaults/miniBatchs_images_Fig_5.npy"

# Makes sure the data_dir exits else creates it.
if not path.exists(data_dir):
    print(data_dir, "Does not exist. It will be created ...")
    os.mkdir(data_dir)
    print(data_dir, "Created!")



for n in [3, 30]:
    r = 1.0/10.0**8
    train_rate = 0.005
    temp = 700
    
    net = KrotovNet(Kx=2, Ky=1, n_deg=n, m_deg=n, M=2, nbMiniBatchs=1, momentum=0*0.6, rate=train_rate, temp=temp,
                    rand_init_mean=-0.001, rand_init_std=r, selected_digits=[1, 4])
    
    net.miniBatchs_images[0] = np.load(dataset)[0]
    net.train_plot_update(3500, isPlotting=False, isSaving=True, saving_dir=data_dir+"trained_net_K"+str(net.K)+"_n"+str(n)+"_T"+str(temp)+".npz", testFreq=500)


    net = KrotovNet(Kx=1, Ky=1, n_deg=n, m_deg=n, M=2, nbMiniBatchs=1, momentum=0*0.6, rate=train_rate, temp=temp/(2.0**(1.0/n)),
                    rand_init_mean=-0.001, rand_init_std=r, selected_digits=[1, 4])
    
    net.miniBatchs_images[0] = np.load(dataset)[0]
    net.train_plot_update(3500, isPlotting=False, isSaving=True, saving_dir=data_dir+"trained_net_K"+str(net.K)+"_n"+str(n)+"_T"+str(temp)+".npz", testFreq=500)                     
 

