import sys
sys.path.append('../')

import numpy as np

from main_module.KrotovV2 import *

data_dir = "data_100_10_200_supplemental/"

# Makes sure the data_dir exits else creates it.
if not path.exists(data_dir):
    print(data_dir, "Does not exist. It will be created ...")
    os.mkdir(data_dir)
    print(data_dir, "Created!")


for noise_r in [5]: # was 8.
    for temp in [400, 550, 670]:#np.arange(400, 680, 20):
        for n in [3, 15, 30]:#np.arange(2, 32, 2):
            print(n, temp)
            
            r = 1.0/10**(noise_r)
            net = KrotovNet(Kx=10, Ky=10, n_deg=n, m_deg=n, M=200, nbMiniBatchs=1, momentum=0*0.6, rate=0.005, temp=temp, rand_init_mean=-0.001, rand_init_std=r)#, selected_digits=[4, 9])
            
            # First run
            #np.save(data_dir+"miniBatchs_images.npy", net.miniBatchs_images); exit()
            
            net.miniBatchs_images = np.load(data_dir+"miniBatchs_images.npy")
            
            
            run=0
            #net.train_plot_update(3500, isPlotting=False, isSaving=True, saving_dir=data_dir+"run_"+str(run)+"_n"+str(n)+"_T"+str(temp)+".npz", testFreq=100)
            net.train_plot_update(5000, isPlotting=False, isSaving=True, saving_dir=data_dir+"run_"+str(run)+"_n"+str(n)+"_T"+str(temp)+".npz", testFreq=100)

            
 

