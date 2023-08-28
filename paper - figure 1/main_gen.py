import sys
sys.path.append('../')

import numpy as np

from main_module.KrotovV2 import *

data_dir = "data_100_10_200/"

# Makes sure the data_dir exits else creates it.
if not path.exists(data_dir):
    print(data_dir, "Does not exist. It will be created ...")
    os.mkdir(data_dir)
    print(data_dir, "Created!")

isFirstRun = False
prefix = "" # Can be "reproduced_"

for noise_r in [8]:
    for temp in [670]:
        for n in [3, 15, 30, 40]:
            print(n, temp)
            
            r = 1.0/10**(noise_r)
            net = KrotovNet(Kx=10, Ky=10, n_deg=n, m_deg=n, M=200, nbMiniBatchs=1, momentum=0*0.6, rate=0.005, temp=temp, rand_init_mean=-0.001, rand_init_std=r)
            
            # First run
            if isFirstRun:
                np.save(data_dir+"miniBatchs_images.npy", net.miniBatchs_images); exit()
            
            net.miniBatchs_images = np.load(data_dir+"miniBatchs_images.npy")

            
            run = 0

            if len(net.selected_digits) < 10:
                run = net.selected_digits # Essentially if this run contains only 1, 4, 7 you specify it in the file name
            
            net.train_plot_update(3500, isPlotting=False, isSaving=True, saving_dir=data_dir+prefix+"run_"+str(run)+"_n"+str(n)+"_T"+str(temp)+".npz", testFreq=100)

 

