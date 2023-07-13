import sys
sys.path.append('../')

import numpy as np

from main_module.KrotovV2 import *

data_dir = "data_2_N_2/"#"data_2_2_2/"

# Makes sure the data_dir exits else creates it.
if not path.exists(data_dir):
    print(data_dir, "Does not exist. It will be created ...")
    os.mkdir(data_dir)
    print(data_dir, "Created!")


tmp_x = np.linspace(-1, 1, 25)

#n, T = 3, 700 #12, 900 #7, 1200

n_range = np.arange(1, 61, 3)
temp_range = np.arange(700, 1000, 40)

for n in n_range:
    for T in temp_range:
        net = KrotovNet(Kx=2, Ky=1, n_deg=n, m_deg=n, M=10*2, nbMiniBatchs=1, momentum=0*0.6, rate=0.01, temp=T, rand_init_mean=-0.03, rand_init_std=0.01, selected_digits=[1, 4])
        
        net.miniBatchs_images[0, 0:10, :] = np.load("miniBatchs_images.npy")[0, 20:20*2-10]
        net.miniBatchs_images[0, 10:, :] = np.load("miniBatchs_images.npy")[0, 20*4:20*5-10]
        
    
        net.train_plot_update(4000, isPlotting=False, isSaving=False, saving_dir=data_dir+"long_n"+str(n)+"_T"+str(T)+".npz")
        net.train_plot_update(1, isPlotting=False, isSaving=True, saving_dir=data_dir+"n"+str(n)+"_T"+str(T)+".npz")


exit()
