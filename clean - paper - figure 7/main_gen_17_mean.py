import sys
sys.path.append('../')

import numpy as np

from main_module.KrotovV2 import *

data_dir = "data_17_mean/"

# Makes sure the data_dir exits else creates it.
if not path.exists(data_dir):
    print(data_dir, "Does not exist. It will be created ...")
    os.mkdir(data_dir)
    print(data_dir, "Created!")

for noise_r in [5]: # was 8.
    for temp in np.arange(500, 800, 20)[7:]: #np.arange(500, 800, 20):#[800]:#np.arange(400, 680, 20):
        for n in np.arange(2, 31, 2):#np.arange(2, 32, 2):
            for k, run in enumerate([1, 7]):
                print(n, temp)
                
                r = 1.0/10**(noise_r)
                net = KrotovNet(Kx=10, Ky=10, n_deg=n, m_deg=n, M=2, nbMiniBatchs=1, momentum=0*0.6, rate=0.001, temp=temp, rand_init_mean=-0.001, rand_init_std=r, selected_digits=[1, 7])
                # First run
                #np.save(data_dir+"miniBatchs_images.npy", net.miniBatchs_images); exit()
                
                

                net.miniBatchs_images[0, 0] = np.mean(np.load(data_dir+"miniBatchs_images.npy")[0, 20*1:20*2], axis=0)
                net.miniBatchs_images[0, 1] = np.mean(np.load(data_dir+"miniBatchs_images.npy")[0, 20*7:20*8], axis=0)

                #net.show_minibatchs()

                #"""
                for i in range(0, net.K):
                    net.visibleDetectors[i, :] = 0.7*net.miniBatchs_images[0, k] + i*0.0001*net.miniBatchs_images[0, 0] + (net.K-i)*0.0001*net.miniBatchs_images[0, 1]
                #net.visibleDetectors[1, :] = 0.5*train_data[1]

                net.hiddenDetectors[:, :] = -1
                net.hiddenDetectors[:, net.selected_digits] = -0.001
                #"""
                

                net.train_plot_update(10000, isPlotting=False, isSaving=False, saving_dir=data_dir+"run_l_"+str(run)+"_n"+str(n)+"_T"+str(temp)+".npz", testFreq=100, l_condition=False)
                net.train_plot_update(1, isPlotting=False, isSaving=True, saving_dir=data_dir+"run_"+str(run)+"_n"+str(n)+"_T"+str(temp)+".npz", testFreq=100)
                #net.train_plot_update(1, isPlotting=False, isSaving=True, saving_dir=data_dir+"run_"+str(run)+"_n"+str(n)+"_T"+str(temp)+".npz", testFreq=100)
                #net.train_plot_update(5000, isPlotting=False, isSaving=True, saving_dir=data_dir+"run_"+str(run)+"_n"+str(n)+"_T"+str(temp)+".npz", testFreq=100)

            
 

