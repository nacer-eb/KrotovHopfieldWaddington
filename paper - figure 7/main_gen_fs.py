import sys
sys.path.append('../')

import numpy as np
import sys

from main_module.KrotovV2 import *


job_number = int(sys.argv[1])
data_dir = "data_17_"+str(job_number)+"/"#"data_17_mean/"

# Makes sure the data_dir exits else creates it.
if not path.exists(data_dir):
    print(data_dir, "Does not exist. It will be created ...")
    os.mkdir(data_dir)
    print(data_dir, "Created!")
    
if job_number == 1:
    temp_range = np.arange(500, 1100, 20)[:15]
    n_range = np.arange(2, 61, 2)[:15]

if job_number == 2:
    temp_range = np.arange(500, 1100, 20)[:15]
    n_range = np.arange(2, 61, 2)[15:]

if job_number == 3:
    temp_range = np.arange(500, 1100, 20)[15:]
    n_range = np.arange(2, 61, 2)[:15]

if job_number == 4:
    temp_range = np.arange(500, 1100, 20)[15:]
    n_range = np.arange(2, 61, 2)[15:]

    #4
for noise_r in [4]: # was 8.
    for temp in temp_range:
        for n in n_range:
            for k, run in enumerate([1, 7]):
                print(n, temp)
                 
                r = 1.0/10**(noise_r)
                net = KrotovNet(Kx=10, Ky=10, n_deg=n, m_deg=n, M=2, nbMiniBatchs=1, momentum=0*0.6, rate=0.001, temp=temp, rand_init_mean=-0.001, rand_init_std=r, selected_digits=[1, 4])
                # First run
                #np.save(data_dir+"miniBatchs_images.npy", net.miniBatchs_images); exit()
                
                net.miniBatchs_images[0, 0] = np.load("miniBatchs_images.npy")[0, 20*1+1]
                net.miniBatchs_images[0, 1] = np.load("miniBatchs_images.npy")[0, 20*4+1]

                #net.show_minibatchs()

                #"""
                for i in range(0, net.K):
                    net.visibleDetectors[i, :] += 0.7*net.miniBatchs_images[0, k] # + i*0.001*net.miniBatchs_images[0, 0] + (net.K-i)*0.001*net.miniBatchs_images[0, 1]
                #net.visibleDetectors[1, :] = 0.5*train_data[1]
                
                net.hiddenDetectors[:, [0, 2, 3, 5, 6, 7, 8, 9]] = -1
                #net.hiddenDetectors[:, net.selected_digits] = -0.001
                #"""
                

                for t in range(0, 8):
                    net.train_plot_update(5000, isPlotting=False, isSaving=False, saving_dir=data_dir+"run_l_"+str(run)+"_n"+str(n)+"_T"+str(temp)+".npz", testFreq=100, l_condition=False)
                net.train_plot_update(1, isPlotting=False, isSaving=True, saving_dir=data_dir+"run_"+str(run)+"_n"+str(n)+"_T"+str(temp)+".npz", testFreq=100)

            
 

