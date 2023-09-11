import sys
sys.path.append('../')

import numpy as np

from main_module.KrotovV2 import *

data_dir = "data_1_10_200_179_long/"

# Makes sure the data_dir exits else creates it.
if not path.exists(data_dir):
    print(data_dir, "Does not exist. It will be created ...")
    os.mkdir(data_dir)
    print(data_dir, "Created!")

for noise_r in [5]: # was 8.
    for temp in [800]:#np.arange(400, 680, 20):
        for n in np.arange(2, 61, 0.1):#np.arange(2, 32, 2):
            for run in [1, 7]:
                print(n, temp)
                
                r = 1.0/10**(noise_r)
                net = KrotovNet(Kx=1, Ky=1, n_deg=n, m_deg=n, M=20*3, nbMiniBatchs=1, momentum=0*0.6, rate=0.03, temp=temp/( 100**(1.0/n) ), rand_init_mean=-0.001, rand_init_std=r, selected_digits=[1, 7, 9])
                # First run
                #np.save(data_dir+"miniBatchs_images.npy", net.miniBatchs_images); exit()
                
                
                take_1479 = np.zeros((200), dtype=int)
                for d in net.selected_digits:
                    take_1479[20*d:20*(d+1)] = True # Take d-th digit
                take_1479 = np.asarray(take_1479, dtype=bool)
                
                net.miniBatchs_images[0] = np.load(data_dir+"miniBatchs_images.npy")[0, take_1479]
                #net.miniBatchs_images = np.load(data_dir+"miniBatchs_images.npy")
                

                average_1 = np.mean(net.miniBatchs_images[0, 0:20, :], axis=0)
                average_7 = np.mean(net.miniBatchs_images[0, 20:40, :], axis=0)
                average_9 = np.mean(net.miniBatchs_images[0, 40:, :], axis=0)

                
                net.hiddenDetectors[0, :] = -1
                net.hiddenDetectors[0, net.selected_digits] = 0 


                if run == 1:
                    net.visibleDetectors[0, :] = average_1 - 0.4*average_7 - 0.4*average_9
                if run == 7:
                    net.visibleDetectors[0, :] = -0.4*average_1 + average_7 - 0.4*average_9
                if run == 9:
                    net.visibleDetectors[0, :] = -0.4*average_1 - 0.4*average_7 + average_9
                   
                
                net.visibleDetectors[0] /= 1.1*np.max(np.abs(net.visibleDetectors[0]))
                

                
                
                net.train_plot_update(1000, isPlotting=False, isSaving=True, saving_dir=data_dir+"run_long_"+str(run)+"_n"+str(n)+"_T"+str(temp)+".npz", testFreq=100)
                net.train_plot_update(1, isPlotting=False, isSaving=True, saving_dir=data_dir+"run_"+str(run)+"_n"+str(n)+"_T"+str(temp)+".npz", testFreq=100)
                #net.train_plot_update(5000, isPlotting=False, isSaving=True, saving_dir=data_dir+"run_"+str(run)+"_n"+str(n)+"_T"+str(temp)+".npz", testFreq=100)

            
 

