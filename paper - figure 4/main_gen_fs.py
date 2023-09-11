import sys
sys.path.append('../')

import numpy as np
import sys

from main_module.KrotovV2 import *


job_number = int(sys.argv[1])
data_dir = "data_4_"+str(job_number)+"/"#"data_17_mean/"

n_range = np.arange(2, 31, 1)
temp_range = np.arange(400, 900, 20)

# Makes sure the data_dir exits else creates it.
if not path.exists(data_dir):
    print(data_dir, "Does not exist. It will be created ...")
    os.mkdir(data_dir)
    print(data_dir, "Created!")

if job_number == 1:
    temp_range = temp_range[:len(temp_range)//2] #15
    n_range = n_range[:len(n_range)//2]

if job_number == 2:
    temp_range = temp_range[:len(temp_range)//2]
    n_range = n_range[len(n_range)//2:]

if job_number == 3:
    temp_range = temp_range[len(temp_range)//2:]
    n_range = n_range[:len(n_range)//2]

if job_number == 4:
    temp_range = temp_range[len(temp_range)//2:]
    n_range = n_range[len(n_range)//2:]


# Makes sure the data_dir exits else creates it.
if not path.exists(data_dir):
    print(data_dir, "Does not exist. It will be created ...")
    os.mkdir(data_dir)
    print(data_dir, "Created!")
    

for noise_r in [4]: # was 8.
    for temp in temp_range:
        for n in n_range:
            print(n, temp)
                 
            r = 1.0/10**(noise_r)
            net = KrotovNet(Kx=1, Ky=1, n_deg=n, m_deg=n, M=2, nbMiniBatchs=1, momentum=0*0.6, rate=0.001, temp=temp, rand_init_mean=-0.001, rand_init_std=r, selected_digits=[4, 4])

            plt.imshow(net.visibleDetectors[0].reshape(28, 28), cmap="bwr", vmin=-1, vmax=1)
            plt.show()
            
            exit()
            # First run
            #net.show_minibatchs()
            #np.save(data_dir+"miniBatchs_images.npy", net.miniBatchs_images); exit()
            
            net.miniBatchs_images = np.load("miniBatchs_images.npy")           
            
            
            
            net.train_plot_update(10000, isPlotting=False, isSaving=False, saving_dir=data_dir+"l_n"+str(n)+"_T"+str(temp)+".npz", testFreq=100, l_condition=False)
            net.train_plot_update(1, isPlotting=False, isSaving=True, saving_dir=data_dir+"n"+str(n)+"_T"+str(temp)+".npz", testFreq=100)

            
 

