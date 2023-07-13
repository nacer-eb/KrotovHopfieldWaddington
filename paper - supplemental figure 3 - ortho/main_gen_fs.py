import sys
sys.path.append('../')

import numpy as np
import sys

from main_module.KrotovV2 import *


job_number = int(sys.argv[1])
data_dir = "data_49_rand/"

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
            net = KrotovNet(Kx=2, Ky=1, n_deg=n, m_deg=n, M=2, nbMiniBatchs=1, momentum=0*0.6, rate=0.001, temp=temp, rand_init_mean=-0.001, rand_init_std=r, selected_digits=[4, 9])
            
            # First run
            #net.training_diversification_2(tol=70)
            #net.training_diversification(tol_min=690, tol_max=800, width_difference=100)
            #net.show_minibatchs()

            #mean_miniBatch = np.zeros((1, 2, 784))
            #mean_miniBatch[0, 0] = np.mean(net.miniBatchs_images[0, 0:100], axis=0)
            #mean_miniBatch[0, 1] = np.mean(net.miniBatchs_images[0, 100:], axis=0)

            
            #plt.imshow(merge_data(mean_miniBatch[0, :], 1, 2), cmap="bwr", vmin=-1, vmax=1)
            #plt.show()
            #np.save(data_dir+"miniBatchs_images.npy", mean_miniBatch); exit()
            
            #np.save(data_dir+"miniBatchs_images.npy", net.miniBatchs_images); exit()
            
            
            net.miniBatchs_images = np.load(data_dir+"miniBatchs_images.npy")
            
            net.hiddenDetectors[:, :] = -1
            net.visibleDetectors[0, :] = 0.5*net.miniBatchs_images[0, 0]; net.visibleDetectors[1, :] = 0.5*net.miniBatchs_images[0, 1]
            net.hiddenDetectors[0, net.selected_digits[0]] = 1; net.hiddenDetectors[1, net.selected_digits[1]] = 1 
            
            
            net.train_plot_update(10000, isPlotting=False, isSaving=False, saving_dir=data_dir+"l_n"+str(n)+"_T"+str(temp)+".npz", testFreq=100, l_condition=False)
            net.train_plot_update(1, isPlotting=False, isSaving=True, saving_dir=data_dir+"n"+str(n)+"_T"+str(temp)+".npz", testFreq=100)

            
 

