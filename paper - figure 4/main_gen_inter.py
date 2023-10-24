import sys
sys.path.append('../')

import numpy as np
import sys
from multiprocessing import Pool

from main_module.KrotovV2 import *

# Change this if you're NOT using a supplied minibatch/training_sample AND don't want mean digits in the newly generated one
isMeanMinibatch = True

selected_digits = [4, 9]
data_dir = "data_"+str(selected_digits)+"_mean_inter/"
    
def single_n(nT_merge):
    n, temp = nT_merge
    print(n, temp)

    noise_r = 4

        
    r = 1.0/10**(noise_r)
    net = KrotovNet(Kx=2, Ky=1, n_deg=n, m_deg=n, M=2, nbMiniBatchs=1, momentum=0*0.6, rate=0.001, temp=temp, rand_init_mean=-0.001, rand_init_std=r, selected_digits=selected_digits)
    
    # First run
    #net.show_minibatchs()
    #np.save(data_dir+"miniBatchs_images.npy", net.miniBatchs_images); exit()
    
    net.miniBatchs_images = np.load(data_dir+"miniBatchs_images.npy")

    net.hiddenDetectors[:, :] = -1
    net.visibleDetectors[0, :] = 0.5*net.miniBatchs_images[0, 0] - 0.2*net.miniBatchs_images[0, 1]
    net.visibleDetectors[1, :] = 0.5*net.miniBatchs_images[0, 1] - 0.2*net.miniBatchs_images[0, 0]

    net.hiddenDetectors[0, net.selected_digits[0]] = 1; net.hiddenDetectors[1, net.selected_digits[1]] = 1 

    net.train_plot_update(1, isPlotting=False, isSaving=True, saving_dir=data_dir+"trained_net_ic_n"+str(n)+"_T"+str(temp)+".npz", testFreq=500)
    net.train_plot_update(10000, isPlotting=False, isSaving=False, saving_dir=data_dir+"trained_net_n"+str(n)+"_T"+str(temp)+".npz", testFreq=500)
    net.train_plot_update(1, isPlotting=False, isSaving=True, saving_dir=data_dir+"trained_net_end_n"+str(n)+"_T"+str(temp)+".npz", testFreq=500)
    




if __name__ == '__main__':

    # Makes sure the data_dir exits else creates it.
    if not path.exists(data_dir):
        print(data_dir, "Does not exist. It will be created ...")
        os.mkdir(data_dir)
        print(data_dir, "Created!")


    # If there is no pre-existing minibatch create it
    if not os.path.isfile(data_dir+"miniBatchs_images.npy") and not isMeanMinibatch:
        net = KrotovNet(Kx=1, Ky=1, n_deg=30, m_deg=30, M=2, nbMiniBatchs=1, momentum=0*0.6, rate=0.001, temp=600, rand_init_mean=-0.001, rand_init_std=0.01, selected_digits=selected_digits)
        np.save(data_dir+"miniBatchs_images.npy", net.miniBatchs_images);

    if not os.path.isfile(data_dir+"miniBatchs_images.npy") and isMeanMinibatch:
        net = KrotovNet(Kx=1, Ky=1, n_deg=30, m_deg=30, M=2*20, nbMiniBatchs=1, momentum=0*0.6, rate=0.001, temp=600, rand_init_mean=-0.001, rand_init_std=0.01, selected_digits=selected_digits)
        miniBatchs_images_mean = np.zeros((1, 2, 784))

        miniBatchs_images_mean[0, 0] = np.mean(net.miniBatchs_images[0, :net.M//2], axis=0)
        miniBatchs_images_mean[0, 1] = np.mean(net.miniBatchs_images[0, net.M//2:], axis=0)

        np.save(data_dir+"miniBatchs_images.npy", miniBatchs_images_mean);



    n_range = np.arange(2, 31, 1)
    temp_range = np.arange(400, 900, 20)
    
    n, T = np.meshgrid(n_range, temp_range)
    nT_merge = np.asarray([n.flatten(), T.flatten()]).T
    
    with Pool(61) as p:
        p.map(single_n, nT_merge)
