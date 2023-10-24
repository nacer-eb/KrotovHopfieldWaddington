import sys
sys.path.append('../')

import numpy as np
import sys

from multiprocessing import Pool

from main_module.KrotovV2 import *


selected_digits = [5, 8]

data_dir = "data_"+str(selected_digits)+"/"


def single_run(nT):
    noise_r = 4
    n, temp = nT
                 
    r = 1.0/10**(noise_r)
    net = KrotovNet(Kx=2, Ky=1, n_deg=n, m_deg=n, M=2, nbMiniBatchs=1, momentum=0*0.6, rate=0.001, temp=temp, rand_init_mean=-0.001, rand_init_std=r, selected_digits=selected_digits)

    # 
    """
    net.training_diversification_2(tol_self=710, tol_mix=600, tol_delta_self=10)
    net.show_minibatchs()
    np.save(data_dir+"miniBatchs_images.npy", net.miniBatchs_images);
    
    
    data_T = np.load(data_dir+"miniBatchs_images.npy")
    A = np.mean(data_T[0, :20], axis=0)
    B = np.mean(data_T[0, 20:], axis=0)

    net.miniBatchs_images[0, 0] = A
    net.miniBatchs_images[0, 1] = B
    np.save(data_dir+"miniBatchs_images.npy", net.miniBatchs_images); exit()
    """
    
    #net.show_minibatchs()   
    #np.save(data_dir+"miniBatchs_images.npy", net.miniBatchs_images); exit()
    # First run
    #net.training_diversification_2(tol=70)
    #net.training_diversification(tol_min=690, tol_max=800, width_difference=100)
    #
    
    #mean_miniBatch = np.zeros((1, 2, 784))
    #mean_miniBatch[0, 0] = np.mean(net.miniBatchs_images[0, 0:100], axis=0)
    #mean_miniBatch[0, 1] = np.mean(net.miniBatchs_images[0, 100:], axis=0)
    
    
    #plt.imshow(merge_data(mean_miniBatch[0, :], 1, 2), cmap="bwr", vmin=-1, vmax=1)
    #plt.show()
    #np.save(data_dir+"miniBatchs_images.npy", mean_miniBatch); exit()
    
    
    
    
    net.miniBatchs_images = np.load(data_dir+"miniBatchs_images.npy")
    
    net.hiddenDetectors[:, :] = -1
    net.visibleDetectors[0, :] = 0.5*net.miniBatchs_images[0, 0] - 0.2*net.miniBatchs_images[0, 1]
    net.visibleDetectors[1, :] = 0.5*net.miniBatchs_images[0, 1] - 0.2*net.miniBatchs_images[0, 0]

    net.hiddenDetectors[0, net.selected_digits[0]] = 1; net.hiddenDetectors[1, net.selected_digits[1]] = 1 
            
    
    net.train_plot_update(15000, isPlotting=False, isSaving=False, saving_dir=data_dir+"l_n"+str(n)+"_T"+str(temp)+".npz", testFreq=400)
    net.train_plot_update(1, isPlotting=False, isSaving=True, saving_dir=data_dir+"n"+str(n)+"_T"+str(temp)+".npz", testFreq=400)


 

if __name__ == "__main__":

    # Makes sure the data_dir exits else creates it.
   
    if not path.exists(data_dir):
        print(data_dir, "Does not exist. It will be created ...")
        os.mkdir(data_dir)
        print(data_dir, "Created!")

   
    #single_run([2, 400]); exit()
        
    n_range = np.arange(2, 31, 1)
    temp_range = np.arange(500, 1000, 20)

    n, T = np.meshgrid(n_range, temp_range)
    nT_merge = np.asarray([n.flatten(), T.flatten()]).T
    
    with Pool(61*4) as p:
        p.map(single_run, nT_merge)


