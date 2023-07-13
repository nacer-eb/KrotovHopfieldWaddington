import sys
sys.path.append('../')

import numpy as np

from main_module.KrotovV2 import *

data_dir = "data_100_10_200/"#"data_222_identical/"#"data_10t/"#"data_222_pop_dynamics_uncontrolled/"#"data_20_10_20/"
#data_dir = "data_10M_10D_10E_small_noise/"

# Makes sure the data_dir exits else creates it.
if not path.exists(data_dir):
    print(data_dir, "Does not exist. It will be created ...")
    os.mkdir(data_dir)
    print(data_dir, "Created!")


for noise_r in [8]:#range(9, 11):
    for temp in [670]:#[400, 550, 670]:#[800, 850]:
        for n in [30]:#[3, 15, 30]:#[5, 10, 30]:#np.arange(2, 32, 1):
            print(n, temp)
            
            r = 1.0/10**(noise_r)
            net = KrotovNet(Kx=10, Ky=10, n_deg=n, m_deg=n, M=200, nbMiniBatchs=1, momentum=0*0.6, rate=0.005, temp=temp, rand_init_mean=-0.001, rand_init_std=r)#, selected_digits=[4, 9])
            
            # First run
            #np.save(data_dir+"miniBatchs_images.npy", net.miniBatchs_images); exit()
            
            net.miniBatchs_images = np.load(data_dir+"miniBatchs_images.npy")

            """
            for i in range(0, net.K):                
                net.visibleDetectors[i, :] = ( (0.1-r) * np.mean(net.miniBatchs_images[0, net.M//10:2*net.M//10, :], axis=0)
                                              - r*np.random.randn(784))
                
                net.hiddenDetectors[i, :] = -0.03 + r*np.random.randn(len(net.selected_digits))

            """


            run = 0
            net.train_plot_update(3500, isPlotting=False, isSaving=True, saving_dir=data_dir+"reproduced_run_"+str(run)+"_n"+str(n)+"_T"+str(temp)+".npz", testFreq=100)

 

