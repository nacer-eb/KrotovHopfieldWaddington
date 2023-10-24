import sys
sys.path.append('../')

import numpy as np

from main_module.KrotovV2 import *

data_dir = "data_2_2_2/"

# Makes sure the data_dir exits else creates it.
if not path.exists(data_dir):
    print(data_dir, "Does not exist. It will be created ...")
    os.mkdir(data_dir)
    print(data_dir, "Created!")


for noise_r in [8]:#range(9, 11):
    for temp in [700]:#[400, 550, 670]:#[800, 850]:
        for n in [15, 30]:#[30]:#[3, 15, 30]:#[5, 10, 30]:#np.arange(2, 32, 1):
            for alpha in [0.2, 0.4, 0.6, 0.8]:
                for l_0 in [-0.5, 0.5]:
                    print(n, temp)
                    
                    r = 1.0/10**(noise_r)
                    dl = 0*r
                    da = 0*r
                    
                    net = KrotovNet(Kx=2, Ky=1, n_deg=n, m_deg=n, M=2, nbMiniBatchs=1, momentum=0*0.6, rate=0.0004, temp=temp, rand_init_mean=-0.001, rand_init_std=r, selected_digits=[1, 4])
                    
                    # First run
                    #np.save(data_dir+"miniBatchs_images.npy", net.miniBatchs_images); exit()

                    #"""
                    
                    take_14 = np.zeros((200), dtype=int)
                    take_14[20*1 + 1] = 1 # Take a 1
                    take_14[20*4 + 1] = 1 # Take a 4.
                    take_14 = np.asarray(take_14, dtype=bool)
                    #"""
                    
                    net.miniBatchs_images[0] = np.load(data_dir+"miniBatchs_images.npy")[0, take_14]
                    A, B = net.miniBatchs_images[0, 0], net.miniBatchs_images[0, 1]
                                        
                    net.hiddenDetectors[:, :] = -1
                    net.hiddenDetectors[0, net.selected_digits[0]] = l_0 + dl
                    net.hiddenDetectors[0, net.selected_digits[1]] = -(l_0 - dl)

                    net.hiddenDetectors[1, net.selected_digits[0]] = l_0 - dl
                    net.hiddenDetectors[1, net.selected_digits[1]] = -(l_0 + dl)

                    
                    net.visibleDetectors[0, :] = (alpha + da) * A + (1 - np.abs(alpha+da)) * B
                    net.visibleDetectors[1, :] = (alpha - da) * A + (1 - np.abs(alpha-da)) * B
                    
                    saving_dir = data_dir + "run_" + str(net.selected_digits) \
                        + "_n" + str(n) \
                        + "_T" + str(temp) \
                        + "_alpha" + str(alpha) \
                        + "_l_0" + str(l_0) \
                        + ".npz"
                    
                    net.train_plot_update(3500, isPlotting=False, isSaving=True, saving_dir=saving_dir, testFreq=100)
                    
 

