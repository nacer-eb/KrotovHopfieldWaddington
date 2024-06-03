import sys
sys.path.append('../')

import numpy as np

from main_module.KrotovV2 import *

data_dir = "data/"

dataset = "../defaults/miniBatchs_images_Fig_6.npy"

# Makes sure the data_dir exits else creates it.
if not path.exists(data_dir):
    print(data_dir, "Does not exist. It will be created ...")
    os.mkdir(data_dir)
    print(data_dir, "Created!")


for noise_r in [8]:
    for temp in [700]:
        for n in [15, 39]:
            alpha = 0.8
            l_0 = 0.5
            
            r = 1.0/10**(noise_r)

            dl = r
            da = r

            train_rate = 0.0004
            if n == 39:
                train_rate *= 2
            
            net = KrotovNet(Kx=2, Ky=1, n_deg=n, m_deg=n, M=2, nbMiniBatchs=1, momentum=0*0.6, rate=train_rate, temp=temp, rand_init_mean=-0.001, rand_init_std=r, selected_digits=[1, 4])

            
            net.miniBatchs_images[0] = np.load(dataset)[0]
            
            
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
                    
 

