import sys
sys.path.append('../')

import numpy as np

from main_module.KrotovV2 import *

r = 1.0/10.0**(5)
temp = 800


data_dir = "FirstDegree/"


for n in np.arange(5, 60, 2):
    for a_i, a in enumerate(np.arange(0, 1, 0.1)):
        for b_i, b in enumerate(np.arange(0, 1-a, 0.1)):

            print("--> n : ", n, " <-----------")
            net = KrotovNet(Kx=1, Ky=1, n_deg=n, m_deg=n, M=3, nbMiniBatchs=1, momentum=0*0.6,
                            rate=0.01, temp=temp/( 3**(1.0/n) ), rand_init_mean=-0.001, rand_init_std=r, selected_digits=[1, 7, 9])
            # First run
            #np.save("miniBatchs_images.npy", net.miniBatchs_images);
            
            net.miniBatchs_images = np.load("miniBatchs_images.npy");
            #net.show_minibatchs();
            
            # Initial Conditions
            A, B, C = net.miniBatchs_images[0]
            
            net.visibleDetectors[0] = a*A + b*B + (1 - a - b)*C
            
            net.train_plot_update(5000, isPlotting=False, isSaving=True, saving_dir=data_dir+"n"+str(n)+"_a"+str(a_i)+"_b"+str(b_i)+"_save.npz", testFreq=2000);
exit()

