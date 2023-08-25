import sys
sys.path.append('../')

import numpy as np

from main_module.KrotovV2 import *

r = 1.0/10.0**(5)
temp = 800

n = 10

data_dir = "FirstDegree/"

net = KrotovNet(Kx=1, Ky=1, n_deg=n, m_deg=n, M=3, nbMiniBatchs=1, momentum=0*0.6,
                rate=0.003, temp=temp/( 3**(1.0/n) ), rand_init_mean=-0.001, rand_init_std=r, selected_digits=[1, 7, 9])
# First run
#np.save("miniBatchs_images.npy", net.miniBatchs_images);

net.miniBatchs_images = np.load("miniBatchs_images.npy");

net.show_minibatchs();

net.train_plot_update(1000, isPlotting=False, isSaving=True, saving_dir=data_dir+'test.npz', testFreq=200);

exit()

