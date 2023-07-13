import sys
sys.path.append('../')

import numpy as np

from main_module.KrotovV2 import *

data_dir = "data_2_2_2/"#"data_2_2_2/"

# Makes sure the data_dir exits else creates it.
if not path.exists(data_dir):
    print(data_dir, "Does not exist. It will be created ...")
    os.mkdir(data_dir)
    print(data_dir, "Created!")


l_0, alpha_0 = 0.095, 0.561 # 0.055, 0.476 
beta_0 = 1 - alpha_0

tmp_x = np.linspace(-1, 1, 25)

n, T = 20, 700 #12, 900 #7, 1200
for i in range(0, 1):
    net = KrotovNet(Kx=2, Ky=1, n_deg=n, m_deg=n, M=2, nbMiniBatchs=1, momentum=0*0.6, rate=0.0001, temp=T, rand_init_mean=-0.03, rand_init_std=0.03, selected_digits=[1, 4])

    net.miniBatchs_images = np.load(data_dir+"miniBatchs_images.npy")

    
    A, B = net.miniBatchs_images[0, 0], net.miniBatchs_images[0, 1]

    d_AA, d_AB, d_BB = A@A, A@B, B@B
    d_BA = d_AB

    
    print(d_AA, d_AB, d_BB, n, T)

    d_alpha = 0.023#0.1
    d_l_0 = -0.1#0.0001

    net.visibleDetectors[0, :] = (alpha_0 + d_alpha) * A + (beta_0 - d_alpha) * B
    net.visibleDetectors[1, :] = (alpha_0 - d_alpha) * A + (beta_0 + d_alpha) * B

    net.hiddenDetectors[0, :] = -1
    net.hiddenDetectors[0, net.selected_digits[0]] = l_0 + d_l_0
    net.hiddenDetectors[0, net.selected_digits[1]] = -l_0 - d_l_0
    
    net.hiddenDetectors[1, :] = -1
    net.hiddenDetectors[1, net.selected_digits[0]] = l_0 - d_l_0
    net.hiddenDetectors[1, net.selected_digits[1]] = -l_0 + d_l_0
    
    net.train_plot_update(4000, isPlotting=False, isSaving=True, saving_dir=data_dir+"test20.npz")


exit()
