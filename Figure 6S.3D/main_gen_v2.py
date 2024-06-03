import sys
sys.path.append('../')

import numpy as n
from multiprocessing import Pool

from main_module.KrotovV2 import *

data_dir = "data/"

dataset = "../defaults/miniBatchs_images_Fig_6.npy"

# Makes sure the data_dir exits else creates it.
if not path.exists(data_dir):
    print(data_dir, "Does not exist. It will be created ...")
    os.mkdir(data_dir)
    print(data_dir, "Created!")


n =  20
noise_r = 3 # 5 - for the 0.01 - 0.1 case
temp = 700
def single_init_run(alpha_l_0):
    alpha, l_0 = alpha_l_0

    if alpha == 0.01:
        noise_r = 5
    
    r = 1.0/10**(noise_r)
    
    dl = r
    da = r

    #dl, da = -0.04*0.1, 0.004*0.1
    
    train_rate = 0.001
    
    net = KrotovNet(Kx=2, Ky=1, n_deg=n, m_deg=n, M=2, nbMiniBatchs=1, momentum=0*0.6, rate=train_rate, temp=temp, rand_init_mean=-0.001, rand_init_std=r, selected_digits=[1, 4])

    
    net.miniBatchs_images[0] = np.load(dataset)[0]
    
    
    A, B = net.miniBatchs_images[0, 0], net.miniBatchs_images[0, 1]
    
    net.hiddenDetectors[:, :] = -1
    
    net.hiddenDetectors[0, net.selected_digits[0]] = l_0 + dl
    net.hiddenDetectors[0, net.selected_digits[1]] = -(l_0 - dl)
    
    net.hiddenDetectors[1, net.selected_digits[0]] = l_0 - dl
    net.hiddenDetectors[1, net.selected_digits[1]] = -(l_0 + dl)

    """
    np.random.seed()
    net.hiddenDetectors[:, net.selected_digits] = np.random.randn(2)*0.001 - 0.1
    """

    net.visibleDetectors[0, :] = (alpha + da) * A + (1 - np.abs(alpha+da)) * B
    net.visibleDetectors[1, :] = (alpha - da) * A + (1 - np.abs(alpha-da)) * B


    
    saving_dir = data_dir + "run_" + str(net.selected_digits) \
        + "_n" + str(n) \
        + "_T" + str(temp) \
        + "_alpha" + str(alpha) \
        + "_l_0" + str(l_0) \
        + ".npz"
    
    net.train_plot_update(1000, isPlotting=False, isSaving=True, saving_dir=saving_dir, testFreq=100)



if __name__ == '__main__':

    alphas_ells = [[0.4, 0.5], [0.6, -0.2], [0.01, -0.04]]

    # The number of cores should be changeable
    with Pool(61) as p:
        p.map(single_init_run, alphas_ells)

