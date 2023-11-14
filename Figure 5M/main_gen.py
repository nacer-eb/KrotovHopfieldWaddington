import argparse
import sys
sys.path.append('../')

import numpy as np
from multiprocessing import Pool

from main_module.KrotovV2 import *


data_dir = "data/"
dataset = "../defaults/miniBatchs_images_Fig_5.npy"

# The digit classes to include in the training
selected_digits = [1, 4] 
prefix = str(selected_digits)+"/" 

# The number of processes to run in parralel, defaults to 1
poolsize = 1

parser = argparse.ArgumentParser(description="This program runs the simulations for Figure 1.")
parser.add_argument('--poolsize', help="The number of processes to run at once. [DEFAULT=1]", default=1, type=int)
parse_args = parser.parse_args()

poolsize = parse_args.poolsize

initial_noise = 1.0/10**8
    
# This defines a function to simulate many runs in parralel
def single_n(nT_merge):
    n, temp = nT_merge

    for ell in [-0.8, 0.8]:
        for alpha in [0.2, 0.5, 0.8]:
    
            net = KrotovNet(Kx=2, Ky=1, n_deg=n, m_deg=n, M=2, nbMiniBatchs=1, momentum=0*0.6, rate=0.005,
                            temp=temp, rand_init_mean=-0.003, rand_init_std=initial_noise, selected_digits=selected_digits)
            

            net.miniBatchs_images = np.load(dataset)

            # Initial Conditions
            A, B = net.miniBatchs_images[0, 0], net.miniBatchs_images[0, 1]

            net.hiddenDetectors[:, [0, 2, 3, 5, 6, 7, 8, 9]] = -1
    
            net.hiddenDetectors[:, net.selected_digits[0]] += ell
            net.hiddenDetectors[:, net.selected_digits[1]] += -ell

            net.visibleDetectors[:, :] += alpha*A + (1-np.abs(alpha))*B
            
            net.train_plot_update(1000, isPlotting=False, isSaving=True,
                                  saving_dir=data_dir+prefix+"trained_net_n"+str(n)+"_T"+str(temp)+"_ell"+str(ell)+"_alpha"+str(alpha)+".npz", testFreq=500)
    



if __name__ == '__main__':
    n_range = np.asarray([6, 10, 20, 30, 35, 37, 38, 40]) # You can add more n if you want, this is the strictly necessary for Fig 4
    temp_range = np.asarray([700])    
    
    n, T = np.meshgrid(n_range, temp_range)
    nT_merge = np.asarray([n.flatten(), T.flatten()]).T

    # Makes sure the data_dir exits else creates it.
    if not path.exists(data_dir):
        print(data_dir, "Does not exist. It will be created ...")
        os.mkdir(data_dir)
        print(data_dir, "Created!")
    
    # Makes sure the data_dir/prefix exits else creates it.
    if not path.exists(data_dir+prefix):
        print(data_dir+prefix, "Does not exist. It will be created ...")
        os.mkdir(data_dir+prefix)
        print(data_dir+prefix, "Created!")
        
        net = KrotovNet(Kx=10, Ky=10, M=20*len(selected_digits), nbMiniBatchs=1, momentum=0*0.6,
                        rate=0.005, rand_init_mean=-0.003, rand_init_std=initial_noise, selected_digits=selected_digits)
        # Creates details.md
        details_file = open(data_dir+prefix+"details.md", "w")
        details = "The networks contain " + str(net.K) + " memories. \n" +\
            "Are initialized with Gaussian initial conditions, N("+str(net.rand_init_mean)+", "+str(net.rand_init_std)+"). \n" +\
            "The learning rate is " + str(net.train_rate) + ".\n" +\
            "The momentum is " + str(net.train_momentum) + ".\n" +\
            "The digits included in training are " + str(net.selected_digits) + ".\n" +\
            "Specifics can be checked by opening the file with KV_window..."
        
        details_file.write(details)               


    # The number of cores should be changeable
    with Pool(poolsize) as p:
        p.map(single_n, nT_merge)

        

