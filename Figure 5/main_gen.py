import argparse
import sys
sys.path.append('../')

import numpy as np
from multiprocessing import Pool

from main_module.KrotovV2 import *
from generate_umap_embedding import *


data_dir = "data/"

dataset = "../defaults/miniBatchs_images.npy"

# The digit classes to include in the training
selected_digits = [1, 4, 7] 
prefix = str(selected_digits)+"/" 

# The number of processes to run in parralel, defaults to 1
poolsize = 1

parser = argparse.ArgumentParser(description="This program runs the simulations for Figure 4.")
parser.add_argument('--poolsize', help="The number of processes to run at once. [DEFAULT=1]", default=1, type=int)
parse_args = parser.parse_args()

poolsize = parse_args.poolsize

initial_noise = 1.0/10**5
    
# This defines a function to simulate many runs in parralel
def single_n(nT_merge):
    n, temp = nT_merge
    print(n, temp)
    
    net = KrotovNet(Kx=10, Ky=10, n_deg=n, m_deg=n, M=20*len(selected_digits), nbMiniBatchs=1, momentum=0*0.6, rate=0.005*0.7, temp=temp, rand_init_mean=-0.003, rand_init_std=initial_noise, selected_digits=selected_digits)
    
    
    # Taking only the digits we need from the full training set
    data_T = np.load(dataset)[0]
    train_mask = np.zeros(200)
    for d in selected_digits:
        train_mask[d*20: (d+1)*20] = 1
    train_mask = np.asarray(train_mask, dtype=bool)
        

    net.miniBatchs_images[0] = data_T[train_mask]


    # Saves initial conditions in case we need to debug
    net.train_plot_update(1, isPlotting=False, isSaving=True, saving_dir=data_dir+prefix+"trained_net_ic_n"+str(n)+"_T"+str(temp)+".npz", testFreq=500)

     # Trains the network - 3500 seems to be good enough and saves space 
    net.train_plot_update(3500, isPlotting=False, isSaving=True, saving_dir=data_dir+prefix+"trained_net_n"+str(n)+"_T"+str(temp)+".npz", testFreq=500)
    
    # This crashes sometimes but still saves... (if it crashes before saving see the README for known issues/fixes)
    generate_umap_embedding(data_dir, prefix, n, temp, verbose=True)
    



if __name__ == '__main__':
    n_range = np.asarray([3, 30]) # You can add more n if you want, this is the strictly necessary for Fig 4
    temp_range = np.asarray([800])    
    
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

        

