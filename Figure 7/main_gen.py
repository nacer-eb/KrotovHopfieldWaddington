import argparse
import sys
sys.path.append('../')

import numpy as np
from multiprocessing import Pool

from main_module.KrotovV2 import *


data_dir = "data/"

dataset = "../dataset/miniBatchs_images.npy"

# The digit classes to include in the training
selected_digits = [1, 7, 8] 
prefix = str(selected_digits)+"/"
prefix_run = "end_states_0/"

# The number of processes to run in parralel, defaults to 1
poolsize = 1

parser = argparse.ArgumentParser(description="This program runs the simulations for Figure 7.")
parser.add_argument('--poolsize', help="The number of processes to run at once, on large servers 61 process is good enough. [DEFAULT=1]", default=1, type=int)
parse_args = parser.parse_args()

poolsize = parse_args.poolsize

initial_noise = 1.0/10**2
    
# This defines a function to simulate many runs in parralel
def single_n(nT_merge):
    n, temp = nT_merge

    # k controls initial conditions
    for k in range(3):
        net = KrotovNet(Kx=10, Ky=10, n_deg=n, m_deg=n, M=20*len(selected_digits), nbMiniBatchs=1, momentum=0*0.6,
                        rate=0.01, temp=temp, rand_init_mean=-0.03, rand_init_std=initial_noise, selected_digits=selected_digits)
            
        
        # Taking only the digits we need from the full training set
        data_T = np.load(data_dir+prefix+dataset)[0]
        train_mask = np.zeros(200)
        for d in selected_digits:
            train_mask[d*20: (d+1)*20] = 1
        train_mask = np.asarray(train_mask, dtype=bool)
        
        net.miniBatchs_images[0] = data_T[train_mask]


        # Setting up A, B, C as the average 1, 7, and 8
        A = np.mean(net.miniBatchs_images[0, :net.M//3], axis=0),
        B = np.mean(net.miniBatchs_images[0, net.M//3:2*net.M//3], axis=0)
        C = np.mean(net.miniBatchs_images[0, 2*net.M//3:], axis=0)


        # Depending on the initial conditions start labels closer to 1, 7, or 8
        net.hiddenDetectors[:, :] += -0.6
        net.hiddenDetectors[:, net.selected_digits[k]] += 2*0.6

        # Depending on the initial conditions start memories closer to 1, 7, or 8
        if k == 0:
            net.visibleDetectors[:, :] = 0.8*A
        if k == 1:
            net.visibleDetectors[:, :] = 0.8*B
        if k == 2:
            net.visibleDetectors[:, :] = 0.8*C
        

        # Save the initial conditions in case you need to debug
        net.train_plot_update(1, isPlotting=False, isSaving=True,
                              saving_dir=data_dir+prefix+prefix_run+"trained_net_ic_n"+str(n)+"_T"+str(temp)+"ic"+str(selected_digits[k])+".npz", testFreq=500)

        # Run the training, but don,t save every epoch
        net.train_plot_update(10000, isPlotting=False, isSaving=False,
                              saving_dir=data_dir+prefix+prefix_run+"trained_net_n"+str(n)+"_T"+str(temp)+"ic"+str(selected_digits[k])+".npz", testFreq=500)

        # Save the final state for analysis
        net.train_plot_update(1, isPlotting=False, isSaving=True,
                              saving_dir=data_dir+prefix+prefix_run+"trained_net_end_n"+str(n)+"_T"+str(temp)+"ic"+str(selected_digits[k])+".npz", testFreq=500)
    


        # Compute saddles if not already done once.
        saving_dir=data_dir+prefix+"saddles/net_saddle_n"+str(n)+"_T"+str(temp)+"ic"+str(selected_digits[k])+".npz"
        if not os.path.isfile(saving_dir):
            print("Computing saddles!")

            # Notie the number of memories and the temperature!
            net = KrotovNet(Kx=1, Ky=1, n_deg=n, m_deg=n, M=20*len(selected_digits), nbMiniBatchs=1, momentum=0*0.6,
                            rate=0.01, temp=temp/(100.0**(1.0/n)), rand_init_mean=-0.003, rand_init_std=r, selected_digits=selected_di\
gits)

            # Again making sure only the useful training data is kept
            data_T = np.load(data_dir+prefix+dataset)[0]
            train_mask = np.zeros(200)
            for d in selected_digits:
                train_mask[d*20: (d+1)*20] = 1
            train_mask = np.asarray(train_mask, dtype=bool)

            net.miniBatchs_images[0] = data_T[train_mask]

        # Preparing the basis for the initial conditions
        A = np.mean(net.miniBatchs_images[0, :net.M//3], axis=0),
        B = np.mean(net.miniBatchs_images[0, net.M//3:2*net.M//3], axis=0)
        C = np.mean(net.miniBatchs_images[0, 2*net.M//3:], axis=0)


        # Initial conditions for the saddles is the same as for the entire system
        net.hiddenDetectors[:, :] += -0.6
        net.hiddenDetectors[:, net.selected_digits[k]] += 2*0.6 # The 2* is important

        if k == 0:
            net.visibleDetectors[:, :] = 0.8*A
        if k == 1:
            net.visibleDetectors[:, :] = 0.8*B
        if k == 2:
            net.visibleDetectors[:, :] = 0.8*C
        

        # Run the training but don't save every epoch
        net.train_plot_update(4000, isPlotting=False, isSaving=False,
                              saving_dir=data_dir+prefix+"saddles/net_saddle_n"+str(n)+"_T"+str(temp)+"ic"+str(selected_digits[k])+".npz", testFreq=500)

        # Save the final state
        net.train_plot_update(1, isPlotting=False, isSaving=True, saving_dir=saving_dir, testFreq=500)
            
        
# For multiprocessing
if __name__ == '__main__':

    # Makes sure the data_dir exits else creates it.
    if not path.exists(data_dir):
        print(data_dir, "Does not exist. It will be created ...")
        os.mkdir(data_dir)
        print(data_dir, "Created!")

    # Makes sure the data_dir/prefix exits else creates it.
    if not path.exists(data_dir+prefix):
        print(data_dir+prefix, "Does not exist. It will be created ...")
        os.mkdir(data_dir+prefix)
        print(data_dir, "Created!")

    # Makes sure the data_dir/prefix/saddles exits else creates it.
    if not path.exists(data_dir+prefix+"saddles/"):
        print(data_dir+prefix+"saddles/", "Does not exist. It will be created ...")
        os.mkdir(data_dir+prefix+"saddles/")
        print(data_dir, "Created!")

    # The parameter range we want to simulate
    n_range = np.arange(2, 60, 1)
    temp_range = np.asarray([550, 750, 800])

    n, T = np.meshgrid(n_range, temp_range)
    nT_merge = np.asarray([n.flatten(), T.flatten()]).T        

    # We want to average over many runs
    for run in range(10):
        prefix_run = "end_states_" +str(run) + "/"

        # Makes sure the data_dir/prefix/prefix_run exits else creates it.
        if not path.exists(data_dir+prefix+prefix_run):
            print(data_dir+prefix+prefix_run, "Does not exist. It will be created ...")
            os.mkdir(data_dir+prefix+prefix_run)
            print(data_dir, "Created!")
        

        # Create a pool of all hyperparameters and start processes (based on the maximum poolsize)
        with Pool(poolsize) as p:
            p.map(single_n, nT_merge)

        

