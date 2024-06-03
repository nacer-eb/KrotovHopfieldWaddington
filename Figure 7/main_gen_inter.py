import argparse
import sys
sys.path.append('../')

import numpy as np
from multiprocessing import Pool

from main_module.KrotovV2 import *


data_dir = "data/"

# The digit classes to include in the training
selected_digits = [4, 9] # It's important (because of how my code is written) that the first class is the same as the class used in the intra case.
prefix = str(selected_digits)+"_mean_inter/"

# The number of processes to run in parralel, defaults to 1
poolsize = 1

parser = argparse.ArgumentParser(description="This program runs the simulations for Figure 7.")
parser.add_argument('--poolsize', help="The number of processes to run at once. [DEFAULT=1]", default=1, type=int)
parse_args = parser.parse_args()

poolsize = parse_args.poolsize

initial_noise = 1.0/10**5
    
# This defines a function to simulate many runs in parralel
def single_n(nT_merge):
    n, temp = nT_merge
    print(n, temp)
    
    net = KrotovNet(Kx=2, Ky=1, n_deg=n, m_deg=n, M=2, nbMiniBatchs=1, momentum=0*0.6, rate=0.001,
                    temp=temp, rand_init_mean=-0.003, rand_init_std=initial_noise, selected_digits=selected_digits)
    

    # Load the training dataset
    net.miniBatchs_images = np.load(data_dir+prefix+"miniBatchs_images.npy")


    # Initial conditions
    net.hiddenDetectors[:, :] = -1

    # Initialize the memories as each near a different digit, helps make the analysis code simpler, but has no effect on the results (we tried it with fully random conditions)
    net.visibleDetectors[0, :] = 0.5*net.miniBatchs_images[0, 0] - 0.2*net.miniBatchs_images[0, 1]
    net.visibleDetectors[1, :] = 0.5*net.miniBatchs_images[0, 1] - 0.2*net.miniBatchs_images[0, 0]

    # Same for the Labels
    net.hiddenDetectors[0, net.selected_digits[0]] = 1; net.hiddenDetectors[1, net.selected_digits[1]] = 1



    # Save initial conditons for debugging
    net.train_plot_update(1, isPlotting=False, isSaving=True, saving_dir=data_dir+prefix+"trained_net_ic_n"+str(n)+"_T"+str(temp)+".npz", testFreq=500)

    # Train the network
    net.train_plot_update(10000, isPlotting=False, isSaving=False, saving_dir=data_dir+prefix+"trained_net_n"+str(n)+"_T"+str(temp)+".npz", testFreq=500)

    # Save the final state for the analysis
    net.train_plot_update(1, isPlotting=False, isSaving=True, saving_dir=data_dir+prefix+"trained_net_end_n"+str(n)+"_T"+str(temp)+".npz", testFreq=500)




if __name__ == '__main__':
    n_range = np.arange(2, 31, 1)
    temp_range = np.arange(400, 900, 20)

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

    
    # If this is the first run and the training data doesn't exist, creates it!
    if not os.path.isfile(data_dir+prefix+"miniBatchs_images.npy"):
        net = KrotovNet(Kx=1, Ky=1, n_deg=30, m_deg=30, M=2*50, nbMiniBatchs=1, momentum=0*0.6, rate=0.001, temp=600,
                        rand_init_mean=-0.001, rand_init_std=0.01, selected_digits=selected_digits)
        
        miniBatchs_images_mean = np.zeros((1, 2, 784))

        miniBatchs_images_mean[0, 0] = np.mean(net.miniBatchs_images[0, :net.M//2], axis=0)
        miniBatchs_images_mean[0, 1] = np.mean(net.miniBatchs_images[0, net.M//2:], axis=0)

        np.save(data_dir+prefix+"miniBatchs_images.npy", miniBatchs_images_mean);

    # Creates the pool for multiprocessing
    with Pool(poolsize) as p:
        p.map(single_n, nT_merge)

        

