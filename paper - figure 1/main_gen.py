import sys
sys.path.append('../')

import numpy as np

from main_module.KrotovV2 import *
from generate_umap_embedding import *


data_dir = "data/"

# Makes sure the data_dir exits else creates it.
if not path.exists(data_dir):
    print(data_dir, "Does not exist. It will be created ...")
    os.mkdir(data_dir)
    print(data_dir, "Created!")

isFirstRun = False

selected_digits = [1, 4, 7] # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]#
prefix = str(selected_digits)+"/" # I used main,and momentum #"main"#


for noise_r in [10]:
    for temp in [800]:
        for n in [3, 15, 30, 40]:
            print(n, temp)
            
            r = 1.0/10**(noise_r)
            net = KrotovNet(Kx=10, Ky=10, n_deg=n, m_deg=n, M=20*len(selected_digits), nbMiniBatchs=1, momentum=0*0.6, rate=0.005, temp=temp, rand_init_mean=-0.001, rand_init_std=r, selected_digits=selected_digits)
            
            
            # Makes sure the data_dir exits else creates it.
            if not path.exists(data_dir+prefix):
                print(data_dir, "Does not exist. It will be created ...")
                os.mkdir(data_dir+prefix)
                print(data_dir, "Created!")

                # Creates details.md
                details_file = open(data_dir+prefix+"details.md", "w")
                details = "The networks contain " + str(net.K) + " memories. \n" +\
                    "Are initialized with Gaussian initial conditions, N("+str(net.rand_init_mean)+", "+str(net.rand_init_std)+"). \n" +\
                    "The learning rate is " + str(net.train_rate) + ".\n" +\
                    "The momentum is " + str(net.train_momentum) + ".\n" +\
                    "The digits included in training are " + str(net.selected_digits) + ".\n" +\
                    "Specifics can be checked by opening the file with KV_window..."

                details_file.write(details)
                


            
           # First run
            if isFirstRun:
                np.save(data_dir+"miniBatchs_images.npy", net.miniBatchs_images); exit()

            # Taking only the digits we need from the full training set
            data_T = np.load(data_dir+"miniBatchs_images.npy")[0]
            train_mask = np.zeros(200)
            for d in selected_digits:
                train_mask[d*20: (d+1)*20] = 1
            train_mask = np.asarray(train_mask, dtype=bool)


            net.miniBatchs_images[0] = data_T[train_mask]
            
            net.train_plot_update(3500, isPlotting=False, isSaving=True, saving_dir=data_dir+prefix+"trained_net_n"+str(n)+"_T"+str(temp)+".npz", testFreq=100)

            generate_umap_embedding(data_dir, prefix, n, temp, verbose=True)
            

