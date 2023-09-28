import sys
sys.path.append('../')

import numpy as np
import sys

from main_module.KrotovV2 import *


data_dir = "data_test/"

# Makes sure the data_dir exits else creates it.
if not path.exists(data_dir):
    print(data_dir, "Does not exist. It will be created ...")
    os.mkdir(data_dir)
    print(data_dir, "Created!")

selected_digits = [4, 9]#
prefix = str(selected_digits)+"/" # I used main,and momentum #"main"#


for noise_r in [1]: # was 8.
    for temp in [840]:
        for n in [30]:
            for k in range(2):
                print(n, temp)
                 
                r = 1.0/10**(noise_r)
                net = KrotovNet(Kx=10, Ky=10, n_deg=n, m_deg=n, M=2, nbMiniBatchs=1, momentum=0*0.6,
                                rate=0.005, temp=temp, rand_init_mean=-0.003, rand_init_std=r, selected_digits=selected_digits)
                                

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

                print("INITIAL CONDITION ", k)
                # Init conditions
                net.miniBatchs_images = np.load(data_dir+"miniBatchs_images.npy")
                
                net.hiddenDetectors[:, :] += -0.3
                net.visibleDetectors[:, :] = 0.3*net.miniBatchs_images[0, k]

                net.hiddenDetectors[:, net.selected_digits[k]] += 0.3 # Silly mistake this should be 0.6 not 0.3

                net.train_plot_update(5000, isPlotting=False, isSaving=True, saving_dir=data_dir+prefix+"trained_net_n"+str(n)+"_T"+str(temp)+"ic"+str(selected_digits[k])+".npz", testFreq=400)
                net.train_plot_update(1, isPlotting=False, isSaving=True,
                                      saving_dir=data_dir+prefix+"trained_net_end_n"+str(n)+"_T"+str(temp)+"ic"+str(selected_digits[k])+".npz", testFreq=400)
                
