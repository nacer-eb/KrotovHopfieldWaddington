import sys
sys.path.append('../')

import numpy as np
from multiprocessing import Pool

from main_module.KrotovV2 import *


data_dir = "data/"

# Makes sure the data_dir exits else creates it.
if not path.exists(data_dir):
    print(data_dir, "Does not exist. It will be created ...")
    os.mkdir(data_dir)
    print(data_dir, "Created!")

isFirstRun = False

selected_digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]#
prefix = str(selected_digits)+"_saddles/" # I used main,and momentum #"main"#
n_range = np.arange(1, 41, 1)

noise_r = 5
def single_n(nT_merge):
    
    ic_range = [[1], [4, 7, 9]]
    for ic_i, ic in enumerate(ic_range):
        n, temp = nT_merge
        print(n, temp)
    
        r = 1.0/10**(noise_r)
        net = KrotovNet(Kx=1, Ky=1, n_deg=n, m_deg=n, M=20*len(selected_digits), nbMiniBatchs=1, momentum=0*0.6, rate=0.05,
                        temp=temp/(100.0**(1.0/n)), rand_init_mean=-0.003, rand_init_std=r, selected_digits=selected_digits)
            
            
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


        mean_Ts = np.zeros((10, 784))
        for d in range(0, 10):
            mean_Ts[d] = np.mean(data_T[20*d:20*(d+1), :], axis=0)

        net.hiddenDetectors[:, :] = -0.8
        net.hiddenDetectors[:, ic] = 0.8
        for sub_ic in ic: # Add ic we want to be near
            print("SUB_IC", sub_ic, len(ic))
            net.visibleDetectors[:, :] += (0.8/len(ic))*mean_Ts[sub_ic]

        for opp_ic in ic_range[1-ic_i]: # Remove ic we want to be away from
            net.visibleDetectors[:, :] -= (0.7/len(ic_range[1-ic_i]))*mean_Ts[opp_ic]


        net.train_plot_update(10000, isPlotting=False, isSaving=False, saving_dir=data_dir+prefix+"trained_net_n"+str(n)+"_T"+str(temp)+"_ic"+str(ic)+".npz", testFreq=500)
        net.train_plot_update(1, isPlotting=False, isSaving=True, saving_dir=data_dir+prefix+"trained_net_end_n"+str(n)+"_T"+str(temp)+"_ic"+str(ic)+".npz", testFreq=500)

            

if __name__ == '__main__':
    n_range = np.arange(1, 61, 1)
    temp_range = np.asarray([550, 650, 750])

    n, T = np.meshgrid(n_range, temp_range)
    nT_merge = np.asarray([n.flatten(), T.flatten()]).T
    
    with Pool(61) as p:
        p.map(single_n, nT_merge)
