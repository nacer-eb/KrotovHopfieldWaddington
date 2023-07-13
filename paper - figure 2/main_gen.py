import sys
sys.path.append('../')

import numpy as np

from main_module.KrotovV2 import *

data_dir = "data_100_10_200/"#"data_222_identical/"#"data_10t/"#"data_222_pop_dynamics_uncontrolled/"#"data_20_10_20/"
#data_dir = "data_10M_10D_10E_small_noise/"

# Makes sure the data_dir exits else creates it.
if not path.exists(data_dir):
    print(data_dir, "Does not exist. It will be created ...")
    os.mkdir(data_dir)
    print(data_dir, "Created!")


for noise_r in [9]:#range(9, 11):
    for temp in [800]: #[670]:#[400, 550, 670]:#[800, 850]:
        for n in [33]:#[3, 15, 30]:#[5, 10, 30]:#np.arange(2, 32, 1):
            print(n, temp)
            
            r = 1.0/10**(noise_r)
            net = KrotovNet(Kx=10, Ky=10, n_deg=n, m_deg=n, M=20*3, nbMiniBatchs=1, momentum=0*0.6, rate=0.005,
                            temp=temp, rand_init_mean=-0.001, rand_init_std=r, selected_digits=[1, 9, 4])

            # First run
            #np.save(data_dir+"miniBatchs_images.npy", net.miniBatchs_images); exit()

            take_1479 = np.zeros((200), dtype=int)
            for d in net.selected_digits:
                take_1479[20*d:20*(d+1)] = True # Take d-th digit
            take_1479 = np.asarray(take_1479, dtype=bool)

            net.miniBatchs_images[0] = np.load(data_dir+"miniBatchs_images.npy")[0, take_1479]

        
            run = 0
            net.train_plot_update(5000, isPlotting=False, isSaving=True, saving_dir=data_dir+"run_"+str(net.selected_digits)+"_n"+str(n)+"_T"+str(temp)+".npz", testFreq=100)

 

