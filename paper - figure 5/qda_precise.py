import sys
sys.path.append('../')

import numpy as np

from main_module.KrotovV2 import *
from main_module.KrotovV2_utils import *

data_dir = "data_1_10_200_179/"#"data_1_10_200_47_900_precise/"

for noise_r in [5]:
    for temp in [800]:#np.arange(400, 680, 20):

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        for run in [1, 7]:

            n_range = np.arange(2, 61, 1)[0:]#np.arange(44, 45, 0.1) #[50:60] @,
            data_Ms = np.zeros((len(n_range), 784))
            
            for n_i, n in enumerate(n_range):
                saving_dir = data_dir+"run_"+str(run)+"_n"+str(n)+"_T"+str(temp)+".npz"
                
                data = np.load(saving_dir)
                data_M = data['M']
                
                data_Ms[n_i] = data_M
                
                data_L = data['L']
                data_T = data['miniBatchs_images'][0]
                data_T_inv = np.linalg.pinv(data_T)
                coefs = np.sum(((data_M@data_T_inv).reshape(len(data_M), 3, 20)), axis=-1)
                
                
                print(np.sum(np.abs(coefs[0])))
                
                
                ax.scatter(coefs[0, 0]/np.sum(np.abs(coefs[0])), coefs[0, 1]/np.sum(np.abs(coefs[0])), n, s=50, c="k")


        plt.show()

