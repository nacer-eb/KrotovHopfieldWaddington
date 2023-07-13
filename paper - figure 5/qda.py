import sys
sys.path.append('../')

import numpy as np

import matplotlib

from main_module.KrotovV2 import *
from main_module.KrotovV2_utils import *

data_dir = "data_1_10_200_179/"#"data_1_10_200_47_900_precise/"

for noise_r in [5]:
    for temp in [800]:#np.arange(400, 680, 20):

        fig, ax = plt.subplots(1, 1)

        for run in [1, 7]:

            n_range = np.arange(2, 61, 1)[:]#np.arange(44, 45, 0.1) #[50:60] @,
            data_Ms = np.zeros((len(n_range), 784))

            alphas = np.zeros((len(n_range)))
            
            for n_i, n in enumerate(n_range):
                saving_dir = data_dir+"run_"+str(run)+"_n"+str(n)+"_T"+str(temp)+".npz"
                
                data = np.load(saving_dir)
                data_M = data['M']
                
                data_Ms[n_i] = data_M
                
                data_L = data['L']
                data_T = data['miniBatchs_images'][0]
                data_T_inv = np.linalg.pinv(data_T)
                coefs = np.sum(((data_M@data_T_inv).reshape(len(data_M), 3, 20)), axis=-1)

                alphas[n_i] = coefs[0, 0]/np.sum(np.abs(coefs[0]))
                
                
            norm = matplotlib.colors.Normalize(vmin=0, vmax=9)
            tab10 = matplotlib.cm.get_cmap('tab10')
            
            
            
            ax.scatter(alphas, n_range, s=50, color=tab10(norm(run)))
                
            n_range_detailed = np.arange(np.min(n_range), np.max(n_range), 0.01)

            import scipy.interpolate as interpolate
            
            tick, u = interpolate.splprep([n_range, alphas], s=0.001)
            unew = np.arange(0, 1.01, 0.01)
            alpha_detailed = interpolate.splev(unew, tick)
            

            #p = np.polyfit(n_range, alphas, deg=25)
            #ax.plot(np.polyval(p, n_range_detailed), n_range_detailed, c=tab10(norm(run)), lw=10, alpha=0.4)
            
            #ax.plot(alpha_detailed[1], alpha_detailed[0], c=tab10(norm(run)), lw=10, alpha=0.4)
                            
                

        plt.show()
