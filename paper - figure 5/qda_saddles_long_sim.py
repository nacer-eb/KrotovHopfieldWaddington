import sys
sys.path.append('../')

import numpy as np
import matplotlib.pyplot as plt

import matplotlib
font = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 14}

matplotlib.rc('font', **font)

from main_module.KrotovV2_utils import *


data_dir = "data_1_10_200_179_full/"

temp = 800

for n_i, n in enumerate([4, 23, 38, 50]):
    
    filename = "run_long_1_n"+str(n)+"_T"+str(temp)+".npz"
    
    data = np.load(data_dir + filename)
    data_M = data["M"]
    
    tmax = len(data_M)
    #tmax = [150, 300, 400, 500][n_i]
    
    fig, ax = plt.subplots(1, 5, figsize=(5*5, 5))
    
    for i in range(0, 5):
        #4-i
        t = (i)*(tmax//5)
        ax[i].imshow(merge_data(data_M[t, :25 , :], 5, 5), cmap="bwr", vmin=-1, vmax=1)
        
        props = dict(boxstyle='round', facecolor='whitesmoke', alpha=0.5)
        
        ax[i].set_title(r"$t=$"+str(t), bbox=props, pad=20)
        ax[i].axis('off')
    plt.show()

