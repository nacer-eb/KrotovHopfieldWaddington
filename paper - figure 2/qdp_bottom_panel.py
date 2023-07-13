import sys
sys.path.append('../')

import numpy as np


import matplotlib
font = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 18}
matplotlib.rc('font', **font)


import matplotlib.pyplot as plt
import matplotlib.animation as anim

from main_module.KrotovV2_utils import *


data_dir = "data_100_10_200/"

n, temp = 30, 670

run = 0
selected_digits = [1, 4, 7]
saving_dir = data_dir+"run_"+str(selected_digits)+"_n"+str(n)+"_T"+str(temp)+".npz"
data_M = np.load(saving_dir)['M']

tmax = len(data_M)
indices = [0, 30, 8, 5, 7, 16, 11, 20, 59] #- 147
t_samples = [200, 440, 590, 830, 2000] #- 147

fig, ax = plt.subplots(1, len(t_samples))
for t_i, t in enumerate(t_samples):
    im = ax[t_i].imshow(merge_data(data_M[t, indices], 3, 3), cmap="bwr", vmin=-1, vmax=1)
    ax[t_i].set_title(r"$t=$"+str(t))
    ax[t_i].axis('off')
plt.tight_layout()
plt.subplots_adjust(wspace=0.03, hspace=0)
plt.show()


