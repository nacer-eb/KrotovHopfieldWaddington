import sys
sys.path.append('../')

import numpy as np

import matplotlib
font = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 22}
matplotlib.rc('font', **font)

import matplotlib.pyplot as plt
import matplotlib.animation as anim

from main_module.KrotovV2_utils import *

data_dir = "data_100_10_200/"

n, temp = 30, 670

run = 0
saving_dir = data_dir+"momentum_run_"+str(run)+"_n"+str(n)+"_T"+str(temp)+".npz"
data_M = np.load(saving_dir)['M']

tmax = len(data_M)
#[1, 4, 7, 9, 3, 5, 6, 8, 0 ,2]
indices = [0, 29, 71, 81, 72, 73, 70, 33, 42, 31, 13, 54, 6, 83, 20, 16, 10, 2, 51, 95] #[0, 1, 40, 82, 3, 9, 20, 19]
#t_samples = [200, 800, 1670, 2300, 3400]
t_samples = [200, 900, 1250, 1530, 1800, 2800, 3440]

fig, ax = plt.subplots(1, len(t_samples), figsize=(28, 4))
for t_i, t in enumerate(t_samples):#[0, 790, 1200, 1700, 2000, 2300, 2700, 3499]:
    im = ax[t_i].imshow(merge_data(data_M[t, indices], 4, 5), cmap="bwr", vmin=-1, vmax=1)
    ax[t_i].set_title(r"$t=$"+str(t))
    ax[t_i].axis('off')
plt.tight_layout()
plt.show()


