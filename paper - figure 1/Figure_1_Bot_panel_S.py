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
data = np.load(saving_dir)
data_M = data['M']

data_L = data['L']
print(np.shape(data_L))

tmax = len(data_M)
#[1, 4, 7, 9, 3, 5, 6, 8, 0, 2]



indices = np.zeros((20), dtype=int)
indices_class = [1, 1, 4, 4, 7, 7, 9, 9, 3, 3, 5, 5, 6, 6, 8, 8, 0, 0, 2, 2]

for i in range(0, 20):
    print(np.shape(np.argwhere(data_L[-1, :, indices_class[i]] == np.max(data_L[-1, :, indices_class[i]], axis=-1) )))
    arglist = np.argwhere(data_L[-1, :, indices_class[i]] == np.max(data_L[-1, :, indices_class[i]], axis=-1) )
    indices[i] = arglist[i%len(arglist)]


print(indices)

#t_samples = [200, 270, 370, 510, 620, 900, 3440] # n=3
#t_samples = [200, 300, 550, 700, 1500, 2000, 3440] #n=15
#t_samples = [200, 620, 1000, 1320, 1910, 2700, 3440] #n=25
t_samples = [200, 620, 1100, 1610, 1870, 2330, 3440] #n=40

fig, ax = plt.subplots(1, len(t_samples), figsize=(28, 4))
for t_i, t in enumerate(t_samples):#[0, 790, 1200, 1700, 2000, 2300, 2700, 3499]:
    im = ax[t_i].imshow(merge_data(data_M[t, indices], 4, 5), cmap="bwr", vmin=-1, vmax=1)
    ax[t_i].set_title(r"$t=$"+str(t))
    ax[t_i].axis('off')
plt.tight_layout()
plt.show()


