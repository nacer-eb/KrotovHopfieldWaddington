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

n, temp = 15, 800

run = 0
saving_dir = data_dir+"run_"+str([1, 4, 7])+"_n"+str(n)+"_T"+str(temp)+".npz"
data = np.load(saving_dir)
data_M = data['M']

data_L = data['L']
print(np.shape(data_L))

tmax = len(data_M)
#[1, 4, 7, 9, 3, 5, 6, 8, 0, 2]



indices = np.zeros((20), dtype=int)
indices_class = [1, 1, 1, 1, 4, 4, 4, 4, 7, 7, 7, 7]

for i in range(0, 4*3):
    print(np.shape(np.argwhere(data_L[-1, :, indices_class[i]] == np.max(data_L[-1, :, indices_class[i]], axis=-1) )))

    all_indices = np.argwhere(data_L[-1, :, indices_class[i]] == np.max(data_L[-1, :, indices_class[i]], axis=-1) )
    
    indices[i] = all_indices[np.random.randint(0, len(all_indices))]


print(indices)


t_samples = [100, 210, 300, 400, 1000, 1500, 2900] #n=3

if n == 15:
    t_samples = [200, 340, 440, 590, 900, 1500, 2900]#n=15

if n == 30:
    t_samples = [200, 640, 1000, 1210, 1450, 2130, 2900]#n=30

print("TEST", len(t_samples))

fig, ax = plt.subplots(1, len(t_samples), figsize=(4*7, 4))
for t_i, t in enumerate(t_samples):#[0, 790, 1200, 1700, 2000, 2300, 2700, 3499]:
    im = ax[t_i].imshow(merge_data(data_M[t, indices], 4, 3), cmap="bwr", vmin=-1, vmax=1)

    props = dict(boxstyle='round', facecolor='whitesmoke', alpha=0.5)
    
    ax[t_i].set_title(r"$t=$"+str(t), bbox=props, pad=20)
    ax[t_i].axis('off')
plt.tight_layout()
plt.show()


