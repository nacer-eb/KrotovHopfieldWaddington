import sys
sys.path.append('../')

import numpy as np

from main_module.KrotovV2 import *
from main_module.KrotovV2_utils import *


import matplotlib
font = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 44}
matplotlib.rc('font', **font)

import matplotlib.animation as anim

data_dir = "data/"

isFirstRun = False

selected_digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]#
prefix = str(selected_digits)+"_multi_temp_tw/" # I used main,and momentum #"main"#

tmax = 10000
N_mem = 100

n_range = [25]
Nn = len(n_range)

temp_range = np.arange(650, 900, 3)
Nt = len(temp_range)

data_Ms = np.zeros((Nt, Nn, N_mem, 784))
data_Ls = np.zeros((Nt, Nn, N_mem, 10))

data_T = np.load(data_dir + "miniBatchs_images.npy")[0]
data_T_inv = np.linalg.pinv(data_T)



if isFirstRun:
    for h, temp in enumerate(temp_range):
        for i, n in enumerate(n_range):
            print(n, temp)
        
            saving_dir = data_dir+prefix+"trained_net_end_n"+str(n)+"_T"+str(temp)+".npz"
            
            data = np.load(saving_dir)
            data_Ms[h, i] = data['M'][-1]
            data_Ls[h, i] = data['L'][-1]
            
    data_coefs = (data_Ms@data_T_inv).reshape(Nt, Nn, N_mem, 10, 20)
    
    np.save(data_dir + prefix + "data_Ms_T.npy", data_Ms)
    np.save(data_dir + prefix + "data_Ls_T.npy", data_Ls)
    np.save(data_dir + prefix + "data_Coefs_T.npy", data_coefs)

data_Ms = np.load(data_dir + prefix + "data_Ms_T.npy")
data_Ls = np.load(data_dir + prefix + "data_Ls_T.npy")
data_coefs = np.load(data_dir + prefix + "data_Coefs_T.npy")

print(np.shape(data_Ls))

n_i = 0

print(np.shape(data_Ls))

cmap_tab10 = matplotlib.cm.tab10
norm = matplotlib.colors.Normalize(vmin=0, vmax=10)


label_population = np.zeros((Nt, 10))
for d in range(10):
    label_population[:, d] = np.sum(data_Ls[:, n_i, :, d] > 0.98, axis=-1)
    
fig = plt.figure(figsize=(30, 30+12+2))

axs = fig.subplot_mosaic("""
000000111111222222333333444444.
000000111111222222333333444444.
000000111111222222333333444444.
000000111111222222333333444444.
000000111111222222333333444444.
000000111111222222333333444444.
000000111111222222333333444444.
000000111111222222333333444444.
000000111111222222333333444444.
000000111111222222333333444444.
000000111111222222333333444444.
000000111111222222333333444444.
000000111111222222333333444444.
000000111111222222333333444444.
000000111111222222333333444444.
000000111111222222333333444444.
000000111111222222333333444444.
000000111111222222333333444444.
000000111111222222333333444444.
000000111111222222333333444444.
000000111111222222333333444444.
000000111111222222333333444444.
000000111111222222333333444444.
000000111111222222333333444444.
000000111111222222333333444444.
000000111111222222333333444444.
000000111111222222333333444444.
000000111111222222333333444444.
000000111111222222333333444444.
000000111111222222333333444444.
...............................
...............................
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAX
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAX
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAX
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAX
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAX
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAX
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAX
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAX
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAX
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAX
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAX
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAX

""")





props = dict(boxstyle='round', facecolor='whitesmoke', alpha=0.5)
t_samples = np.asarray([0, 25, 40, 46, 55])

for d in range(5):
    t_sample = t_samples[d]
    indices = np.zeros(20)
    for i in range(len(indices)):
        strictness = 1
        all_indices = np.argwhere(data_Ls[t_sample, 0, :, i//2] >= strictness )
        while len(all_indices) == 0:
            strictness -= 0.05
            all_indices = np.argwhere(data_Ls[t_sample, 0, :, i//2] >= strictness ) #np.argmax(data_Ls[t_sample, 0, :], axis=-1) == i//2 #

            if strictness < 0.5:
                all_indices = np.sum(data_Ls[t_sample, 0, :], axis=-1) <= -8.8 #np.argmax(data_Ls[t_sample, 0, :], axis=-1) == i//2 #
                
            
        indices[i] =  all_indices[np.random.randint(len(all_indices))] # -> Pick randomly when Label is mostly # digit class i//2 $
    indices = np.asarray(indices, dtype=int)
    
    
    axs[str(d)].imshow(merge_data(data_Ms[t_sample, 0, indices], 2, 10), cmap="bwr", vmin=-1, vmax=1)
    axs[str(d)].set_title(r"$T=$"+str(temp_range[t_sample]), pad=35, fontsize=55, bbox=props) # Time stamps / Cosmetics
    axs[str(d)].set_xticks([]); axs[str(d)].set_yticks([])



#for d in range(10):
    #axs['A'].plot(label_population[:, d], color=cmap_tab10(norm(d)), lw=3, alpha=0.5, marker=".")

axs['A'].stackplot(temp_range, label_population.T, colors=cmap_tab10(norm(np.arange(0, 10, 1))))


cb = matplotlib.colorbar.ColorbarBase(axs['X'], cmap=cmap_tab10, norm=norm, orientation='vertical')
cb.set_ticks(np.arange(0, 10, 1) + 0.5) # Finally found how to center these things 
cb.set_ticklabels(np.arange(0, 10, 1))
cb.set_label("Digit class")

axs['A'].set_xlim(min(temp_range), max(temp_range))
axs['A'].set_xlabel("Temperature")
axs['A'].set_ylabel("Memory population (by label)")

plt.savefig("MemoryPopTemperatureExperiment.png")#, transparent="True")




exit()

