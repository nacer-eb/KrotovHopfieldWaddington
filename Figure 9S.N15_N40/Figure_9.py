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


selected_digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]#
prefix = str(selected_digits)+"/" # I used main,and momentum #"main"#

tmax = 10000
N_mem = 100

n = 40 # or 40

temp_range = np.arange(650, 1000, 10)
temp_range_rescaled = temp_range/784.0
Nt = len(temp_range)

data_Ms = np.zeros((Nt, N_mem, 784))
data_Ls = np.zeros((Nt, N_mem, 10))

dataset = "../defaults/miniBatchs_images.npy"
data_T = np.load(dataset)[0]
data_T_inv = np.linalg.pinv(data_T)


isFirstRun = True
if isFirstRun:
    for h, temp in enumerate(temp_range):
        print(n, temp)
    
        saving_dir = data_dir+prefix+"trained_net_end_n"+str(n)+"_T"+str(temp)+".npz"
        
        data = np.load(saving_dir)
        data_Ms[h] = data['M'][-1]
        data_Ls[h] = data['L'][-1]
        
    data_coefs = (data_Ms@data_T_inv).reshape(Nt, N_mem, 10, 20)
    
    np.save(data_dir + prefix + "data_Ms_T.npy", data_Ms)
    np.save(data_dir + prefix + "data_Ls_T.npy", data_Ls)
    np.save(data_dir + prefix + "data_Coefs_T.npy", data_coefs)

data_Ms = np.load(data_dir + prefix + "data_Ms_T.npy")
data_Ls = np.load(data_dir + prefix + "data_Ls_T.npy")
data_coefs = np.load(data_dir + prefix + "data_Coefs_T.npy")


cmap_tab10 = matplotlib.cm.tab10
norm = matplotlib.colors.Normalize(vmin=0, vmax=10)

label_population = np.zeros((Nt, 10))
for d in range(10):
    label_population[:, d] = np.sum(data_Ls[:, :, d] > 0.98, axis=-1)
    
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


# handpicked
if n==15:
    t_samples = np.asarray([10, 17, 20, 25, 34], dtype=int)
if n==40:
    t_samples = np.asarray([2, 12, 15, 20, 34], dtype=int)

print(temp_range_rescaled[t_samples])
    
for d in range(5):
    t_sample = t_samples[d]
    indices = np.zeros(20)
    for i in range(len(indices)):
        strictness = 1
        all_indices = np.argwhere(data_Ls[t_sample, :, i//2] >= strictness )
        while len(all_indices) == 0:
            strictness -= 0.05
            all_indices = np.argwhere(data_Ls[t_sample, :, i//2] >= strictness ) #np.argmax(data_Ls[t_sample, 0, :], axis=-1) == i//2 #

            if strictness < 0.5:
                all_indices = np.sum(data_Ls[t_sample, :], axis=-1) <= -8.8 #np.argmax(data_Ls[t_sample, 0, :], axis=-1) == i//2 #
                
            
        indices[i] =  all_indices[np.random.randint(len(all_indices))] # -> Pick randomly when Label is mostly # digit class i//2 $
    indices = np.asarray(indices, dtype=int)
    
    
    axs[str(d)].imshow(merge_data(data_Ms[t_sample, indices], 2, 10), cmap="bwr", vmin=-1, vmax=1)
    axs[str(d)].set_title(r"$T_r = $"+'{0:.2f}'.format(temp_range_rescaled[t_sample]), pad=55, fontsize=55, bbox=props) # Time stamps / Cosmetics
    axs[str(d)].set_xticks([]); axs[str(d)].set_yticks([])



axs['A'].stackplot(temp_range_rescaled, label_population.T, colors=cmap_tab10(norm(np.arange(0, 10, 1))))


cb = matplotlib.colorbar.ColorbarBase(axs['X'], cmap=cmap_tab10, norm=norm, orientation='vertical')
cb.set_ticks(np.arange(0, 10, 1) + 0.5) # Finally found how to center these things 
cb.set_ticklabels(np.arange(0, 10, 1))
cb.set_label("Digit class")

axs['A'].set_xlim(min(temp_range_rescaled), max(temp_range_rescaled))
axs['A'].set_xlabel("Rescaled Temperature")
axs['A'].set_ylabel("Memory population (by label)")

plt.savefig("Figure_9_N"+str(n)+"_tmp.png")#, transparent="True")

