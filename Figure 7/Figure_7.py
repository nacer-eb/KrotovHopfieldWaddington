import sys
sys.path.append('../')

import numpy as np
import sys

from main_module.KrotovV2 import *

import matplotlib.pyplot as plt

import matplotlib
font = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 54}
matplotlib.rc('font', **font)


data_dir = "data/"
selected_digits = [1, 7, 8]#
prefix = str(selected_digits)+"/" # _coarse_stable

N_runs = 10

temp_range = [550, 750, 800]
n_range = np.arange(2, 60, 1)


N_mem = 100

data_Ms = np.zeros((N_runs, len(temp_range), len(n_range), 3, N_mem, 784))
data_M_saddles = np.zeros((len(temp_range), len(n_range), 3, 784))
data_Ls = np.zeros((N_runs, len(temp_range), len(n_range), 3, N_mem, 10))

isFirstRun = False
if isFirstRun:

    for i, temp in enumerate(temp_range):
        for j, n in enumerate(n_range):
            for k in range(3):
                for r in range(N_runs):
                    
                    run_prefix = "end_states_" + str(r) + "/"
                    saving_dir=data_dir+prefix+run_prefix+"trained_net_end_n"+str(n)+"_T"+str(temp)+"ic"+str(selected_digits[k])+".npz"
                    
                    if os.path.isfile(saving_dir):
                        data_Ms[r, i, j, k] = np.load(saving_dir)['M']
                        data_Ls[r, i, j, k] = np.load(saving_dir)['L']
                    else:
                        print("WARNING: File not found, ", saving_dir)

                saving_dir=data_dir+prefix+"saddles/net_saddle_n"+str(n)+"_T"+str(temp)+"ic"+str(selected_digits[k])+".npz"
                
                if os.path.isfile(saving_dir):
                    data_M_saddles[i, j, k] = np.load(saving_dir)['M'][0]
                    
                else:
                    print("WARNING: File not found, ", saving_dir)
                    
        print(temp)

    np.save(data_dir+prefix+"data_Ms.npy", data_Ms)
    np.save(data_dir+prefix+"data_M_saddles.npy", data_M_saddles)
    np.save(data_dir+prefix+"data_Ls.npy", data_Ls)


# Then
data_Ms = np.load(data_dir+prefix+"data_Ms.npy")
data_M_saddles = np.load(data_dir+prefix+"data_M_saddles.npy")
data_Ls = np.load(data_dir+prefix+"data_Ls.npy")

data_T = np.load(data_dir+prefix+"miniBatchs_images.npy")[0]
data_T_inv = np.linalg.pinv(data_T)


data_M_saddles_coefs = data_M_saddles@data_T_inv


data_M_saddles_coefs = (data_M_saddles_coefs.reshape(len(temp_range), len(n_range), 3, 10, 20)).sum(axis=-1)
data_M_saddles_coefs = data_M_saddles_coefs[:, :, :, selected_digits]


data_Ms_pop_run = np.zeros((N_runs, len(temp_range), len(n_range), 3, 3)) # Population proportion
data_Ms_pop = np.zeros((len(temp_range), len(n_range), 3, 3)) # Population proportion

for r in range(N_runs):
    for i, temp in enumerate(temp_range):
        for j, n in enumerate(n_range):
            for k in range(3):
                for l in range(3):
                    #data_Ms_pop_run[r, i, j, k, l] = np.sum(np.argmax(data_Ls[r, i, j, k], axis=-1) == selected_digits[l], axis=-1) # not strict
                    data_Ms_pop_run[r, i, j, k, l] = np.sum( (data_Ls[r, i, j, k, :, selected_digits[l]] >= 0.9), axis=-1 ) # stricter


# Standard mean
data_Ms_pop = np.mean(data_Ms_pop_run, axis=0)


tab10_cmap = matplotlib.colormaps["tab10"]
tab10_norm = matplotlib.colors.Normalize(0, 10)

fig = plt.figure(figsize=(3+3+3+1+9+1+9+1, 139-107+1))

axs = fig.subplot_mosaic("""

aaa.bbb...111111111.222222222!
aaa.bbb...111111111.222222222!
aaa.bbb...111111111.222222222!
ccc.ddd...111111111.222222222!
ccc.ddd...111111111.222222222!
ccc.ddd...111111111.222222222!
eee.fff...111111111.222222222!
eee.fff...111111111.222222222!
eee.fff...111111111.222222222!
..............................
..............................
..............................
..ggg.....333333333.444444444@
..ggg.....333333333.444444444@
..ggg.....333333333.444444444@
..hhh.....333333333.444444444@
..hhh.....333333333.444444444@
..hhh.....333333333.444444444@
iii.jjj...333333333.444444444@
iii.jjj...333333333.444444444@
iii.jjj...333333333.444444444@
..............................
..............................
..............................
kkk.lll...555555555.666666666#
kkk.lll...555555555.666666666#
kkk.lll...555555555.666666666#
..mmm.....555555555.666666666#
..mmm.....555555555.666666666#
..mmm.....555555555.666666666#
nnn.ooo...555555555.666666666#
nnn.ooo...555555555.666666666#
nnn.ooo...555555555.666666666#

""")

axs_saddles_samples = np.asarray([axs['a'], axs['b'], axs['c'], axs['d'], axs['e'], axs['f'],
                                axs['g'], axs['h'], axs['i'], axs['j'],
                                axs['k'], axs['l'], axs['m'], axs['n'], axs['o']])


n_top = 58
n_mid = 30
n_bot = 5

for ax in axs_saddles_samples:
    ax.set_xticks([]); ax.set_yticks([])

sample_temp = np.asarray([0, 0, 0, 0, 0, 0,
                          1, 1, 1, 1,
                          2, 2, 2, 2, 2])

sample_n = np.asarray([n_top, n_top, n_mid, n_mid, n_bot, n_bot,
                       n_top, n_mid, n_bot, n_bot,
                       n_top, n_top, n_mid, n_bot, n_bot])

sample_ic = np.asarray([1, 0, 1, 0, 1, 0,
                        0, 0, 1, 0,
                        1, 0, 0, 1, 0])

props = dict(boxstyle='round', facecolor='whitesmoke', alpha=0.5)

for i in range(len(axs_saddles_samples)):
    axs_saddles_samples[i].imshow((data_M_saddles[sample_temp[i], sample_n[i]-2, sample_ic[i]]).reshape(28, 28), cmap="bwr", vmin=-1, vmax=1)

    if i in [0, 2, 4, 6, 7, 8, 10, 12, 13]:
        print(i)
        axs_saddles_samples[i].set_ylabel(r"$n=$"+str(sample_n[i]), fontsize=41, labelpad=40, verticalalignment='center', ha='center', bbox=props)


axs_saddles_alphas = np.asarray([axs['1'], axs['3'], axs['5']])
for t_i, temp in enumerate(temp_range):
    for ic in [0, 1]:
        axs_saddles_alphas[t_i].scatter(data_M_saddles_coefs[t_i, :, ic, 0], n_range, color=tab10_cmap(tab10_norm(selected_digits[ic])), s=20*2)
    axs_saddles_alphas[t_i].set_xlabel(r"$\alpha_1$")
    axs_saddles_alphas[t_i].set_ylabel(r"$n$-power")
axs_saddles_alphas[0].set_title("Saddles", pad=40)

axs_pop_proportion = np.asarray([axs['2'], axs['4'], axs['6']])
for t_i, temp in enumerate(temp_range):
    for ic in [0, 1]:
        pop_1s = data_Ms_pop[t_i, :, ic, 0]
        pop_7s = data_Ms_pop[t_i, :, ic, 1]

        pop_norm = pop_1s + pop_7s
        
        axs_pop_proportion[t_i].scatter(pop_1s/pop_norm, n_range, color=tab10_cmap(tab10_norm(selected_digits[ic])), s=20*2)
    axs_pop_proportion[t_i].set_yticks([])
    axs_pop_proportion[t_i].set_xlabel(r"Proportion of $1$s")

    axs_pop_proportion[t_i].set_ylabel(r"$T_r$ = "+'{0:.2f}'.format(temp/784.0), labelpad=1300, bbox=props) #Hacky trick

axs_pop_proportion[0].set_title("Population proportion", pad=40)


# Set red and green cmap
rg_cmap = matplotlib.colors.ListedColormap([tab10_cmap(tab10_norm(1)),tab10_cmap(tab10_norm(7))])
rg_norm = matplotlib.colors.BoundaryNorm([0, 0.5, 1], rg_cmap.N)
    
axs_cb = np.asarray([axs['!'], axs['@'], axs['#']])
for ax_cb in axs_cb:
    cb = matplotlib.colorbar.ColorbarBase(ax_cb, cmap=rg_cmap, norm=rg_norm, orientation='vertical')
    cb.set_ticks([0.25, 0.75]) # Finally found how to center these things 
    cb.set_ticklabels(["Near 1", "Near 7"], rotation=90, va='center')
    cb.set_label("Initial conditions", labelpad=20) 
    
plt.savefig("Figure_7_tmp.png")
