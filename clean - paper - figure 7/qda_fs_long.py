import sys
sys.path.append('../')

import numpy as np
import matplotlib

from main_module.KrotovV2 import *

font = {'family' : 'Times new roman',
        'weight' : 'normal',
        'size'   : 14}
matplotlib.rc('font', **font)

temp_range = np.arange(500, 800, 20)
n_range = np.arange(2, 61, 2) # messed up here
n_range = np.concatenate((np.arange(2, 31, 2), np.arange(31, 61, 2))) # This is the range I used

data_dir = "data_17_mean_2nd/"

saving_dir= data_dir+"run_1_n"+str(n_range[0])+"_T"+str(temp_range[0])+".npz"
data = np.load(saving_dir)
data_T = data['miniBatchs_images'][0]
data_T_inv = np.linalg.pinv(data_T)

data_M = data['M'][-1]

first_run = False

if first_run:
    
    data_Ms = np.zeros((len(temp_range), len(n_range), len(data_M), 784))
    data_coefs = np.zeros((len(temp_range), len(n_range), len(data_M), len(data_T)))
    
    for k, run in enumerate([1, 7]):
        for  i, temp in enumerate(temp_range):  ## IMPORTANT PART OF THE RANGE IS EXCLUDED!!!
            for j, n in enumerate(n_range):
                
                saving_dir=data_dir+"run_"+str(run)+"_n"+str(n)+"_T"+str(temp)+".npz"
            
                data = np.load(saving_dir)
            
                data_Ms[i, j] = data['M'][-1]          
                data_coefs[i, j] = data_Ms[i, j] @ data_T_inv

                print(i, j, k)
            
        np.save(data_dir+"data_Ms_"+str(run)+".npy", data_Ms)
        np.save(data_dir+"data_coefs_"+str(run)+".npy", data_coefs)


if not first_run:
    run = 7

    data_Ms = np.load(data_dir+"data_Ms_"+str(run)+".npy")
    data_coefs = np.load(data_dir+"data_coefs_"+str(run)+".npy")

    actual_digit = [1, 7]
        
    fig = plt.figure()
    axs = fig.subplot_mosaic(
        """
        AAAABBBBCCCC
        AAAABBBBCCCC
        AAAABBBBCCCC
        AAAABBBBCCCC
        DDDDEEEEFFFF
        DDDDEEEEFFFF
        DDDDEEEEFFFF
        DDDDEEEEFFFF
        ............
        ............
        GGGGGGHHHHHH
        GGGGGGHHHHHH
        GGGGGGHHHHHH
        GGGGGGHHHHHH
        GGGGGGHHHHHH
        GGGGGGHHHHHH
        """
    )

    ax_mem = np.asarray([axs['A'], axs['D']])
    ax_sec_mem = np.asarray([axs['B'], axs['E']])
    ax_phase_ortho = np.asarray([axs['C'], axs['F']])
    ax_pop_phase = np.asarray([axs['G'], axs['H']])

    ax_mem[0].set_title("Sample memories")
    ax_mem[0].set_xticks([]); ax_mem[1].set_xlabel("$n$")
    for ax in ax_mem:
        ax.set_ylabel("Temperature")

    ax_sec_mem[0].set_title("Background digit")
    ax_sec_mem[0].set_xticks([]); ax_sec_mem[1].set_xlabel("$n$")
    for ax in ax_sec_mem:
        ax.set_yticks([])

    ax_phase_ortho[0].set_title("Background digit proportion")
    ax_phase_ortho[0].set_xticks([]); ax_phase_ortho[1].set_xlabel("$n$")
    for ax in ax_phase_ortho:
        ax.set_yticks([])

    ax_pop_phase[1].set_yticks([]); ax_pop_phase[0].set_ylabel("Temperature")
    for ax in ax_pop_phase:
        ax.set_xlabel("$n$");


    ax_pop_phase[0].set_title("Proportion of 1s")
    ax_pop_phase[1].set_title("Predicted proportion of 1s")
        
    extent = [min(n_range), max(n_range), max(temp_range), min(temp_range)]
    aspect = (min(n_range) - max(n_range))/(min(temp_range) - max(temp_range))


    for digit_index in range(0, 2): # 0 or can be 1
        #digit_picker = -(1-digit_index)
        digit_picker = data_coefs[:, :, :, 0] > data_coefs[:, :, :, 1]

        for i in range(len(temp_range)):
            for j in range(len(n_range)):
                arg_max = np.argmax(digit_picker[i, j], axis=-1)
                digit_picker[i, j, :] = 0
                digit_picker[i, j, arg_max] = 1
        
    
        data_coefs_secondary = np.copy(data_coefs)
        data_coefs_secondary[:, :, :, digit_index] = 0
        secondary_Ms = data_coefs_secondary@data_T

        ax_mem[digit_index].imshow(merge_data(data_Ms[digit_picker, :].reshape(len(temp_range)*len(n_range), 784), len(n_range), len(temp_range)), cmap="bwr", vmin=-1, vmax=1, extent=extent, aspect=aspect)
        ax_sec_mem[digit_index].imshow(merge_data(secondary_Ms[digit_picker, :].reshape(len(temp_range)*len(n_range), 784), len(n_range), len(temp_range)), cmap="bwr", vmin=-1, vmax=1, extent=extent, aspect=aspect)
        
        ax_phase_ortho[digit_index].imshow(data_coefs[digit_picker, (1-digit_index)], cmap="bwr", vmin=-1, vmax=1, extent=extent, aspect=aspect) # It is a coincidence (ish) that digit picker works for both indices

        
    pop_1s = np.sum(data_coefs[:, :, :, 0] > data_coefs[:, :, :, 1], axis=-1)
    ax_pop_phase[0].imshow(pop_1s/100, cmap="bwr", vmin=0, vmax=1, extent=extent, aspect=aspect)

    
    A = data_Ms[:, :, -1, :]; B = data_Ms[:, :, 0, :]
    d_AA = A@data_T[0]; d_AB = A@data_T[1]
    d_BA = B@data_T[0]; d_BB = B@data_T[1]

    pop_predict = np.zeros((len(temp_range), len(n_range)))
    for j, n in enumerate(n_range):
        pop_A_tmp = d_AA[:, j]**n + d_AB[:, j]**n
        pop_B_tmp = d_BA[:, j]**n + d_BB[:, j]**n
        pop_predict[:, j] = pop_B_tmp/(pop_A_tmp + pop_B_tmp)

    
    ax_pop_phase[1].imshow(pop_predict, cmap="bwr", vmin=0, vmax=1, extent=extent, aspect=aspect)
        
        
    plt.subplots_adjust(top=0.95, bottom=0.085, left=0.3, right=0.7, hspace=0.2, wspace=0.02)
    plt.show()
    exit()
    
