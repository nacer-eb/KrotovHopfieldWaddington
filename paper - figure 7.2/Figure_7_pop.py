import sys
sys.path.append('../')

import numpy as np
import sys

from main_module.KrotovV2 import *

import matplotlib.pyplot as plt

import matplotlib
font = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 42}
matplotlib.rc('font', **font)


data_dir = "data/"
selected_digits = [1, 7]#
prefix = str(selected_digits)+"/" # _coarse_stable

N_runs = 2

temp_range = np.arange(700, 900, 20)[::1] #temp_range = np.arange(500, 900, 20)
n_range = np.arange(2, 61, 1)[::1] #n_range = np.arange(2, 32, 2)

#temp_range = np.arange(600, 900, 20)[6::2] #temp_range = np.arange(500, 900, 20)
#n_range = np.arange(2, 61, 1)[::2] #n_range = np.arange(2, 32, 2)

N_mem = 100

data_Ms = np.zeros((N_runs, len(temp_range), len(n_range), 2, N_mem, 784))
data_M_saddles = np.zeros((len(temp_range), len(n_range), 2, 784))
data_Ls = np.zeros((N_runs, len(temp_range), len(n_range), 2, N_mem, 10))

isFirstRun = True
if isFirstRun:

    for i, temp in enumerate(temp_range):
        for j, n in enumerate(n_range):
            for k in range(2):
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

data_Ms_pop_run = np.zeros((N_runs, len(temp_range), len(n_range), 2, 2)) # Population proportion
data_Ms_pop = np.zeros((len(temp_range), len(n_range), 2, 2)) # Population proportion

for r in range(N_runs):
    for i, temp in enumerate(temp_range):
        for j, n in enumerate(n_range):
            for k in range(2):
                for l in range(2):
                    data_Ms_pop_run[r, i, j, k, l] = np.sum(np.argmax(data_Ls[r, i, j, k], axis=-1) == selected_digits[l], axis=-1) # not strict
                    #data_Ms_pop_run[r, i, j, k, l] = np.sum( (data_Ls[r, i, j, k, :, selected_digits[l]] >= -0.95), axis=-1 ) # stricter


# Standard mean



import scipy.signal as sig

data_Ms_pop = np.mean(data_Ms_pop_run, axis=0)








fig = plt.figure(figsize=(7+9*2, 144-105+1))
axs = fig.subplot_mosaic("""
AAAAAAAA!.......aaaaaaaa@
AAAAAAAA!.......aaaaaaaa@
AAAAAAAA!.......aaaaaaaa@
AAAAAAAA!.......aaaaaaaa@
AAAAAAAA!.......aaaaaaaa@
AAAAAAAA!.......aaaaaaaa@
AAAAAAAA!.......aaaaaaaa@
AAAAAAAA!.......aaaaaaaa@
.........................
.........................
BBBBBBBB#.......bbbbbbbb$
BBBBBBBB#.......bbbbbbbb$
BBBBBBBB#.......bbbbbbbb$
BBBBBBBB#.......bbbbbbbb$
BBBBBBBB#.......bbbbbbbb$
BBBBBBBB#.......bbbbbbbb$
BBBBBBBB#.......bbbbbbbb$
BBBBBBBB#.......bbbbbbbb$
.........................
.........................
.........................
.........................
11111111%.......22222222?
11111111%.......22222222?
11111111%.......22222222?
11111111%.......22222222?
11111111%.......22222222?
11111111%.......22222222?
11111111%.......22222222?
11111111%.......22222222?
.........................
.........................
33333333&.......44444444*
33333333&.......44444444*
33333333&.......44444444*
33333333&.......44444444*
33333333&.......44444444*
33333333&.......44444444*
33333333&.......44444444*
33333333&.......44444444*
""")

axs_pop = np.asarray( [[ axs['A'], axs['B'] ], [ axs['1'], axs['3'] ]]  )
axs_saddle = np.asarray( [[ axs['a'], axs['b'] ], [ axs['2'], axs['4'] ]] )

axs_pop_cb = np.asarray( [[ axs['!'], axs['#'] ], [ axs['%'], axs['&'] ]]  )
axs_saddle_cb = np.asarray( [[ axs['@'], axs['$'] ], [ axs['?'], axs['*'] ]]  )


tab10_cmap = matplotlib.colormaps["tab10"]
tab10_norm = matplotlib.colors.Normalize(0, 10)

def get_custom_cmap(digit_class):
    custom_cmap = np.zeros((256, 4))

    colors = np.asarray([np.asarray([1.0, 1.0, 1.0, 1.0]),  np.asarray(tab10_cmap(tab10_norm(digit_class))), np.asarray([0.0, 0.0, 0.0, 1.0])])
    
    x = np.linspace(0, 1, 128)
    for i in range(128):
        custom_cmap[i] = x[i]*colors[1] + (1.0-x[i])*colors[0]
        custom_cmap[i+128] = x[i]*colors[2] + (1.0-x[i])*colors[1]
    custom_cmap = matplotlib.colors.ListedColormap(custom_cmap)
    
    return custom_cmap

digit_classes = selected_digits # REDUNDANT, IMPROVE THIS...

for d_ic in range(2):
    for d_probe in range(2):
        cmap_digit = get_custom_cmap(digit_classes[d_probe])
    
        data_pop = data_Ms_pop[:, :, d_ic, d_probe]/N_mem
        norm_pop = matplotlib.colors.Normalize(vmin=0.0, vmax=1.0)
        #norm_pop = matplotlib.colors.Normalize(vmin=np.min(data_pop[-1])-0.05, vmax=np.max(data_pop[-1])+0.07)


    
        axs_pop[d_ic, d_probe].imshow(data_pop, cmap=cmap_digit, norm=norm_pop,
                             extent=[min(n_range), np.max(n_range), max(temp_range), np.min(temp_range)],
                             aspect=(max(n_range)-min(n_range))/(max(temp_range) - min(temp_range)))


        data_saddles = data_M_saddles_coefs[:, :, d_ic, d_probe]
        norm_saddles = matplotlib.colors.Normalize(vmin=-0.2, vmax=1.1)
        #norm_saddles = matplotlib.colors.Normalize(vmin=np.min(data_saddles)-0.051, vmax=np.max(data_saddles)+0.05)



        axs_saddle[d_ic, d_probe].imshow(data_saddles, cmap=cmap_digit, norm=norm_saddles,
                                extent=[min(n_range), np.max(n_range), max(temp_range), np.min(temp_range)],
                                aspect=(max(n_range)-min(n_range))/(max(temp_range) - min(temp_range)))

    
        cb = matplotlib.colorbar.ColorbarBase(axs_pop_cb[d_ic, d_probe], cmap=cmap_digit, norm=norm_pop, orientation='vertical')
        cb.set_label("Proportion of "+str(digit_classes[d_probe]))
        

        cb = matplotlib.colorbar.ColorbarBase(axs_saddle_cb[d_ic, d_probe], cmap=cmap_digit, norm=norm_saddles, orientation='vertical')
        cb.set_label(r"$\alpha_"+str(digit_classes[d_probe])+r"$ of the saddle")





plt.subplots_adjust(wspace=0.075)
plt.savefig("Figure_pop_new.png")
exit()
