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
selected_digits = [4, 9]#
prefix = str(selected_digits)+"_2/" # I used main,and momentum #"main"#

temp_range = np.arange(500, 900, 20)
n_range = np.arange(2, 32, 2)

data_Ms = np.zeros((len(temp_range), len(n_range), 2, 100, 784))
data_Ls = np.zeros((len(temp_range), len(n_range), 2, 100, 10))

data_M_saddles = np.zeros((len(temp_range), len(n_range), 2, 784))

isFirstRun = True
if isFirstRun:
    for i, temp in enumerate(temp_range):
        for j, n in enumerate(n_range):
            for k in range(2):

                # Final states
                saving_dir=data_dir+prefix+"trained_net_end_n"+str(n)+"_T"+str(temp)+"ic"+str(selected_digits[k])+".npz"
                data_Ms[i, j, k] = np.load(saving_dir)['M']
                data_Ls[i, j, k] = np.load(saving_dir)['L']

                # Final states
                saving_dir=data_dir+prefix+"net_saddle_n"+str(n)+"_T"+str(temp)+"ic"+str(selected_digits[k])+".npz"
                data_M_saddles[i, j, k] = np.load(saving_dir)['M']
                
        print(temp)

    np.save(data_dir+prefix+"data_Ms.npy", data_Ms)
    np.save(data_dir+prefix+"data_M_saddles.npy", data_M_saddles)
    np.save(data_dir+prefix+"data_Ls.npy", data_Ls)



# Then

data_Ms = np.load(data_dir+prefix+"data_Ms.npy")
data_M_saddles = np.load(data_dir+prefix+"data_M_saddles.npy")
data_Ls = np.load(data_dir+prefix+"data_Ls.npy")


data_T = np.load(data_dir+"miniBatchs_images.npy")[0]
data_T_inv = np.linalg.pinv(data_T)

data_M_saddles_coefs = data_M_saddles@data_T_inv

data_Ms_pop = np.zeros((len(temp_range), len(n_range), 2, 2)) # Population proportion

for i, temp in enumerate(temp_range):
        for j, n in enumerate(n_range):
            for k in range(2):
                for l in range(2):
                    data_Ms_pop[i, j, k, l] = np.sum(np.argmax(data_Ls[i, j, k], axis=-1) == selected_digits[l], axis=-1)
                    




fig = plt.figure(figsize=(7+9*2, 110-79+1))
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
CCCCCCCC%.......cccccccc?
CCCCCCCC%.......cccccccc?
CCCCCCCC%.......cccccccc?
CCCCCCCC%.......cccccccc?
CCCCCCCC%.......cccccccc?
CCCCCCCC%.......cccccccc?
CCCCCCCC%.......cccccccc?
CCCCCCCC%.......cccccccc?
""")

axs_pop = np.asarray( [ axs['A'], axs['B'] ]  )
axs_saddle = np.asarray( [ axs['a'], axs['b'] ] )

axs_bifurcation = np.asarray([axs['C'], axs['c']])

axs_cb = np.asarray( [ [ axs['!'], axs['@'] ], [ axs["#"], axs['$'] ] ]  )


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

digit_classes = [4, 9]

for d_ic in range(2):
    d_probe = d_ic
    cmap_digit = get_custom_cmap(digit_classes[d_probe])
    
    data_pop = data_Ms_pop[:, :, d_ic, d_probe]/100.0
    norm_pop = matplotlib.colors.Normalize(vmin=np.min(data_pop)-0.05, vmax=np.max(data_pop)+0.05)
    
    axs_pop[d_ic].imshow(data_pop, cmap=cmap_digit, norm=norm_pop,
                         extent=[min(n_range), np.max(n_range), max(temp_range), np.min(temp_range)],
                         aspect=(max(n_range)-min(n_range))/(max(temp_range) - min(temp_range)))


    data_saddles = data_M_saddles_coefs[:, :, d_ic, d_probe]
    norm_saddles = matplotlib.colors.Normalize(vmin=np.min(data_saddles)-0.051, vmax=np.max(data_saddles)+0.05)



    axs_saddle[d_ic].imshow(data_saddles, cmap=cmap_digit, norm=norm_saddles,
                        extent=[min(n_range), np.max(n_range), max(temp_range), np.min(temp_range)],
                         aspect=(max(n_range)-min(n_range))/(max(temp_range) - min(temp_range)))

    
    cb = matplotlib.colorbar.ColorbarBase(axs_cb[d_ic, 0], cmap=cmap_digit, norm=norm_pop, orientation='vertical')
    cb.set_label("Proportion of "+str(digit_classes[d_probe]))

    cb = matplotlib.colorbar.ColorbarBase(axs_cb[d_ic, 1], cmap=cmap_digit, norm=norm_saddles, orientation='vertical')
    cb.set_label(r"$\alpha_"+str(digit_classes[d_probe])+r"$ of the saddle")


# bifurcation plots
t_index = -1
axs_bifurcation[0].scatter(data_Ms_pop[t_index, :, 0, 0], n_range)
axs_bifurcation[0].scatter(data_Ms_pop[t_index, :, 1, 0], n_range)

axs_bifurcation[1].scatter(data_M_saddles_coefs[t_index, :, 0, 0], n_range)
axs_bifurcation[1].scatter(data_M_saddles_coefs[t_index, :, 1, 0], n_range)




plt.subplots_adjust(wspace=0.075)
plt.savefig("Figure_pop.png")
exit()
