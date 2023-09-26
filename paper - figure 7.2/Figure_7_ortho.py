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
prefix = str(selected_digits)+"/" # I used main,and momentum #"main"#

temp_range = np.arange(500, 900, 20)
n_range = np.arange(2, 32, 2)

data_Ms = np.zeros((len(temp_range), len(n_range), 2, 100, 784))
data_Ls = np.zeros((len(temp_range), len(n_range), 2, 100, 10))

isFirstRun = False
if isFirstRun:
    for i, temp in enumerate(temp_range):
        for j, n in enumerate(n_range):
            for k in range(2):
                
                saving_dir=data_dir+prefix+"trained_net_end_n"+str(n)+"_T"+str(temp)+"ic"+str(selected_digits[k])+".npz"
                data_Ms[i, j, k] = np.load(saving_dir)['M']
                data_Ls[i, j, k] = np.load(saving_dir)['L']
                
        print(temp)

    np.save(data_dir+prefix+"data_Ms.npy", data_Ms)
    np.save(data_dir+prefix+"data_Ls.npy", data_Ls)



# Then

data_Ms = np.load(data_dir+prefix+"data_Ms.npy")
data_Ls = np.load(data_dir+prefix+"data_Ls.npy")

data_T = np.load(data_dir+"miniBatchs_images.npy")[0]
data_T_inv = np.linalg.pinv(data_T)


data_Ms_unique = np.zeros((len(temp_range), len(n_range), 2, 2, 784))

for i, temp in enumerate(temp_range):
        for j, n in enumerate(n_range):
            for k in range(2):
                for l in range(2):
                    if np.any(np.argmax(data_Ls[i, j, k], axis=-1) == selected_digits[l]):
                        index = np.argmax(np.argmax(data_Ls[i, j, k], axis=-1) == selected_digits[l])
                        data_Ms_unique[i, j, k, l] = data_Ms[i, j, k, index]

data_coefs = data_Ms_unique@data_T_inv                        

fig = plt.figure(figsize=(7+9*2, 105-68+1))
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
.........................
DDDDDDDD&.......dddddddd*
DDDDDDDD&.......dddddddd*
DDDDDDDD&.......dddddddd*
DDDDDDDD&.......dddddddd*
DDDDDDDD&.......dddddddd*
DDDDDDDD&.......dddddddd*
DDDDDDDD&.......dddddddd*
DDDDDDDD&.......dddddddd*
""")

axs_sample = np.asarray( [ [ axs['A'], axs['B'] ], [ axs['C'], axs['D'] ] ]  )
axs_ortho = np.asarray( [ [ axs['a'], axs['b'] ], [ axs['c'], axs['d'] ] ]  )
axs_cb_ortho = np.asarray( [ [ axs['@'], axs['$'] ], [ axs['?'], axs['*'] ] ]  )


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
    for d_probe in range(2):
        axs_sample[d_ic, d_probe].imshow(merge_data(data_Ms_unique[:, :, d_ic, d_probe, :].reshape(len(temp_range)*len(n_range), 784), len(n_range), len(temp_range)),
                                         cmap="bwr", vmin=-1, vmax=1,
                                         extent=[min(n_range), np.max(n_range), max(temp_range), np.min(temp_range)],
                                         aspect=(max(n_range)-min(n_range))/(max(temp_range) - min(temp_range)))

        cmap_ortho = get_custom_cmap(digit_classes[1-d_probe])
        data_ortho = data_coefs[:, :, d_ic, d_probe, 1-d_probe]
        norm_ortho = matplotlib.colors.Normalize(vmin=np.clip(np.min(data_ortho)-0.1, -1, 0.6), vmax=np.clip(np.max(data_ortho)+0.1, -1, 0.6))
        
        axs_ortho[d_ic, d_probe].imshow(data_ortho, cmap=cmap_ortho, norm=norm_ortho,
                                        extent=[min(n_range), np.max(n_range), max(temp_range), np.min(temp_range)],
                                        aspect=(max(n_range)-min(n_range))/(max(temp_range) - min(temp_range)))
        

        # Cosmetics
        axs_sample[d_ic, d_probe].set_ylabel("Temperature", labelpad=15)
        axs_ortho[d_ic, d_probe].set_ylabel("Temperature", labelpad=15)

        if d_probe != 0:
            axs_ortho[d_ic, d_probe].set_xlabel("$n$")
            axs_sample[d_ic, d_probe].set_xlabel("$n$")
        

        # Color bars
        std_norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)       
        cb = matplotlib.colorbar.ColorbarBase(axs_cb_ortho[d_ic, d_probe], cmap=cmap_ortho, norm=norm_ortho, orientation='vertical')
        cb.set_label(r"$\alpha_"+str(digit_classes[1-d_probe])+"$ value")


norm_std = matplotlib.colors.Normalize(vmin=-1, vmax=1)
for c in ['!', '#', '%', '&']:
    cb = matplotlib.colorbar.ColorbarBase(axs[c], cmap="bwr", norm=norm_std, orientation='vertical')
    cb.set_label("Pixel value")


alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
rx = [1.0, 1.0]
ry = [1.0, 1.0]
for i, char in enumerate(['A', 'C']):
    axs[char].text(-0.3*rx[i], 1.0+0.1*ry[i], alphabet[i], transform=axs[char].transAxes, fontsize=81, verticalalignment='bottom', ha='right', fontfamily='Times New Roman', fontweight='bold')
    
plt.subplots_adjust(wspace=0.075)
plt.savefig("Figure_ortho.png")
exit()
