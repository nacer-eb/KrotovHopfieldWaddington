import sys
sys.path.append('../')

import numpy as np
import sys

from main_module.KrotovV2 import *

import matplotlib
font = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 36}
matplotlib.rc('font', **font)


data_dir = "data_49_mean/"

temp_range = np.arange(400, 900, 20)
n_range = np.arange(2, 31, 1)

print(np.shape(temp_range))
print(np.shape(n_range))

data_Ms = np.zeros((len(temp_range), len(n_range), 2, 784))

data_T = np.load(data_dir+"n"+str(n_range[0])+"_T"+str(temp_range[0])+".npz")["miniBatchs_images"][0]
data_T_inv = np.linalg.pinv(data_T)


first_run = False

if first_run:

    for i, temp in enumerate(temp_range):
        for j, n in enumerate(n_range):
            saving_dir=data_dir+"n"+str(n)+"_T"+str(temp)+".npz"
            
            data = np.load(saving_dir)
            data_Ms[i, j] = data['M']
    
        print(i)
    np.save(data_dir + "data_Ms.npy", data_Ms)




    
data_Ms = np.load(data_dir + "data_Ms.npy")
data_coefs = data_Ms @ data_T_inv

aspect = (np.min(n_range) - np.max(n_range))/(np.min(temp_range) - np.max(temp_range))
extent = [np.min(n_range), np.max(n_range), np.max(temp_range), np.min(temp_range)]

fig = plt.figure(figsize=(22+1, 8+3+11+3+11))
axs = fig.subplot_mosaic("""
...AAAAAAAAAAAAAAAAx...
...AAAAAAAAAAAAAAAAx...
...AAAAAAAAAAAAAAAAx...
...AAAAAAAAAAAAAAAAx...
...AAAAAAAAAAAAAAAAx...
...AAAAAAAAAAAAAAAAx...
...AAAAAAAAAAAAAAAAx...
...AAAAAAAAAAAAAAAAx...
.......................
.......................
.......................
BBBBBBBBBBBbbbbbbbbbbbX
BBBBBBBBBBBbbbbbbbbbbbX
BBBBBBBBBBBbbbbbbbbbbbX
BBBBBBBBBBBbbbbbbbbbbbX
BBBBBBBBBBBbbbbbbbbbbbX
BBBBBBBBBBBbbbbbbbbbbbX
BBBBBBBBBBBbbbbbbbbbbbX
BBBBBBBBBBBbbbbbbbbbbbX
BBBBBBBBBBBbbbbbbbbbbbX
BBBBBBBBBBBbbbbbbbbbbbX
BBBBBBBBBBBbbbbbbbbbbbX
.......................
.......................
.......................
CCCCCCCCCCCccccccccccc#
CCCCCCCCCCCccccccccccc#
CCCCCCCCCCCccccccccccc#
CCCCCCCCCCCccccccccccc#
CCCCCCCCCCCccccccccccc#
CCCCCCCCCCCccccccccccc#
CCCCCCCCCCCccccccccccc#
CCCCCCCCCCCccccccccccc#
CCCCCCCCCCCccccccccccc#
CCCCCCCCCCCccccccccccc#
CCCCCCCCCCCccccccccccc#
""")


selected_digits = [4, 9]
axs['A'].imshow(merge_data(data_T, 2, 1), cmap="bwr", vmin=-1, vmax=1)

# Cosmetics
axs['A'].set_title("Training Samples", pad=20)
axs['A'].set_xticks([]); axs['A'].set_yticks([])


ax = np.asarray([[axs['B'], axs['b']], [axs['C'], axs['c']] ])
ax_cb = np.asarray([axs['X'], axs['#']])



for digit in [0, 1]:
    ax[digit, 0].imshow(merge_data(data_Ms[::3, ::3, digit, :].reshape(len(n_range[::3])*len(temp_range[::3]), 784), len(n_range[::3]), len(temp_range[::3])  ),
             cmap="bwr", vmin=-1, vmax=1, extent=extent, aspect=aspect)
    im = ax[digit, 1].imshow(data_coefs[:, :, digit, 1-digit], cmap="bwr", vmin=-1, vmax=1, extent=extent, aspect=aspect)

    n = np.arange(np.min(n_range), np.max(n_range), 0.01)
    T_calc = (data_T[0]@data_T[0] + data_T[0]@data_T[1])/( 2 * ( np.arctanh( 1 - (1.0/2.0)**(1.0/(2.0*n)) ) )**(1.0/n) )
    ax[digit, 1].scatter(n, T_calc, s=1, color="k")
    ax[digit, 1].set_ylim(max(temp_range), min(temp_range))

    ax[digit, 1].set_yticks([])
    ax[digit, 0].set_xlabel(r"$n$", labelpad=10); ax[digit, 1].set_xlabel(r"$n$", labelpad=10)
    ax[digit, 0].set_ylabel("Temperature", labelpad=10)

    bwr_cmap = matplotlib.cm.bwr
    bwr_norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
    cb_bwr = matplotlib.colorbar.ColorbarBase(ax_cb[digit], cmap=bwr_cmap, norm=bwr_norm, orientation='vertical')
    cb_bwr.set_label(r"$\alpha_" + str(selected_digits[1-digit]) + "$")


bwr_cmap = matplotlib.cm.bwr
bwr_norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
cb_bwr = matplotlib.colorbar.ColorbarBase(axs['x'], cmap=bwr_cmap, norm=bwr_norm, orientation='vertical')
cb_bwr.set_label("Digit class")


alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
rx = [0.7, 8.0/11.0, 8.0/11.0]
ry = [1.0, 8.0/11.0, 8.0/11.0]
for i, char in enumerate(['A', 'B', 'C']):
    axs[char].text(-0.2*rx[i], 1.0+0.1*ry[i], alphabet[i], transform=axs[char].transAxes, fontsize=77, verticalalignment='bottom', ha='right', fontfamily='Times New Roman', fontweight='bold')

plt.subplots_adjust(wspace=1.0)
plt.savefig("Figure-5.png")
