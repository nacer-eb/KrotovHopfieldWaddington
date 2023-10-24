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


data_dir = "data_4_rand/"

temp_range = np.arange(400, 900, 20)
n_range = np.arange(2, 31, 1)

print(np.shape(temp_range))
print(np.shape(n_range))

data_Ms = np.zeros((len(temp_range), len(n_range), 1, 784))

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

  
fig = plt.figure(figsize=(8*3+1+4+1, 73-56+1))
axs = fig.subplot_mosaic("""
00000000000000................
00000000000000................
00000000000000................
00000000000000................
00000000000000................
00000000000000................
00000000000000................
..............................
..............................
..............................
AAAAAAAA.BBBBBBBBX....DDDDDDDD
AAAAAAAA.BBBBBBBBX....DDDDDDDD
AAAAAAAA.BBBBBBBBX....DDDDDDDD
AAAAAAAA.BBBBBBBBX....DDDDDDDD
AAAAAAAA.BBBBBBBBX....DDDDDDDD
AAAAAAAA.BBBBBBBBX....DDDDDDDD
AAAAAAAA.BBBBBBBBX....DDDDDDDD
AAAAAAAA.BBBBBBBBX....DDDDDDDD
""")



axs['0'].imshow(merge_data(data_T, 2, 1), cmap="bwr", vmin=-1, vmax=1)

# Cosmetics
axs['0'].set_title("Training Samples", pad=20)
axs['0'].set_xticks([]); axs['0'].set_yticks([])


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


ax = np.asarray([axs['A'], axs['B']])

# Plotting memory samples and their coefficients
ax[0].imshow(merge_data(data_Ms[::4, ::4, 0, :].reshape(len(n_range[::4])*len(temp_range[::4]), 784), len(n_range[::4]), len(temp_range[::4])  ),
             cmap="bwr", vmin=-1, vmax=1, extent=extent, aspect=aspect)


cmap_ortho = get_custom_cmap(4)
data_ortho = data_coefs[:, :, 0, 1]
norm_ortho = matplotlib.colors.Normalize(0.5, 1)
ax[1].imshow(data_ortho, cmap=cmap_ortho, norm=norm_ortho, extent=extent, aspect=aspect) #

# Plotting slices
axs['D'].plot(data_coefs[-2, :, 0, 1], marker=".", color="red", label="T = "+str(temp_range[-2]))
axs['D'].plot(data_coefs[-3, :, 0, 1], marker=".", color="orange", label="T = "+str(temp_range[-3]))
axs['D'].plot(data_coefs[-4, :, 0, 1], marker=".", color="blue", label="T = "+str(temp_range[-4]))
axs['D'].set_xlabel(r"$n$")
#axs['D'].set_ylabel(r"$\alpha_{4, 1}$")
axs['D'].legend()
             
# Plotting the theoretical prediction (see appendix)
n = np.arange(np.min(n_range), np.max(n_range), 0.01)
T_calc = (data_T[0]@data_T[0] + data_T[0]@data_T[1])/( 2 * ( np.arctanh( 1 - (1.0/2.0)**(1.0/(2.0*n)) ) )**(1.0/n) )
ax[1].scatter(n, T_calc, s=1, color="red")
ax[1].set_ylim(max(temp_range), min(temp_range))



# Cosmetics
ax[1].set_yticks([])
ax[0].set_xlabel(r"$n$", labelpad=10); ax[1].set_xlabel(r"$n$", labelpad=10)
ax[0].set_ylabel("Temperature", labelpad=10)

    

ax_cb_intra = axs['X']
cb_intra = matplotlib.colorbar.ColorbarBase(ax_cb_intra, cmap=cmap_ortho, norm=norm_ortho, orientation='vertical')
cb_intra.set_label(r"$\alpha_{4, 1}$", labelpad=50)


plt.subplots_adjust(wspace=0.05)
plt.savefig("Figure-4_tomerge.png")
