import sys
sys.path.append('../')

import numpy as np
import sys

from main_module.KrotovV2 import *

import matplotlib
font = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 30}
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

  
fig = plt.figure(figsize=(18+4+1, 10+3+6+3+11))
axs = fig.subplot_mosaic("""

111111..222222..333333x
111111..222222..333333x
111111..222222..333333x
111111..222222..333333x
111111..222222..333333x
111111..222222..333333x
.......................
.......................
.......................
AAAAAAAAAAABBBBBBBBBBBX
AAAAAAAAAAABBBBBBBBBBBX
AAAAAAAAAAABBBBBBBBBBBX
AAAAAAAAAAABBBBBBBBBBBX
AAAAAAAAAAABBBBBBBBBBBX
AAAAAAAAAAABBBBBBBBBBBX
AAAAAAAAAAABBBBBBBBBBBX
AAAAAAAAAAABBBBBBBBBBBX
AAAAAAAAAAABBBBBBBBBBBX
AAAAAAAAAAABBBBBBBBBBBX
AAAAAAAAAAABBBBBBBBBBBX
.......................
.......................
.......................
DDDDDDDDDDDDDDDDDDDDDDD
DDDDDDDDDDDDDDDDDDDDDDD
DDDDDDDDDDDDDDDDDDDDDDD
DDDDDDDDDDDDDDDDDDDDDDD
DDDDDDDDDDDDDDDDDDDDDDD
DDDDDDDDDDDDDDDDDDDDDDD
DDDDDDDDDDDDDDDDDDDDDDD
DDDDDDDDDDDDDDDDDDDDDDD
DDDDDDDDDDDDDDDDDDDDDDD
DDDDDDDDDDDDDDDDDDDDDDD

""")


ax = np.asarray([axs['A'], axs['B']])

# Plotting memory samples and their coefficients
ax[0].imshow(merge_data(data_Ms[::4, ::4, 0, :].reshape(len(n_range[::4])*len(temp_range[::4]), 784), len(n_range[::4]), len(temp_range[::4])  ),
             cmap="bwr", vmin=-1, vmax=1, extent=extent, aspect=aspect)
    
ax[1].imshow(data_coefs[:, :, 0, 1], cmap="Reds", vmin=0.5, vmax=1, extent=extent, aspect=aspect) #

# Plotting slices
axs['D'].plot(data_coefs[-2, :, 0, 1], marker=".", color="red", label="T = "+str(temp_range[-2]))
axs['D'].plot(data_coefs[-3, :, 0, 1], marker=".", color="orange", label="T = "+str(temp_range[-3]))
axs['D'].plot(data_coefs[-4, :, 0, 1], marker=".", color="blue", label="T = "+str(temp_range[-4]))
axs['D'].set_xlabel(r"$n$-power")
axs['D'].set_ylabel(r"$\alpha_{4, 1}$")
axs['D'].legend()
             
# Plotting the theoretical prediction (see appendix)
n = np.arange(np.min(n_range), np.max(n_range), 0.01)
T_calc = (data_T[0]@data_T[0] + data_T[0]@data_T[1])/( 2 * ( np.arctanh( 1 - (1.0/2.0)**(1.0/(2.0*n)) ) )**(1.0/n) )
ax[1].scatter(n, T_calc, s=1, color="k")
ax[1].set_ylim(max(temp_range), min(temp_range))

# Setting the generalist and specialist images
axs['1'].imshow(np.mean(data_T, axis=0).reshape(28, 28), cmap="bwr", vmin=-1, vmax=1)
axs['3'].imshow(data_T[0].reshape(28, 28), cmap="bwr", vmin=-1, vmax=1)


# Cosmetics
ax[1].set_yticks([])
ax[0].set_xlabel(r"$n$", labelpad=10); ax[1].set_xlabel(r"$n$", labelpad=10)
ax[0].set_ylabel("Temperature", labelpad=10)

axs['1'].set_title("Generalist", pad=10)
axs['3'].set_title("Specialist", pad=10)

for c in ['1', '2', '3']:
    axs[c].set_xticks([]); axs[c].set_yticks([])


# Color bars
ax_cb_bwr = axs['x']
bwr_cmap = matplotlib.cm.bwr
bwr_norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
cb_bwr = matplotlib.colorbar.ColorbarBase(ax_cb_bwr, cmap=bwr_cmap, norm=bwr_norm, orientation='vertical')
cb_bwr.set_label("Pixel value")
    

ax_cb_intra = axs['X']
intra_cmap = matplotlib.cm.Reds
intra_norm = matplotlib.colors.Normalize(vmin=0.5, vmax=1)
cb_intra = matplotlib.colorbar.ColorbarBase(ax_cb_intra, cmap=intra_cmap, norm=intra_norm, orientation='vertical')
cb_intra.set_label(r"$\alpha_{4, 1}$")


# Panel labels

alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
rx = [1.0, 6.0/11.0, 6.0/23.0]
ry = [1.0, 6.0/11.0, 6.0/10.0]
for i, char in enumerate(['1', 'A', 'D']):
    axs[char].text(-0.2*rx[i], 1.0+0.1*ry[i], alphabet[i], transform=axs[char].transAxes, fontsize=63, verticalalignment='bottom', ha='right', fontfamily='Times New Roman', fontweight='bold')


    
plt.subplots_adjust(wspace=0.5)
plt.savefig("Figure-4.png")
