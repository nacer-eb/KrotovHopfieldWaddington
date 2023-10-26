import sys
sys.path.append('../')

import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from main_module.KrotovV2_utils import *

fontsize=24*1.5
font = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : fontsize}
matplotlib.rc('font', **font)


fig = plt.figure(figsize=(31, 59-20+1 ))
axs = fig.subplot_mosaic("""
AAAAAAAAAAAAAA.BBBBBBBBBBBBBB$
AAAAAAAAAAAAAA.BBBBBBBBBBBBBB$
AAAAAAAAAAAAAA.BBBBBBBBBBBBBB$
AAAAAAAAAAAAAA.BBBBBBBBBBBBBB$
AAAAAAAAAAAAAA.BBBBBBBBBBBBBB$
AAAAAAAAAAAAAA.BBBBBBBBBBBBBB$
AAAAAAAAAAAAAA.BBBBBBBBBBBBBB$
..............................
..............................
..............................
CCCCCCCC.DDDDDDDD!....EEEEEEEE
CCCCCCCC.DDDDDDDD!....EEEEEEEE
CCCCCCCC.DDDDDDDD!....EEEEEEEE
CCCCCCCC.DDDDDDDD!....EEEEEEEE
CCCCCCCC.DDDDDDDD!....EEEEEEEE
CCCCCCCC.DDDDDDDD!....EEEEEEEE
CCCCCCCC.DDDDDDDD!....EEEEEEEE
CCCCCCCC.DDDDDDDD!....EEEEEEEE
..............................
..............................
..............................
FFFFFFFF.GGGGGGGG@....HHHHHHHH
FFFFFFFF.GGGGGGGG@....HHHHHHHH
FFFFFFFF.GGGGGGGG@....HHHHHHHH
FFFFFFFF.GGGGGGGG@....HHHHHHHH
FFFFFFFF.GGGGGGGG@....HHHHHHHH
FFFFFFFF.GGGGGGGG@....HHHHHHHH
FFFFFFFF.GGGGGGGG@....HHHHHHHH
FFFFFFFF.GGGGGGGG@....HHHHHHHH
..............................
..............................
..............................
IIIIIIII.JJJJJJJJ#....KKKKKKKK
IIIIIIII.JJJJJJJJ#....KKKKKKKK
IIIIIIII.JJJJJJJJ#....KKKKKKKK
IIIIIIII.JJJJJJJJ#....KKKKKKKK
IIIIIIII.JJJJJJJJ#....KKKKKKKK
IIIIIIII.JJJJJJJJ#....KKKKKKKK
IIIIIIII.JJJJJJJJ#....KKKKKKKK
IIIIIIII.JJJJJJJJ#....KKKKKKKK

""")

#############################################################
#############################################################
# Obtain & Plot the training data for intra and inter cases #
#############################################################
#############################################################
data_dir_intra =  "data_44_intra/" #"data_[2, 2]_intra/"

digit_classes = [4, 9]#[3, 8]#[4, 7]#[4, 9]
data_dir_inter = "data_49_mean_inter/" # "data_"+str(digit_classes)+"_mean_inter/"

data_T_intra = np.load(data_dir_intra+"trained_net_end_n"+str(2)+"_T"+str(400)+".npz")['miniBatchs_images'][0]
data_T_inter = np.load(data_dir_inter+"trained_net_end_n"+str(2)+"_T"+str(400)+".npz")['miniBatchs_images'][0]

data_T_intra_inv = np.linalg.pinv(data_T_intra)
data_T_inter_inv = np.linalg.pinv(data_T_inter)

axs['A'].imshow(merge_data(data_T_intra, 2, 1), cmap="bwr", vmin=-1, vmax=1)
axs['B'].imshow(merge_data(data_T_inter, 2, 1), cmap="bwr", vmin=-1, vmax=1)

axs['A'].set_xticks([]); axs['A'].set_yticks([])
axs['B'].set_xticks([]); axs['B'].set_yticks([])


ax_cb_mem = axs['$']
bwr_cmap = matplotlib.cm.bwr
bwr_norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
cb_mem = matplotlib.colorbar.ColorbarBase(ax_cb_mem, cmap=bwr_cmap, norm=bwr_norm, orientation='vertical')
cb_mem.set_label("Pixel value")


#########################################
#########################################
# Obtain & Plot data for the 1 mem case #
#########################################
#########################################
temp_range = np.arange(400, 900, 20)
n_range = np.arange(2, 31, 1)

data_Ms_intra = np.zeros((len(temp_range), len(n_range), 1, 784))

first_run = True
if first_run:
    for i, temp in enumerate(temp_range):
        for j, n in enumerate(n_range):
            saving_dir=data_dir_intra+"trained_net_end_n"+str(n)+"_T"+str(temp)+".npz"
            
            data = np.load(saving_dir)
            data_Ms_intra[i, j] = data['M']
    
        print(i)
    np.save(data_dir_intra + "data_Ms.npy", data_Ms_intra)
data_Ms_intra = np.load(data_dir_intra + "data_Ms.npy")

data_coefs_intra = data_Ms_intra @ data_T_intra_inv
    
aspect = (np.min(n_range) - np.max(n_range))/(np.min(temp_range) - np.max(temp_range))
extent = [np.min(n_range), np.max(n_range), np.max(temp_range), np.min(temp_range)]

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


ax = np.asarray([axs['C'], axs['D']])
ax[0].imshow(merge_data(data_Ms_intra[::4, ::4, 0, :].reshape(len(n_range[::4])*len(temp_range[::4]), 784), len(n_range[::4]), len(temp_range[::4])  ),
             cmap="bwr", vmin=-1, vmax=1, extent=extent, aspect=aspect)
ax[0].set_xlabel(r"$n$")
ax[0].set_ylabel(r"Temperature", labelpad=20)

cmap_ortho = get_custom_cmap(4)
data_ortho = data_coefs_intra[:, :, 0, 1] # <<<<<<<<<<<<----- MAKE THE CHOICE OF INDEX AUTOMATIC!!!!
norm_ortho = matplotlib.colors.Normalize(0.5, 1)
ax[1].imshow(data_ortho, cmap=cmap_ortho, norm=norm_ortho, extent=extent, aspect=aspect) #
ax[1].set_xlabel(r"$n$")
ax[1].set_yticks([])

# Plotting the theoretical prediction (see appendix)
n = np.arange(np.min(n_range), np.max(n_range), 0.01)
T_calc = (data_T_intra[0]@data_T_intra[0] + data_T_intra[0]@data_T_intra[1])/( 2 * ( np.arctanh( 1 - (1.0/2.0)**(1.0/(2.0*n)) ) )**(1.0/n) )
ax[1].scatter(n, T_calc, s=1, color="red")
ax[1].set_ylim(max(temp_range), min(temp_range))


# Plotting slices
axs['E'].plot(data_coefs_intra[-2, :, 0, 1], marker=".", color="red", label="T = "+str(temp_range[-2]))
axs['E'].plot(data_coefs_intra[-3, :, 0, 1], marker=".", color="orange", label="T = "+str(temp_range[-3]))
axs['E'].plot(data_coefs_intra[-4, :, 0, 1], marker=".", color="blue", label="T = "+str(temp_range[-4]))
axs['E'].set_xlabel(r"$n$")
axs['E'].legend()

ax_cb_intra = axs['!']
cb_intra = matplotlib.colorbar.ColorbarBase(ax_cb_intra, cmap=cmap_ortho, norm=norm_ortho, orientation='vertical')
cb_intra.set_label(r"$\alpha_{4, 1}$", labelpad=50)



#########################################
#########################################
# Obtain & Plot data for the 2 mem case #
#########################################
#########################################


temp_range = np.arange(400, 900, 20)
n_range = np.arange(2, 31, 1)
data_Ms_inter = np.zeros((len(temp_range), len(n_range), 2, 784))

first_run = True
if first_run:
    for i, temp in enumerate(temp_range):
        for j, n in enumerate(n_range):
            saving_dir=data_dir_inter+"trained_net_end_n"+str(n)+"_T"+str(temp)+".npz"
            
            data = np.load(saving_dir)
            data_Ms_inter[i, j] = data['M']
    
        print(i)
    np.save(data_dir_inter + "data_Ms.npy", data_Ms_inter)
data_Ms_inter = np.load(data_dir_inter + "data_Ms.npy")

data_coefs_inter = data_Ms_inter @ data_T_inter_inv

aspect = (np.min(n_range) - np.max(n_range))/(np.min(temp_range) - np.max(temp_range))
extent = [np.min(n_range), np.max(n_range), np.max(temp_range), np.min(temp_range)]

ax = np.asarray([[axs['F'], axs['G']], [axs['I'], axs['J']] ])
ax_1d = np.asarray([axs['H'], axs['K']])
ax_cb = np.asarray([axs['@'], axs['#']])



for digit in [0, 1]:
    ax[digit, 0].imshow(merge_data(data_Ms_inter[::3, ::3, digit, :].reshape(len(n_range[::3])*len(temp_range[::3]), 784), len(n_range[::3]), len(temp_range[::3])  ),
             cmap="bwr", vmin=-1, vmax=1, extent=extent, aspect=aspect)


    cmap_ortho = get_custom_cmap(digit_classes[1-digit])
    data_ortho = data_coefs_inter[:, :, digit, 1-digit]
    norm_ortho = matplotlib.colors.Normalize(vmin=np.clip(np.min(data_ortho)-0.1, -1, 0.6), vmax=np.clip(np.max(data_ortho)+0.1, -1, 0.6))
    im = ax[digit, 1].imshow(data_ortho, cmap=cmap_ortho, norm=norm_ortho, extent=extent, aspect=aspect)

    n = np.arange(np.min(n_range), np.max(n_range), 0.01)

    # I used 1-digit here to deal with potential asymmetry, but it shouldn't matter A@A should be roughly B@B
    T_calc = (data_T_inter[1-digit]@data_T_inter[1-digit] + data_T_inter[0]@data_T_inter[1])/( 2 * ( np.arctanh( 1 - (1.0/2.0)**(1.0/(2.0*n)) ) )**(1.0/n) )
    ax[digit, 1].scatter(n, T_calc, s=1, c="red")
    ax[digit, 1].set_ylim(max(temp_range), min(temp_range))

    ax[digit, 1].set_yticks([])
    ax[digit, 0].set_xlabel(r"$n$", labelpad=10); ax[digit, 1].set_xlabel(r"$n$", labelpad=10)
    ax[digit, 0].set_ylabel("Temperature", labelpad=20)

    cb_bwr = matplotlib.colorbar.ColorbarBase(ax_cb[digit], cmap=cmap_ortho, norm=norm_ortho, orientation='vertical')
    cb_bwr.set_label(r"$\alpha_" + str(digit_classes[1-digit]) + "$", labelpad=20)

    colors=["red", "orange", "blue"]
    for t_i, t in enumerate([-1, -5, -8]):
        ax_1d[digit].plot(data_coefs_inter[t, :, digit, 1-digit], marker=".", label="T = "+str(temp_range[t]), color=colors[t_i])
        ax_1d[digit].set_xlabel(r"$n$")
        ax_1d[digit].legend()



plt.savefig("Figure_4_tmp.png")
