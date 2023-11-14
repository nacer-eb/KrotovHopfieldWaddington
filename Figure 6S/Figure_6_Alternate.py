import sys
sys.path.append('../')

import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from main_module.KrotovV2_utils import *

fontsize=63
font = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : fontsize}
matplotlib.rc('font', **font)



fig = plt.figure(figsize=(36, 85-22+1 ))
axs = fig.subplot_mosaic("""

AAAAAAAAAA...........BBBBBBBBBB......
AAAAAAAAAA...........BBBBBBBBBB......
AAAAAAAAAA...........BBBBBBBBBB......
AAAAAAAAAA...........BBBBBBBBBB......
AAAAAAAAAA...........BBBBBBBBBB......
.....................................
.....................................
.....................................
.....................................
.....................................
.....................................
.....................................
.....................................
CCCCCCCCCC.....FFFFFFFFFF..IIIIIIIIII
CCCCCCCCCC.....FFFFFFFFFF..IIIIIIIIII
CCCCCCCCCC.....FFFFFFFFFF..IIIIIIIIII
CCCCCCCCCC.....FFFFFFFFFF..IIIIIIIIII
CCCCCCCCCC.....FFFFFFFFFF..IIIIIIIIII
CCCCCCCCCC.....FFFFFFFFFF..IIIIIIIIII
CCCCCCCCCC.....FFFFFFFFFF..IIIIIIIIII
CCCCCCCCCC.....FFFFFFFFFF..IIIIIIIIII
CCCCCCCCCC.....FFFFFFFFFF..IIIIIIIIII
CCCCCCCCCC.....FFFFFFFFFF..IIIIIIIIII
.....................................
.....................................
.....................................
.....................................
.....................................
.....................................
.....................................
.....................................
.....................................
.....................................
.....................................
.....................................
dddddddddd.....gggggggggg..jjjjjjjjjj
DDDDDDDDDD.....GGGGGGGGGG..JJJJJJJJJJ
DDDDDDDDDD.....GGGGGGGGGG..JJJJJJJJJJ
DDDDDDDDDD.....GGGGGGGGGG..JJJJJJJJJJ
DDDDDDDDDD.....GGGGGGGGGG..JJJJJJJJJJ
DDDDDDDDDD.....GGGGGGGGGG..JJJJJJJJJJ
DDDDDDDDDD.....GGGGGGGGGG..JJJJJJJJJJ
DDDDDDDDDD.....GGGGGGGGGG..JJJJJJJJJJ
DDDDDDDDDD.....GGGGGGGGGG..JJJJJJJJJJ
DDDDDDDDDD.....GGGGGGGGGG..JJJJJJJJJJ
DDDDDDDDDD.....GGGGGGGGGG..JJJJJJJJJJ
.....................................
.....................................
.....................................
EEEEEEEEEE.....HHHHHHHHHH..KKKKKKKKKK
EEEEEEEEEE.....HHHHHHHHHH..KKKKKKKKKK
EEEEEEEEEE.....HHHHHHHHHH..KKKKKKKKKK
EEEEEEEEEE.....HHHHHHHHHH..KKKKKKKKKK
EEEEEEEEEE.....HHHHHHHHHH..KKKKKKKKKK
EEEEEEEEEE.....HHHHHHHHHH..KKKKKKKKKK
EEEEEEEEEE.....HHHHHHHHHH..KKKKKKKKKK
EEEEEEEEEE.....HHHHHHHHHH..KKKKKKKKKK
EEEEEEEEEE.....HHHHHHHHHH..KKKKKKKKKK
EEEEEEEEEE.....HHHHHHHHHH..KKKKKKKKKK
.....................................
.....................................
.....................................
.....................................
.....................................
""")





#############################################################
#############################################################
# Obtain & Plot the training data for intra and inter cases #
#############################################################
#############################################################
data_dir_intra =  "data/[5, 5]_intra/" #"data_[2, 2]_intra/"

digit_classes = [5, 8]#[3, 8]#[4, 7]#[4, 9]
data_dir_inter = "data/[5, 8]_mean_inter/" # "data_"+str(digit_classes)+"_mean_inter/"

data_T_intra = np.load(data_dir_intra+"trained_net_end_n"+str(2)+"_T"+str(400)+".npz")['miniBatchs_images'][0]
data_T_inter = np.load(data_dir_inter+"trained_net_end_n"+str(2)+"_T"+str(400)+".npz")['miniBatchs_images'][0]

data_T_intra_inv = np.linalg.pinv(data_T_intra)
data_T_inter_inv = np.linalg.pinv(data_T_inter)

axs['A'].imshow(merge_data(data_T_intra, 2, 1), cmap="bwr", vmin=-1, vmax=1)
axs['B'].imshow(merge_data(data_T_inter, 2, 1), cmap="bwr", vmin=-1, vmax=1)

axs['A'].set_xticks([]); axs['A'].set_yticks([])
axs['B'].set_xticks([]); axs['B'].set_yticks([])


"""

ax_cb_mem = axs['$']
bwr_cmap = matplotlib.cm.bwr
bwr_norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
cb_mem = matplotlib.colorbar.ColorbarBase(ax_cb_mem, cmap=bwr_cmap, norm=bwr_norm, orientation='vertical')
cb_mem.set_label("Pixel value")

"""

#########################################
#########################################
# Obtain & Plot data for the 1 mem case #
#########################################
#########################################
temp_range = np.arange(400, 900, 20)
temp_range_rescaled = temp_range/784.0
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
    
aspect = (np.min(n_range) - np.max(n_range))/(np.min(temp_range_rescaled) - np.max(temp_range_rescaled))
extent = [np.min(n_range), np.max(n_range), np.max(temp_range_rescaled), np.min(temp_range_rescaled)]

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

def get_custom_cmap_2(digit_class):
    custom_cmap = np.zeros((256, 4))
    colors = np.asarray([np.asarray(tab10_cmap(tab10_norm(digit_class))), np.asarray([1.0, 1.0, 1.0, 1.0])])
    
    x = np.linspace(0, 1, 256)
    for i in range(256):
        custom_cmap[i] = x[i]*colors[1] + (1.0-x[i])*colors[0]
    custom_cmap = matplotlib.colors.ListedColormap(custom_cmap)
    
    return custom_cmap

def get_custom_cmap_2_inv(digit_class):
    custom_cmap = np.zeros((256, 4))
    colors = np.asarray([np.asarray([1.0, 1.0, 1.0, 1.0]), np.asarray(tab10_cmap(tab10_norm(digit_class)))])
    
    x = np.linspace(0, 1, 256)
    for i in range(256):
        custom_cmap[i] = x[i]*colors[1] + (1.0-x[i])*colors[0]
    custom_cmap = matplotlib.colors.ListedColormap(custom_cmap)
    
    return custom_cmap

def get_custom_cmap_3(digit_class, digit_class_2):
    custom_cmap = np.zeros((256, 4))
    colors = np.asarray([np.asarray(tab10_cmap(tab10_norm(digit_class))), np.asarray([1.0, 1.0, 1.0, 1.0]), np.asarray(tab10_cmap(tab10_norm(digit_class_2)))])
    
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
ax[0].set_ylabel(r"Rescaled Temperature", labelpad=20)
    
cmap_ortho = get_custom_cmap(digit_classes[0]) # By default
data_ortho = data_coefs_intra[:, :, 0, 0] # <<<<<<<<<<<<----- MAKE THE CHOICE OF INDEX AUTOMATIC!!!!
norm_ortho = matplotlib.colors.Normalize(0.4, 1)
ax[1].imshow(data_ortho, cmap=cmap_ortho, norm=norm_ortho, extent=extent, aspect=aspect) #
ax[1].set_xlabel(r"$n$")
ax[1].set_ylabel(r"Rescaled Temperature", labelpad=20)





# Plotting slices
colors=["red", "orange", "blue"]
for t_i, t in enumerate([-1, -3, -5]):
    ax[1].axhline(y=temp_range_rescaled[t]-0.004, xmax=1, color=colors[t_i], linewidth=10, label=r"$T_r$ = "+'{0:.2f}'.format(temp_range_rescaled[t]))
    axs['E'].plot(data_ortho[t], marker=".", color=colors[t_i], linewidth=10)
    axs['E'].set_xlabel(r"$n$")
    axs['E'].set_ylabel(r"$\alpha$", labelpad=20)
    axs['E'].set_xlim(0, 30)
    axs['E'].set_xticks([10, 20, 30])
    axs['E'].set_xticklabels([10, 20, 30])

import matplotlib.patheffects as pe

# Plotting the theoretical prediction (see appendix)
n = np.arange(np.min(n_range), np.max(n_range), 0.01)
T_calc = (data_T_intra[0]@data_T_intra[0] + data_T_intra[0]@data_T_intra[1])/( 2 * ( np.arctanh( 1 - (1.0/2.0)**(1.0/(2.0*n)) ) )**(1.0/n) )
T_calc_rescaled = T_calc/784.0
ax[1].plot(n, T_calc_rescaled, color='white', linewidth=10, path_effects=[pe.Stroke(linewidth=15, foreground='k'), pe.Normal()], label="Theoretical line")
ax[1].set_ylim(max(temp_range_rescaled), min(temp_range_rescaled))


ax_cb_intra = axs['d']
cb_bwr = matplotlib.colorbar.ColorbarBase(ax_cb_intra, cmap=cmap_ortho, norm=norm_ortho, orientation='horizontal')
cb_bwr.set_label(r"$\alpha_{"+str(digit_classes[0])+", 1}$", labelpad=40)
ax_cb_intra.xaxis.set_ticks_position('top')
ax_cb_intra.xaxis.set_label_position('top')

ax[1].legend(bbox_to_anchor=(1, -1.6))



#########################################
#########################################
# Obtain & Plot data for the 2 mem case #
#########################################
#########################################


temp_range = np.arange(400, 900, 20)
temp_range_rescaled = temp_range/784.0
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

aspect = (np.min(n_range) - np.max(n_range))/(np.min(temp_range_rescaled) - np.max(temp_range_rescaled))
extent = [np.min(n_range), np.max(n_range), np.max(temp_range_rescaled), np.min(temp_range_rescaled)]

ax = np.asarray([[axs['F'], axs['G']], [axs['I'], axs['J']] ])
ax_1d = np.asarray([axs['H'], axs['K']])

ax_cb = np.asarray([axs['g'], axs['j']])

# -0.5 to 0.5

for digit in [0, 1]:
    ax[digit, 0].imshow(merge_data(data_Ms_inter[::4, ::4, digit, :].reshape(len(n_range[::4])*len(temp_range[::4]), 784), len(n_range[::4]), len(temp_range[::4])  ),
             cmap="bwr", vmin=-1, vmax=1, extent=extent, aspect=aspect)


    cmap_ortho = get_custom_cmap_2(digit_classes[digit]) # get_custom_cmap(digit_classes[1-digit])
    data_ortho = data_coefs_inter[:, :, digit, 1-digit]

    if digit == 0:
        norm_ortho = matplotlib.colors.Normalize(vmin=-0.5, vmax=0.5)
    if digit == 1:
        norm_ortho = matplotlib.colors.Normalize(vmin=-0.5, vmax=0.5)
    
    im = ax[digit, 1].imshow(data_ortho, cmap=cmap_ortho, norm=norm_ortho, extent=extent, aspect=aspect)

    n = np.arange(np.min(n_range), np.max(n_range), 0.01)

    # I used 1-digit here to deal with potential asymmetry, but it shouldn't matter A@A should be roughly B@B
    T_calc = (data_T_inter[1-digit]@data_T_inter[1-digit] + data_T_inter[0]@data_T_inter[1])/( 2 * ( np.arctanh( 1 - (1.0/2.0)**(1.0/(2.0*n)) ) )**(1.0/n) )
    T_calc_rescaled = T_calc/784.0
    
    
    ax[digit, 1].set_ylim(max(temp_range_rescaled), min(temp_range_rescaled))

    ax[1, 0].set_yticks([])
    ax[1, 1].set_yticks([])
    
    
    ax[digit, 0].set_xlabel(r"$n$", labelpad=10); ax[digit, 1].set_xlabel(r"$n$", labelpad=10)
    ax[0, 0].set_ylabel("Rescaled Temperature", labelpad=20)
    ax[0, 1].set_ylabel("Rescaled Temperature", labelpad=20)
    
    
    cb_bwr = matplotlib.colorbar.ColorbarBase(ax_cb[digit], cmap=cmap_ortho, norm=norm_ortho, orientation='horizontal')
    cb_bwr.set_label(r"$\alpha_" + str(digit_classes[1-digit]) + "$", labelpad=40)
    ax_cb[digit].xaxis.set_ticks_position('top')
    ax_cb[digit].xaxis.set_label_position('top')
    cb_bwr.set_ticks([-0.5, 0, 0.5]) # Finally found how to center these things 
    cb_bwr.set_ticklabels([-0.5, 0, 0.5])

    colors=["red", "orange", "blue"]
    for t_i, t in enumerate([-1, -5, -8]):
        ax[digit, 1].axhline(y=temp_range_rescaled[t]-0.004, xmax=1, color=colors[t_i], linewidth=10, label=r"$T_r$ = "+'{0:.2f}'.format(temp_range_rescaled[t]))
        ax_1d[digit].plot(data_coefs_inter[t, :, digit, 1-digit], marker=".", color=colors[t_i], linewidth=10)
        ax_1d[digit].set_xlabel(r"$n$")
        ax_1d[digit].set_xlim(0, 30)
        ax_1d[digit].set_ylim(-0.4, 0.4)
        ax_1d[digit].set_xticks([10, 20, 30])
        ax_1d[digit].set_xticklabels([10, 20, 30])

    ax_1d[0].set_ylabel(r"$\alpha$")
    ax_1d[1].set_yticks([])
        

    ax[digit, 1].plot(n, T_calc_rescaled, color='white', linewidth=10, path_effects=[pe.Stroke(linewidth=15, foreground='k'), pe.Normal()], label="Theoretical line")
    ax[digit, 1].legend(bbox_to_anchor=(1, -1.6))


cmap_ortho = get_custom_cmap_2_inv(digit_classes[1])
cb_bwr = matplotlib.colorbar.ColorbarBase(ax_cb[1], cmap=cmap_ortho, norm=norm_ortho, orientation='horizontal')
cb_bwr.set_label(r"$\alpha_" + str(digit_classes[1-digit]) + "$", labelpad=40)
ax_cb[1].xaxis.set_ticks_position('top')
ax_cb[1].xaxis.set_label_position('top')
cb_bwr.set_ticks([-0.5, 0, 0.5]) # Finally found how to center these things 
cb_bwr.set_ticklabels([0.5, 0, -0.5])

plt.savefig("Figure_6_tmp.png")
