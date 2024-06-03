import numpy as np
import matplotlib.pyplot as plt

import matplotlib
font = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 40}

matplotlib.rc('font', **font)


data_dir = "C_Code_FPs/"
data_files = ["low", "mid", "high"]

axs_3d_pos = [0]*3

fig = plt.figure(figsize=((11*3+4)*0.85, (54-20+1)*0.85))
axs = fig.subplot_mosaic(
    """
    11111111111..22222222222..33333333333
    11111111111..22222222222..33333333333
    11111111111..22222222222..33333333333
    11111111111..22222222222..33333333333
    11111111111..22222222222..33333333333
    11111111111..22222222222..33333333333
    11111111111..22222222222..33333333333
    11111111111..22222222222..33333333333
    11111111111..22222222222..33333333333
    11111111111..22222222222..33333333333
    11111111111..22222222222..33333333333
    .....................................
    44444444444..55555555555..66666666666
    44444444444..55555555555..66666666666
    44444444444..55555555555..66666666666
    44444444444..55555555555..66666666666
    44444444444..55555555555..66666666666
    44444444444..55555555555..66666666666
    44444444444..55555555555..66666666666
    44444444444..55555555555..66666666666
    44444444444..55555555555..66666666666
    44444444444..55555555555..66666666666
    .....................................
    77777777777..88888888888..99999999999
    77777777777..88888888888..99999999999
    77777777777..88888888888..99999999999
    77777777777..88888888888..99999999999
    77777777777..88888888888..99999999999
    77777777777..88888888888..99999999999
    77777777777..88888888888..99999999999
    77777777777..88888888888..99999999999
    77777777777..88888888888..99999999999
    77777777777..88888888888..99999999999
    77777777777..88888888888..99999999999
    """
)

ax = np.asarray([ [axs['1'], axs['2'], axs['3']],
                  [axs['4'], axs['5'], axs['6']],
                  [axs['7'], axs['8'], axs['9']]])



for f_i in range(0, 3):
    data_file = data_dir + data_files[f_i] + ".dat"
    data = np.loadtxt(data_file, skiprows=1, delimiter=",")
    
    p_mask = (np.abs(data[:, -2]) > 0.001) * (np.abs(data[:, -3]) > 0.001 ) * (data[:, 1] <= 7) + (np.abs(data[:, -2]) > 0.03) * (np.abs(data[:, -3]) > 0.03 ) * (data[:, 1] > 7)
    data = data[p_mask, :]


    #purple orange cmap
    norm = plt.Normalize(np.min(data[:, 1]), np.max(data[:, 1]))
    
    c1 = np.asarray([191/256.0, 127/256.0, 191/256.0, 1])
    c2 = np.asarray([255/256.0, 209/256.0, 127/256.0, 1])
    
    k = np.linspace(0, 1, 256)
    
    vals = np.zeros((256, 4))
    for i in range(0, 256):
        vals[i] = c1*(1 - k[i]) + c2*k[i]
    cmap = matplotlib.colors.ListedColormap(vals)
    
    ax[f_i, 0].scatter(data[:, 3], data[:, 1], c=data[:, 1], cmap=cmap, norm=norm, s=4)
    ax[f_i, -1].scatter(data[:, -1], data[:, 1], c=data[:, 1], cmap=cmap, norm=norm, s=4)

    ax[0, 0].set_xticks([]); ax[1, 0].set_xticks([]); ax[0, -1].set_xticks([]); ax[1, -1].set_xticks([]);
    ax[-1, 0].set_xlabel(r"$\alpha$"); ax[f_i, 0].set_ylabel(r"$n$")

    axs_3d_pos[f_i] = ax[f_i, 1].get_position()
    ax[f_i, 1].remove()
    
    ax[-1, -1].set_xlabel(r"$\ell$"); ax[f_i, -1].set_ylabel(r"$n$")
    ax[f_i, -1].yaxis.set_label_position("right"); ax[f_i, -1].yaxis.tick_right()



# 3d

for f_i in range(0, 3):
    
    dx_adjust = 0.02*0
    dy_adjust = 0.0
    default_pos_ax2 = axs_3d_pos[f_i]
    center_ax_3d = fig.add_axes([default_pos_ax2.x0 + dx_adjust,
                                 default_pos_ax2.y0 + dy_adjust,
                                 default_pos_ax2.x1-default_pos_ax2.x0-2*dx_adjust,
                                 default_pos_ax2.y1 - default_pos_ax2.y0-2*dy_adjust],
                                projection='3d')

    data_file = data_dir + data_files[f_i] + ".dat"
    data = np.loadtxt(data_file, skiprows=1, delimiter=",")
    
    p_mask = (np.abs(data[:, -2]) > 0.001) * (np.abs(data[:, -3]) > 0.001 ) * (data[:, 1] <= 7) + (np.abs(data[:, -2]) > 0.03) * (np.abs(data[:, -3]) > 0.03 ) * (data[:, 1] > 7)
    data = data[p_mask, :]


    #purple orange cmap
    norm = plt.Normalize(np.min(data[:, 1]), np.max(data[:, 1]))
    
    c1 = np.asarray([191/256.0, 127/256.0, 191/256.0, 1])
    c2 = np.asarray([255/256.0, 209/256.0, 127/256.0, 1])
    
    k = np.linspace(0, 1, 256)
    
    vals = np.zeros((256, 4))
    for i in range(0, 256):
        vals[i] = c1*(1 - k[i]) + c2*k[i]
    cmap = matplotlib.colors.ListedColormap(vals)
    
    center_ax_3d.scatter(data[:, 3], data[:, -1], data[:, 1], c=data[:, 1], cmap=cmap, norm=norm, s=5)
    center_ax_3d.set_xlabel(r"$\alpha$", labelpad=25); center_ax_3d.set_ylabel(r"$\ell$", labelpad=25); center_ax_3d.set_zlabel(r"$n$", labelpad=10)
    center_ax_3d.set_xticks([0, 0.5, 1]); center_ax_3d.set_xticklabels([0, 0.5, 1])
    center_ax_3d.set_yticks([-1, 0, 1]); center_ax_3d.set_yticklabels([-1, 0, 1])
    center_ax_3d.set_zticks([20, 40]); center_ax_3d.set_zticklabels([20, 40])
    
    center_ax_3d.locator_params(axis='x', nbins=5)
    center_ax_3d.locator_params(axis='y', nbins=5)

#plt.subplots_adjust(top=0.95, bottom=0.06, left=0.13, right=0.9, hspace=0.05, wspace=0.05)
plt.savefig("Figure_6S_SaddleNode_1_tmp.png")
#plt.show()
    
