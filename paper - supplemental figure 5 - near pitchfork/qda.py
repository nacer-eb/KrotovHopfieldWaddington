import numpy as np
import matplotlib.pyplot as plt

import matplotlib
font = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 14}

matplotlib.rc('font', **font)


data_dir = "C_code/"
data_files = ["low", "mid", "high"]

fig, ax = plt.subplots(3, 3)

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
    
    ax[f_i, 0].scatter(data[:, 3], data[:, 1], c=data[:, 1], cmap=cmap, norm=norm, s=0.1)
    ax[f_i, -1].scatter(data[:, -1], data[:, 1], c=data[:, 1], cmap=cmap, norm=norm, s=0.1)

    ax[0, 0].set_xticks([]); ax[1, 0].set_xticks([]); ax[0, -1].set_xticks([]); ax[1, -1].set_xticks([]);
    ax[-1, 0].set_xlabel(r"$\alpha$"); ax[f_i, 0].set_ylabel(r"$n$")
    ax[f_i, 1].remove()
    ax[-1, -1].set_xlabel(r"$l_0$"); ax[f_i, -1].set_ylabel(r"$n$")
    
    ax[f_i, -1].yaxis.set_label_position("right"); ax[f_i, -1].yaxis.tick_right()

plt.subplots_adjust(top=0.95, bottom=0.06, left=0.13, right=0.9, hspace=0.05, wspace=0.05)
plt.show()


# 3d

for f_i in range(0, 3):
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')

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
    
    ax.scatter(data[:, 3], data[:, -1], data[:, 1], c=data[:, 1], cmap=cmap, norm=norm, s=1)
    ax.set_xlabel(r"$\alpha$", labelpad=10); ax.set_ylabel(r"$l_0$", labelpad=13); ax.set_zlabel(r"$n$", labelpad=5)
    plt.tight_layout()
    
    plt.locator_params(axis='x', nbins=5)
    plt.locator_params(axis='y', nbins=5)
    plt.show()
    
