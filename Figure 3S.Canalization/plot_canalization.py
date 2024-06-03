import sys
sys.path.append('../')

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec

from main_module.KrotovV2 import *



data_dir = "data/"

# The digit classes to include in the training
selected_digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] 
prefix = str(selected_digits)+"/"

n, temp = 30, 670 #601 701
saving_dir=data_dir+prefix+"trained_net_n"+str(n)+"_T"+str(temp)+".npz"


data = np.load(saving_dir)
data_M = data['M']
data_L = data['L']
data_T = data['miniBatchs_images'][0]
data_T_inv = np.linalg.pinv(data_T)

tmax, N_mem, N = data_M.shape

alphas = data_M@data_T_inv


alphas_d = np.sum(alphas.reshape(tmax, N_mem, 10, 20), axis=-1)

fig = plt.figure(figsize=(16, 9), layout="constrained")
gs = GridSpec(9, 16, figure=fig)
#ax = fig.add_subplot()#(projection='3d')
ax_samples = [0 for i in range(4)]

ax = fig.add_subplot(gs[4:, :16])
ax.set_xlabel("Epoch of training")
ax.set_ylabel("Deviation from mean")

ax.set_xlim(-100, 600+100)
ax.set_xticks(np.arange(0, 600+100, 200))

for i in range(4):
    ax_samples = fig.add_subplot(gs[:4, 4*i:4*(i+1)])
    ax_samples.set_xticks([])
    ax_samples.set_yticks([])
    ax_samples.imshow(merge_data(data_M[200*i], 10, 10), cmap="bwr", vmin=-1, vmax=1)

    ax.axvline(200*i, color="grey", linewidth=3, alpha=0.9)


tab10_cmap = matplotlib.cm.tab10
tab10_norm = matplotlib.colors.Normalize(vmin=0, vmax=10)

t_f = 600+100
dt = 1


#data_M = alphas@data_T # This is more honest (and works very well) but so hard to explain shortly. Alternative works very well too
for mu in range(N_mem):
    #keys = np.argmax(data_L[-1], axis=-1)
    data_M_digitclass = data_M#[:, keys==keys[mu]]
    mean_Ms = np.mean(data_M_digitclass, axis=1)[:t_f]

    #if np.argmax(data_L[-1, mu]) == 1:
    ax.plot(np.sum(np.abs(mean_Ms - data_M[:t_f, mu]), axis=-1), color=tab10_cmap(tab10_norm(np.argmax(data_L[-1, mu]))), zorder=2)
    #ax.plot(np.sum(np.abs(mean_Ms - data_M[:t_f, mu]), axis=-1), color=tab10_cmap(tab10_norm(np.argmax(data_L[-1, mu]))))
    #ax.plot(np.sum(alphas_d[:t_f, mu, [1, 7]], axis=-1), color=tab10_cmap(tab10_norm(np.argmax(data_L[-1, mu]))))
    #ax.plot(alphas_d[:t_f, mu, 7], color=tab10_cmap(tab10_norm(np.argmax(data_L[-1, mu]))))



plt.savefig("layout.pdf")


