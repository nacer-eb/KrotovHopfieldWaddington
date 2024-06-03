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
data_M = data['M'][:1000]
data_L = data['L'][:1000]
data_T = data['miniBatchs_images'][0]
data_T_inv = np.linalg.pinv(data_T)
tmax, N_mem, N = data_M.shape

alphas = data_M@data_T_inv
alphas_d = np.sum(alphas.reshape(tmax, N_mem, 10, 20), axis=-1)


pulseTime=-1
saving_dir=data_dir+prefix+"trained_net_test_noisy_"+str(pulseTime)+"_n"+str(n)+"_T"+str(temp)+".npz"
data_noise = np.load(saving_dir)
data_noise_M = data_noise['M']
data_noise_L = data_noise['L']
data_T_noise = data_noise['miniBatchs_images'][0]

fig = plt.figure(figsize=(16, 9), layout="constrained")
gs = GridSpec(9, 16, figure=fig)

ax_1 = fig.add_subplot(gs[:5, :8])
ax_2 = fig.add_subplot(gs[:5, 8:])

ax_1.imshow(merge_data(data_M[240], 10, 10), cmap="bwr", vmin=-1, vmax=1)
ax_2.imshow(merge_data(data_noise_M[240], 10, 10), cmap="bwr", vmin=-1, vmax=1)

ax = fig.add_subplot(gs[5:, :])

ax.set_xlabel("Epoch of training")
ax.set_ylabel("Deviation")


tab10_cmap = matplotlib.cm.tab10
tab10_norm = matplotlib.colors.Normalize(vmin=0, vmax=10)

pulseTimes = np.arange(0, 10, 1, dtype=int)

data_noise_Ms = np.zeros((10, tmax, N_mem, 784))

for pulseTime in pulseTimes[::1]:
    saving_dir=data_dir+prefix+"trained_net_test_noisy_"+str(pulseTime)+"_n"+str(n)+"_T"+str(temp)+".npz"
    data_noise = np.load(saving_dir)
    data_noise_Ms[pulseTime] = data_noise['M']


    #ax.plot(np.mean(np.sum(np.abs(data_L[:] - data_noise_L[:]), axis=-1), axis=-1))#, color=tab10_cmap(tab10_norm(np.argmax(data_L[-1, mu]))))


mean_M = np.mean(data_noise_Ms, axis=0)
for pulseTime in pulseTimes:
    ax.plot(np.mean(np.sum(np.abs(mean_M - data_noise_Ms[pulseTime]), axis=-1), axis=-1))


plt.savefig("layout2.pdf")





