import sys
sys.path.append('../')

import numpy as np

from main_module.KrotovV2 import *
from main_module.KrotovV2_utils import *

import matplotlib
fontsize=35
font = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : fontsize}
matplotlib.rc('font', **font)

dataset = "../defaults/miniBatchs_images_Fig_6.npy"
data_T = np.load(dataset)[0]
data_T_inv = np.linalg.pinv(data_T)

data_dir = "data/"

digit_classes=[1, 4]
prefix = str(digit_classes)+"/"

temp = 700.0

n_range = np.arange(0, 60, 0.1)



alphas = np.zeros((len(n_range), 2, 2))
ells = np.zeros((len(n_range), 2))

isFirstRun = False
if isFirstRun:
    for i, n in enumerate(n_range):
        for j, p in enumerate([(-1, 0), (1, 1)]):
            saving_dir=data_dir+prefix+"trained_net_n"+'{0:.2f}'.format(n)+"_T"+str(temp)+"_ic"+str(j)+".npz"
            
            data = np.load(saving_dir)
            alphas[i, j, :] = data['M'][-1]@data_T_inv
            ells[i, j] = data['L'][-1, 0, digit_classes[0]]
        print(n)
        
    np.save(data_dir + prefix + "alphas.npy", alphas)
    np.save(data_dir + prefix + "ells.npy", ells)

alphas = np.load(data_dir + prefix + "alphas.npy")
ells = np.load(data_dir + prefix + "ells.npy")


fig, ax = plt.subplots(2, 1, figsize=(9, 16))



# theory
C_data_dir = "../Figure 6/C_Code_FPs/"
data_theory = np.loadtxt(C_data_dir+"save0.dat", delimiter=",", skiprows=1)
p_mask = (np.abs(data_theory[:, -2]) > 0.001) * (np.abs(data_theory[:, -3]) > 0.001 ) * (data_theory[:, 1] <= 7) + (np.abs(data_theory[:, -2]) > 0.03) * (np.abs(data_theory[:, -3]) > 0.03 ) * (data_theory[:, 1] > 7)
FP_data_theory = data_theory[p_mask, :]

ns_theory = FP_data_theory[:-34000, 1]

alphas_theory = np.zeros((len(ns_theory), 3))
l_0s_theory = np.zeros((len(ns_theory), 3))
for i, n in enumerate(ns_theory):
    print(n)
    n_mask = FP_data_theory[:, 1]==n
    index_sort = np.argsort(FP_data_theory[n_mask, -1])
    
    alphas_theory[i] = FP_data_theory[n_mask, -3][index_sort][[0, np.sum(n_mask)//2, -1]]
    l_0s_theory[i] = FP_data_theory[n_mask, -1][index_sort][[0, np.sum(n_mask)//2, -1]]


# Creating the colormap for n
norm = plt.Normalize(np.min(ns_theory), np.max(ns_theory)) # Norm map for n-power
c1 = np.asarray([191/256.0, 127/256.0, 191/256.0, 0.5]) # purple
c2 = np.asarray([255/256.0, 209/256.0, 127/256.0, 0.5]) # golden yellow

k = np.linspace(0, 1, 256)
vals = np.zeros((256, 4))
for i in range(0, 256):
    vals[i] = c1*(1 - k[i]) + c2*k[i]
cmap = matplotlib.colors.ListedColormap(vals)
    
for i in range(3):
    ax[0].scatter(alphas_theory[:, i], ns_theory, color=c2)#c=ns_theory, cmap=cmap, norm=norm, s=25)
    ax[1].scatter(l_0s_theory[:, i], ns_theory, color=c2)#c=ns_theory, cmap=cmap, norm=norm, s=25)



# simulation
ax[0].scatter(alphas[:, 0, 0], n_range, c="k", s=3)
ax[0].scatter(alphas[:, 1, 0], n_range, c="k", s=3)

ax[1].scatter(ells[:, 0], n_range, c="k", s=3)
ax[1].scatter(ells[:, 1], n_range, c="k", s=3)


ax[0].set_ylabel(r"$n$")
ax[1].set_ylabel(r"$n$")

ax[0].set_xlabel(r"$\alpha_1$")
ax[1].set_xlabel(r"$\ell$")

plt.subplots_adjust(hspace=0.5)
plt.savefig("Figure_6S_tmp.png")        
