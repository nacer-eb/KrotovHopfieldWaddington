import sys
sys.path.append('../')

import numpy as np

import matplotlib
font = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 18}
matplotlib.rc('font', **font)

import matplotlib.pyplot as plt
import matplotlib.animation as anim

from Figure_5_functions import *


data_dir = "data/"

selected_digits = [1, 4]
prefix = str(selected_digits) + "/"

dataset = "../defaults/miniBatchs_images_Fig_5.npy"
data_T = np.load(dataset)[0]
data_T_inv = np.linalg.pinv(data_T)

# Loading data
n_range = np.asarray([6, 10, 20, 30, 35, 37, 38, 40]) # You can add more n if you want, this is the strictly necessary for Fig 4
temp = 700
tmax = 5000

ell_range = np.asarray([-0.8, 0.8])
alpha_range = np.asarray([0.2, 0.5, 0.8])

# n, ell, alpha, epoch, # mems (2), coefs (2)
data_coefs = np.zeros((len(n_range), len(ell_range), len(alpha_range), tmax, 2, 2))
data_ells = np.zeros((len(n_range), len(ell_range), len(alpha_range), tmax, 2))
for i, n in enumerate(n_range):
    print(i)
    for j, ell in enumerate(ell_range):
        for k, alpha in enumerate(alpha_range):

            saving_dir=data_dir+prefix+"trained_net_n"+str(n)+"_T"+str(temp)+"_ell"+str(ell)+"_alpha"+str(alpha)+".npz"
            data = np.load(saving_dir)
            
            data_coefs[i, j, k] = data['M'] @ data_T_inv
            data_ells[i, j, k] = (data['L'][:, :, selected_digits[0]] - data['L'][:, :, selected_digits[1]])/2.0

           
        



fig = plt.figure(figsize=(15*1.5, 3.5*2*1.5 ))
axs = fig.subplot_mosaic(
    """
    AAA.BBB.CCC.DDD
    AAA.BBB.CCC.DDD
    AAA.BBB.CCC.DDD
    ...............
    EEE.FFF.GGG.HHH
    EEE.FFF.GGG.HHH
    EEE.FFF.GGG.HHH
    """
)



            
# Figure 5 object
f5 = Figure_5("run_[1, 4]_n15_T700_alpha0.8_l_00.5.npz", data_dir="../Figure 5/data/", C_data_dir="../Figure 5/C_Code_FPs/") 

# Fetch all nullcline axes
nullcline_axes = np.asarray([axs['A'], axs['B'], axs['C'], axs['D'],
                             axs['E'], axs['F'], axs['G'], axs['H']])
# Cosmetics
for i in range(0, len(nullcline_axes)):
    nullcline_axes[i].set_xlabel(r'$l_0$');
    if i != 0 and i != 4:
        nullcline_axes[i].set_yticks([]);
        continue
    nullcline_axes[i].set_ylabel(r'$\alpha$')


#points_1 = [[ [0]*len(ell_range) for i in range(len(alpha_range))] for j in range(len(n_range))]
#points_2 = [[ [0]*len(ell_range) for i in range(len(alpha_range))] for j in range(len(n_range))]

points_1 = [[[0 for k in range(len(alpha_range))] for j in range(len(ell_range))] for i in range(len(n_range))]
points_2 = [[[0 for k in range(len(alpha_range))] for j in range(len(ell_range))] for i in range(len(n_range))]


t=0
for i, n in enumerate(n_range):
    for j, ell in enumerate(ell_range):
        for k, alpha in enumerate(alpha_range):
            points_1[i][j][k], = nullcline_axes[i].plot(data_ells[i, j, k, t, 0], data_coefs[i, j, k, t, 0, 0], linestyle="", marker=matplotlib.markers.CARETUPBASE, ms=10, color="black")
            points_2[i][j][k], = nullcline_axes[i].plot(data_ells[i, j, k, t, 1], data_coefs[i, j, k, t, 1, 0], linestyle="", marker=matplotlib.markers.CARETDOWNBASE, ms=10, color="red")

props = dict(boxstyle='round', facecolor='whitesmoke', alpha=0.5)
fig.suptitle("epoch=0", fontsize=24, bbox=props)
    
for i, n in enumerate(n_range):
    alphas, betas = f5.plot_nullclines(nullcline_axes[i], n, plotDynamics=True)
    
#plt.savefig("tmp.png")
#exit()
def update(t):
    print(t)
    for i, n in enumerate(n_range):
        for j, ell in enumerate(ell_range):
            for k, alpha in enumerate(alpha_range):
                points_1[i][j][k].set_data(data_ells[i, j, k, t, 0], data_coefs[i, j, k, t, 0, 0])
                points_2[i][j][k].set_data(data_ells[i, j, k, t, 1], data_coefs[i, j, k, t, 1, 0])
    
    fig.suptitle("epoch="+str(t), fontsize=24, bbox=props)
    return *nullcline_axes,

ani = anim.FuncAnimation(fig, update, frames=tmax-1, interval=100, blit=False)
ani.save("nullcline_movie_n"+str(n)+".mov", writer="ffmpeg", fps=60)
    
#plt.savefig("Figure_5S_Nullclines.png")
