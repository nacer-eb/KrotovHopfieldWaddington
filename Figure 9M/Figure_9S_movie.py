import sys
sys.path.append('../')

import numpy as np

from main_module.KrotovV2 import *
from main_module.KrotovV2_utils import *

import matplotlib.animation as anim

data_dir = "../Figure 9S.N15_N40/data/"

isFirstRun = True

import matplotlib
font = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 44}
matplotlib.rc('font', **font)

selected_digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]#
prefix = str(selected_digits)+"/" # I used main,and momentum #"main"#

tmax = 10000
N_mem = 100

n = 15 # 15 or 40

temp_range = np.arange(650, 1000, 10)
Nt = len(temp_range)

data_Ms = np.zeros((Nt, N_mem, 784))
data_Ls = np.zeros((Nt, N_mem, 10))

dataset = "../defaults/miniBatchs_images.npy"
data_T = np.load(dataset)[0]
data_T_inv = np.linalg.pinv(data_T)



if isFirstRun:
    for h, temp in enumerate(temp_range):
        print(n, temp)
        
        saving_dir = data_dir+prefix+"trained_net_end_n"+str(n)+"_T"+str(temp)+".npz"
        
        data = np.load(saving_dir)
        data_Ms[h] = data['M'][-1]
        data_Ls[h] = data['L'][-1]
            
    data_coefs = (data_Ms@data_T_inv).reshape(Nt, N_mem, 10, 20)
    
    np.save(data_dir + prefix + "data_Ms_T.npy", data_Ms)
    np.save(data_dir + prefix + "data_Ls_T.npy", data_Ls)
    np.save(data_dir + prefix + "data_Coefs_T.npy", data_coefs)

data_Ms = np.load(data_dir + prefix + "data_Ms_T.npy")
data_Ls = np.load(data_dir + prefix + "data_Ls_T.npy")
data_coefs = np.load(data_dir + prefix + "data_Coefs_T.npy")

print(np.shape(data_Ls))

fig = plt.figure(figsize=(16, 9))

axs = fig.subplot_mosaic("""
AAAAAAAAAA.00000
AAAAAAAAAA.11111
AAAAAAAAAA.22222
AAAAAAAAAA.33333
AAAAAAAAAA.44444
AAAAAAAAAA.55555
AAAAAAAAAA.66666
AAAAAAAAAA.77777
AAAAAAAAAA.88888
AAAAAAAAAA.99999
""")



axs["0"].set_title("Labels", fontsize=35)

ax_im = axs['A']
ax_l = [0]*10

for d in range(10):
    ax_l[d] = axs[str(d)]

    ax_l[d].set_xlim(-1.05, 1.05)
    ax_l[d].set_yticks([])
    if d < 9:
        ax_l[d].set_xticks([])
    

t_i = 0

i_sort = np.argsort(np.argmax(data_Ls[t_i], -1))

im = ax_im.imshow(merge_data(data_Ms[t_i, i_sort], 10, 10), cmap="bwr", vmin=-1, vmax=1)
ax_im.set_xticks([]); ax_im.set_yticks([])
ax_im.set_title(r"$T_r=$"+str(temp_range[t_i])+", n="+str(n))

cmap_tab10 = matplotlib.cm.tab10
norm = matplotlib.colors.Normalize(vmin=0, vmax=10)

p_l = [0]*10
for d in range(0, 10):
    p_l[d], =  ax_l[d].plot(data_Ls[t_i, :, d], range(100), linestyle="None", marker=".", markersize=5, color=cmap_tab10(norm(d)))



def update(t_i):
    print(t_i)
    
    i_sort = np.argsort(np.argmax(data_Ls[t_i], -1))
    im.set_data(merge_data(data_Ms[t_i, i_sort], 10, 10))
    ax_im.set_title(r"$T_r$="+'{0:.2f}'.format(temp_range[t_i]/784)+", n="+str(n))
    for d in range(0, 10):
        p_l[d].set_data(data_Ls[t_i, :, d], range(100))

    return im, *p_l, ax_im, *ax_l


ani = anim.FuncAnimation(fig, update, frames=len(temp_range)-1, interval=100, blit=False)
ani.save("Figure_9S_Movie_n"+str(n)+".mov", writer="ffmpeg")
#plt.show()

exit()

