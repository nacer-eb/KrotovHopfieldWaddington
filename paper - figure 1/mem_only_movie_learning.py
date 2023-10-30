import sys
sys.path.append('../')

import numpy as np

import matplotlib
fontsize = 44
font = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : fontsize}
matplotlib.rc('font', **font)

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation as anim

from main_module.KrotovV2 import *
from main_module.KrotovV2_utils import *



n, temp = 30, 670
data_dir = "data/"
subdir = "main/"
saving_dir = data_dir+subdir+"trained_net_n"+str(n)+"_T"+str(temp)+".npz"

# Loading data - will improve dir struct soon..
data_M = np.load(saving_dir)['M']
data_L = np.load(saving_dir)['L']
data_T = np.load(data_dir+"miniBatchs_images.npy")[0]

print(np.shape(data_M))



fig = plt.figure(figsize=(10, 10))

axs = fig.subplot_mosaic("""

MMMMMMMMMM
MMMMMMMMMM
MMMMMMMMMM
MMMMMMMMMM
MMMMMMMMMM
MMMMMMMMMM
MMMMMMMMMM
MMMMMMMMMM
MMMMMMMMMM
MMMMMMMMMM

""")



order_sort = np.argsort(np.argmax(data_L[-1], axis=-1))

t=0
im_M = axs['M'].imshow(merge_data(data_M[t, order_sort, :], 10, 10), cmap="bwr", vmin=-1, vmax=1)
axs['M'].set_xticks([]); axs['M'].set_yticks([])


props = dict(boxstyle='round', facecolor='whitesmoke', alpha=0.5)
time_indicator = axs['M'].set_title(r"$t=$"+str(t), fontsize=24, verticalalignment='bottom', ha='center', bbox=props, pad=20)

def update(t):
    # Labels
    print(t)
    
    im_M.set_data(merge_data(data_M[t, order_sort, :], 10, 10))
    
    time_indicator.set_text(r"$t=$"+str(t))
    
    return im_M, time_indicator

ani = anim.FuncAnimation(fig, update, frames=len(data_M), interval=100, blit=False)
ani.save("mem_only_learning_move_n"+str(n)+".mov", writer="ffmpeg", fps=60)
#plt.show()



