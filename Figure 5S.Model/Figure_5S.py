import sys
sys.path.append('../')

import numpy as np
import matplotlib.pyplot as plt

from main_module.KrotovV2 import *
from main_module.KrotovV2_utils import *

import matplotlib
fontsize=50
font = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : fontsize}
matplotlib.rc('font', **font)

fig = plt.figure(figsize=(44, 37))

axs = fig.subplot_mosaic("""


AAAAAAAAAA.BBBBBBBBBB.CCCCCCCCCC.DDDDDDDDDD
AAAAAAAAAA.BBBBBBBBBB.CCCCCCCCCC.DDDDDDDDDD
AAAAAAAAAA.BBBBBBBBBB.CCCCCCCCCC.DDDDDDDDDD
AAAAAAAAAA.BBBBBBBBBB.CCCCCCCCCC.DDDDDDDDDD
AAAAAAAAAA.BBBBBBBBBB.CCCCCCCCCC.DDDDDDDDDD
...........................................
aaaaaaaaaa.bbbbbbbbbb.cccccccccc.dddddddddd
aaaaaaaaaa.bbbbbbbbbb.cccccccccc.dddddddddd
aaaaaaaaaa.bbbbbbbbbb.cccccccccc.dddddddddd
aaaaaaaaaa.bbbbbbbbbb.cccccccccc.dddddddddd
aaaaaaaaaa.bbbbbbbbbb.cccccccccc.dddddddddd
aaaaaaaaaa.bbbbbbbbbb.cccccccccc.dddddddddd
aaaaaaaaaa.bbbbbbbbbb.cccccccccc.dddddddddd
aaaaaaaaaa.bbbbbbbbbb.cccccccccc.dddddddddd
aaaaaaaaaa.bbbbbbbbbb.cccccccccc.dddddddddd
aaaaaaaaaa.bbbbbbbbbb.cccccccccc.dddddddddd
...........................................
...........................................
11111111111111111111...22222222222222222222
11111111111111111111...22222222222222222222
11111111111111111111...22222222222222222222
11111111111111111111...22222222222222222222
11111111111111111111...22222222222222222222
11111111111111111111...22222222222222222222
11111111111111111111...22222222222222222222
11111111111111111111...22222222222222222222
11111111111111111111...22222222222222222222
11111111111111111111...22222222222222222222
11111111111111111111...22222222222222222222
11111111111111111111...22222222222222222222
11111111111111111111...22222222222222222222
11111111111111111111...22222222222222222222
11111111111111111111...22222222222222222222
11111111111111111111...22222222222222222222
11111111111111111111...22222222222222222222
11111111111111111111...22222222222222222222
11111111111111111111...22222222222222222222
11111111111111111111...22222222222222222222

""")

data_dir = "data/"
n, temp = 3, 700

for char in ['a', 'b', 'c', 'd', 'A', 'B', 'C', 'D']:
    axs[char].set_xticks([])
    axs[char].set_yticks([])


props = dict(boxstyle='round', facecolor='whitesmoke', alpha=0.5)


t_range = np.asarray([100, 200, 300, 330])

if n == 30:
    t_range = np.asarray([100, 200, 400, 550])

saving_dir=data_dir+"trained_net_K1_n"+str(n)+"_T"+str(temp)+".npz"
data_1 = np.load(saving_dir)
data_M_1 = data_1['M']
data_L_1 = data_1['L']
data_T = data_1['miniBatchs_images'][0]
data_T_inv = np.linalg.pinv(data_T)
coefs_1 = data_M_1 @ data_T_inv

saving_dir=data_dir+"trained_net_K2_n"+str(n)+"_T"+str(temp)+".npz"
data_2 = np.load(saving_dir)
data_M_2 = data_2['M']
data_L_2 = data_2['L']
coefs_2 = data_M_2 @ data_T_inv


for i, char in enumerate(['a', 'b', 'c', 'd']):
    axs[char].imshow(data_M_1[t_range[i]].reshape(28, 28), cmap="bwr", vmin=-1, vmax=1)


for i, char in enumerate(['A', 'B', 'C', 'D']):
    axs[char].set_title("Epoch "+str(t_range[i]), bbox=props, fontsize=fontsize, verticalalignment='bottom', horizontalalignment='center', pad=60)
    axs[char].imshow(merge_data(data_M_2[t_range[i]], 2, 1), cmap="bwr", vmin=-1, vmax=1)


axs['1'].set_xlabel('Epoch'); axs['2'].set_xlabel('Epoch')
axs['1'].set_ylabel(r'$\alpha$'); axs['2'].set_ylabel('Label')


tab10_cmap = matplotlib.cm.tab10
tab10_norm = matplotlib.colors.Normalize(vmin=0, vmax=10)

digit_classes = [1, 4]

t = np.arange(0, t_range[-1], 1)
for d_i in range(0, 2):
    axs['1'].plot(t, coefs_1[:t_range[-1], 0, d_i], "-", color=tab10_cmap(tab10_norm(digit_classes[d_i])), alpha=0.5, lw=15)

    
    axs['1'].plot(t, coefs_2[:t_range[-1], 0, d_i], ".-", color=tab10_cmap(tab10_norm(digit_classes[d_i])), alpha=1, lw=7)
    axs['1'].plot(t, coefs_2[:t_range[-1], 1, d_i], "--", color=tab10_cmap(tab10_norm(digit_classes[d_i])), alpha=1, lw=7)

    


for d_i in range(0, 2):
    axs['2'].plot(t, data_L_1[:t_range[-1], 0, digit_classes[d_i]], "-", color=tab10_cmap(tab10_norm(digit_classes[d_i])), alpha=0.5, lw=15)

    axs['2'].plot(t, data_L_2[:t_range[-1], 0, digit_classes[d_i]], ".-", color=tab10_cmap(tab10_norm(digit_classes[d_i])), alpha=1, lw=7)
    axs['2'].plot(t, data_L_2[:t_range[-1], 1, digit_classes[d_i]], "--", color=tab10_cmap(tab10_norm(digit_classes[d_i])), alpha=1, lw=7)
    
    axs['2'].plot(t, data_L_1[:t_range[-1], 0, -1], "-", color=tab10_cmap(tab10_norm(3)), alpha=0.5, lw=15)

    axs['2'].plot(t, data_L_2[:t_range[-1], 0, -1], "-", color=tab10_cmap(tab10_norm(3)), alpha=0.5, lw=15)
    axs['2'].plot(t, data_L_2[:t_range[-1], 1, -1], "-", color=tab10_cmap(tab10_norm(3)), alpha=0.5, lw=15)
    
    
plt.savefig("Figure_5S_tmp_n"+str(n)+".png")
