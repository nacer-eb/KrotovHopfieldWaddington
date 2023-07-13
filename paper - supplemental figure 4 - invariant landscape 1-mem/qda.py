import sys
sys.path.append('../')

import numpy as np
import sys

from main_module.KrotovV2 import *


data_dir = "data_14/"

n, temp = 30, 700
tmax = 1500
if n == 30:
    tmax = 2500

fig, ax = plt.subplots(1, 2)
for n_mem in [1, 2]:
    saving_dir=data_dir+str(n_mem)+"m_l_n"+str(n)+"_T"+str(temp)+".npz"

    data = np.load(saving_dir)
    data_M = data['M']
    data_L = data['L']

    data_T = data["miniBatchs_images"][0]
    data_T_inv = np.linalg.pinv(data_T)

    data_coefs = data_M@data_T_inv

    print(np.shape(data_coefs))

    alpha = np.arange(0, 1, 0.01)
    beta = 1 - np.abs(alpha)

    ax[0].set_xlabel(r"$\alpha_1$"); ax[0].set_ylabel(r"$\alpha_4$")
    ax[1].set_xlabel("Training epoch"); ax[1].set_ylabel(r"Label coefficient")
    
    if n_mem == 1:
        ax[0].plot(data_coefs[:tmax, 0, 0], data_coefs[:tmax, 0, 1], "k--")
        ax[1].plot(data_L[:tmax, 0, 1], "b--"), ax[1].plot(data_L[:tmax, 0, 4], "r--"); ax[1].plot(data_L[:tmax, 0, -1], "k--")

    if n_mem == 2:
        ax[0].plot(alpha, beta, linewidth=8, alpha=0.1, color='grey')
        
        ax[0].plot(data_coefs[:tmax, 0, 0], data_coefs[:tmax, 0, 1], "r", linewidth=3, alpha=0.26)
        ax[0].plot(data_coefs[:tmax, 1, 0], data_coefs[:tmax, 1, 1], "b", linewidth=3, alpha=0.26)
        
        ax[1].plot(data_L[:tmax, 0, 1], "blue", linewidth=3, alpha=0.26)
        ax[1].plot(data_L[:tmax, 1, 1], "cyan", linewidth=3, alpha=0.26)

        ax[1].plot(data_L[:tmax, 0, 4], "red", linewidth=3, alpha=0.26)
        ax[1].plot(data_L[:tmax, 1, 4], "magenta", linewidth=3, alpha=0.26)
        
        ax[1].plot(data_L[:tmax, 0, -1], "k", linewidth=3, alpha=0.26)
        ax[1].plot(data_L[:tmax, 1, -1], "k", linewidth=3, alpha=0.26)


plt.subplots_adjust(top=0.9, bottom=0.15, left=0.059, right=0.983, hspace=0.2, wspace=0.2)
plt.show()


# The memories

fig = plt.figure()
ax = fig.subplots(2, 5)

for i, n_mem in enumerate([1, 2]):
    saving_dir=data_dir+str(n_mem)+"m_l_n"+str(n)+"_T"+str(temp)+".npz"

    data = np.load(saving_dir)
    data_M = data['M']
    
    for j in range(0, 5):
        ax[i, j].imshow(merge_data(data_M[(j+1)*tmax//5], n_mem, 1), cmap="bwr", vmin=-1, vmax=1)
        ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
        
plt.show()
    


exit()

#Second colorbar
from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color="black", linestyle="--"),
                Line2D([0], [0], color="r", linewidth=3, alpha=0.26),
                Line2D([0], [0], color="b", linewidth=3, alpha=0.26)]

fig = plt.figure()
fig.legend(custom_lines, [r'1-Memory System ($\langle \hat{l}_{A, B, \gamma} | L \rangle $)', r'2-Memory System ($\langle \hat{l}_{A, B, \gamma} | L_1 \rangle $)', r'2-Memory System ($\langle \hat{l}_{A, B, \gamma} | L_2 \rangle $)'], loc="center", ncols=4)
plt.show()

exit()
# First color bar
from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], linewidth=8, alpha=0.1, color='grey'),
                Line2D([0], [0], color="black", linestyle="--"),
                Line2D([0], [0], color="r", linewidth=3, alpha=0.26),
                Line2D([0], [0], color="b", linewidth=3, alpha=0.26)]

fig = plt.figure()
fig.legend(custom_lines, [r'$\alpha + \beta = 1$', '1-Memory System', r'2-Memory System ($| M_1 \rangle$, $| M_2 \rangle$)', r'2-Memory System ($| M_2 \rangle$)'], loc="center", ncols=4)
plt.show()
            
            
            

            
 

