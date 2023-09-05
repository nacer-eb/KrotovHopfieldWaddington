import sys
sys.path.append('../')

import numpy as np
import matplotlib.pyplot as plt

import matplotlib
font = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 14}

matplotlib.rc('font', **font)

from main_module.KrotovV2_utils import *


data_dir = "data_1_10_200_179_full/"

temp = 800

fig = plt.figure(figsize=(11*2, 3*2), dpi=160)
axs = fig.subplot_mosaic("""

ABCDE.abcde
FGHIJ.fghij
KLMNO.klmno

""")

ax_all = np.asarray([ [ [axs['A'], axs['B'], axs['C'], axs['D'], axs['E']],
                       [axs['F'], axs['G'], axs['H'], axs['I'], axs['J']],
                       [axs['K'], axs['L'], axs['M'], axs['N'], axs['O']] ],
                     [[axs['a'], axs['b'], axs['c'], axs['d'], axs['e']],
                      [axs['f'], axs['g'], axs['h'], axs['i'], axs['j']],
                      [axs['k'], axs['l'], axs['m'], axs['n'], axs['o']]]])

for ax in ax_all.ravel():
    ax.set_xticks([]); ax.set_yticks([])

props = dict(boxstyle='round', facecolor='whitesmoke', alpha=0.5)
for i, d in enumerate([1, 7]):

    ax_all[i, 0, 2].text(0.5, 1.3, "Started near "+str(d), transform=ax_all[i, 0, 2].transAxes,
                         fontsize=14, verticalalignment='bottom', horizontalalignment='center', bbox=props)
    
    for j, n in enumerate([4, 23, 50]):

        # Cosmetics
        ax_all[i, j, 0].text(-0.2, 0.5, r"$n=$"+str(n), transform=ax_all[i, j, 0].transAxes, fontsize=14, verticalalignment='center', horizontalalignment='right', rotation=90, bbox=props)


        # Plotting

        # Fetch file 
        filename = "run_long_"+str(d)+"_n"+str(n)+"_T"+str(temp)+".npz"
        data = np.load(data_dir + filename)
        data_M = data["M"]
    
        tmax = len(data_M)

        for k in range(0, 5):
            t = (k)*(tmax//5)
            ax_all[i, j, k].imshow(merge_data(data_M[t, :25 , :], 5, 5), cmap="bwr", vmin=-1, vmax=1)

plt.savefig("Figure_5_supp.png")

