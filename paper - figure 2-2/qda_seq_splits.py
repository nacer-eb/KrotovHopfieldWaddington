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

from main_module.KrotovV2_utils import *

data_dir = "data_100_10_200/"

selected_digits = [1, 7, 9]
for temp in [800]:
    
    fig = plt.figure()
    axs = fig.subplot_mosaic(
        """        
        CDE.AAAA
        CDE.AAAA
        CDE.AAAA
        ........
        FGH.BBBB
        FGH.BBBB
        FGH.BBBB
        """
    )

    ax = np.asarray([axs['A'], axs['B']])

    ax_m = np.asarray([[axs['C'], axs['D'], axs['E']], [axs['F'], axs['G'], axs['H']]])
    
    indices = np.asarray([[1, 3, 4], [1, 2, 3]], dtype=int)
    t_samples = np.asarray([[50, 200, 400], [450, 1200, 2600]], dtype=int)
    
    for n_i, n in enumerate([3, 30]):
        saving_dir=data_dir+"run_"+str(selected_digits)+"_n"+str(n)+"_T"+str(temp)+".npz"
        
        data = np.load(saving_dir)

        data_T = data['miniBatchs_images'][0]
        data_T_inv = np.linalg.pinv(data_T)

        data_L = data['L']
        data_L_key = np.argmax(data_L[-1], axis=-1)
        
        data_M = data['M']

        # Plotting samples here
       
        for t_i in range(0, 3):
            props = dict(boxstyle='round', facecolor='whitesmoke', alpha=0.5)
            ax_m[n_i, t_i].set_xticks([]); ax_m[n_i, t_i].set_yticks([]);
            ax_m[n_i, t_i].text(0.5, 1.11, r"$t=$"+str(t_samples[n_i, t_i]), transform=ax_m[n_i, t_i].transAxes, fontsize=18, verticalalignment='top', ha='center', bbox=props)
            
            ax_m[n_i, t_i].imshow(merge_data( data_M[t_samples[n_i, t_i], indices[n_i], :], 1, 3), cmap="bwr", vmin=-1, vmax=1)
            
        
        coefs = np.sum((data_M@data_T_inv).reshape(len(data_M), 100, 3, 20), axis=-1)
        

        t = 0
        p = [0]*100

        #ax.set_xlim(0, 1); ax.set_ylim(0, 1);# ax.set_zlim(0, 1)
        
        tab_10 = matplotlib.cm.tab10
        normalizer = matplotlib.colors.Normalize(vmin=0, vmax=9)

        tmin = [50, 450]
        ax[n_i].plot(coefs[tmin[n_i], :, 0], coefs[tmin[n_i], :, 1], linestyle="", lw=1, marker="*", ms=10, color="gold")
        for i in range(100):
            p[i], = ax[n_i].plot(coefs[tmin[n_i]:1900, i, 0], coefs[tmin[n_i]:1900, i, 1], linestyle="-", lw=1, color=tab_10(normalizer(data_L_key[i])))
        ax[n_i].set_xlabel(r"$\alpha_1$"); ax[n_i].set_ylabel(r"$\alpha_7$");

        
        #props = dict(boxstyle='round', facecolor='whitesmoke', alpha=0.5)
        #ax[n_i].text(0.5, 1.05, r"$n=$"+str(n), transform=ax[n_i].transAxes, fontsize=18, verticalalignment='top', ha='center', bbox=props)

        from matplotlib.lines import Line2D
        custom_lines = [Line2D([0], [0], linestyle="", lw=1, marker="*", ms=10, color="gold"),
                        Line2D([0], [0], linestyle="-", lw=3, color=tab_10(normalizer(9)))]
        
        fig.legend(custom_lines, ['Starting point', 'Memory trajectory'], loc='upper center', ncol=2)
        plt.subplots_adjust(top=0.85, bottom=0.09, left=0.235, right=0.7, hspace=0.02, wspace=0.05)

        cbar_ax = fig.add_axes([0.73, 0.09, 0.02, 0.85-0.09])
        cb = matplotlib.colorbar.ColorbarBase(cbar_ax, cmap=matplotlib.cm.tab10, norm=matplotlib.colors.Normalize(vmin=0, vmax=9))
        cb.ax.set_ylabel("Digit class")
        
    plt.show()
        
        
