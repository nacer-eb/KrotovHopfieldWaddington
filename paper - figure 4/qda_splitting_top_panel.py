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

from nullcline_gather.GatherNullClines import *


data_dir = "data_2_2_2/"



selected_digits=[1, 4]

for temp in [700]:
    for n in [15]:

        fig = plt.figure()
        axs = fig.subplot_mosaic(
            """
            EF.GH.IJ.KL
            AA.BB.CC.DD
            AA.BB.CC.DD
            """
        )

        ax_m = np.asarray([[axs['E'], axs['F']],
                           [axs['G'], axs['H']],
                           [axs['I'], axs['J']],
                           [axs['K'], axs['L']]])
        
        ax = np.asarray([axs['A'], axs['B'], axs['C'], axs['D']])
        
        alpha_range = np.linspace(0, 1, 1000); l_range = np.linspace(-1, 1, 1000)
        l_0_mesh, alpha_mesh = np.meshgrid(l_range, alpha_range)
        
        GNC = GatherNullClines(753, 494, 719, n, temp/(2.0**(1.0/n)), +1)
        alpha_nullcline = GNC.alpha_nullcline(alpha_mesh, l_0_mesh)
        l_nullcline = GNC.l_0_nullcline(alpha_mesh, l_0_mesh)

        dt_l_0, dt_alpha = GNC.get_dt(alpha_mesh, l_0_mesh)
                
        
        for i in range(0, 4):

            if i < 2:
                ax[i].streamplot(l_0_mesh, alpha_mesh, dt_l_0, dt_alpha, color="grey", density=1)
            
            ax[i].contour(l_0_mesh, alpha_mesh, alpha_nullcline, [0], colors="purple", linewidths=4, alpha=0.5)
            ax[i].contour(l_0_mesh, alpha_mesh, l_nullcline, [0], colors="orange", linewidths=4, alpha=0.5)

            if i > 0:
                ax[i].set_yticks([])

            ax[i].set_xlabel(r"$l_0$")
        ax[0].set_ylabel(r"$\alpha$")
                
        for (alpha, l_0) in [(0.8, 0.5)]: #
            saving_dir = data_dir + "run_" + str(selected_digits) \
                + "_n" + str(n) \
                + "_T" + str(temp) \
                + "_alpha" + str(alpha) \
                + "_l_0" + str(l_0) \
                + ".npz"
            
            
            
            data = np.load(saving_dir)
            data_T = data['miniBatchs_images'][0]
            
            data_T_inv = np.linalg.pinv(data_T)
            
            data_M = data['M']
            data_L = data['L']

            l_0 = 0.5*(data_L[:, :, selected_digits[0]] - data_L[:, :, selected_digits[1]])
            
            coefs = data_M @ data_T_inv

            t_range = np.asarray([0, 1000, 2000, 3000])
            for i, t in enumerate(t_range):
                ax[i].plot(l_0[t_range[i-1]:t, 0], coefs[t_range[i-1]:t, 0, 0], color="red", lw=2)
                ax[i].scatter(l_0[t, 0], coefs[t, 0, 0], color="red", marker=matplotlib.markers.CARETUP, s=100)
                
                ax[i].plot(l_0[t_range[i-1]:t, 1], coefs[t_range[i-1]:t, 1, 0], color="black", lw=2)
                ax[i].scatter(l_0[t, 1], coefs[t, 1, 0], color="black", marker=matplotlib.markers.CARETDOWN, s=100)

                ax_m[i, 0].set_xticks([]); ax_m[i, 1].set_xticks([]); ax_m[i, 0].set_yticks([]); ax_m[i, 1].set_yticks([])
                ax_m[i, 0].imshow(data_M[t, 0, :].reshape(28, 28), cmap="bwr"); ax_m[i, 1].imshow(data_M[t, 1, :].reshape(28, 28), cmap="bwr")

                props = dict(boxstyle='round', facecolor='whitesmoke', alpha=0.5)

                ax[i].text(0.5, 1.65, r"$t=$"+str(t), transform=ax[i].transAxes, fontsize=18, verticalalignment='top', ha='center', bbox=props)

                
        from matplotlib.lines import Line2D
        custom_lines = [Line2D([0], [0], color="purple", linewidth=7, alpha=0.5),
                        Line2D([0], [0], color="orange", linewidth=7, alpha=0.5),
                        (Line2D([0], [0], color="black", marker=matplotlib.markers.CARETUP, linestyle="", ms=10),
                         Line2D([0], [0], color="red", marker=matplotlib.markers.CARETDOWN, linestyle="",  ms=10)),
                        Line2D([0], [0], color="black", marker="", linestyle="-",  lw=2)]
        
    
        fig.legend(custom_lines, ["Memory nullcline", "Label nullcline", "Memory pair", "Trajectory/trail"], loc='upper center', ncol=4)
        plt.subplots_adjust(top=0.7, bottom=0.23, left=0.055, right=0.95, hspace=0.05, wspace=0.05)
        plt.show()
            
