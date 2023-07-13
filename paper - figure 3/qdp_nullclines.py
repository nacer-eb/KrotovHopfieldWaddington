import sys
sys.path.append('../')

import glob

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


fig, ax = plt.subplots(2, 4, sharex=True, sharey=True)

ax[0, 0].set_ylabel(r"$\alpha$"); ax[1, 0].set_ylabel(r"$\alpha$")
ax[-1, 0].set_xlabel(r"$l_0$"); ax[-1, 1].set_xlabel(r"$l_0$"); ax[-1, 2].set_xlabel(r"$l_0$"); ax[-1, 3].set_xlabel(r"$l_0$");

for temp in [700]:
    for n_i, n in enumerate([15, 30]):

        filenames = glob.glob(data_dir+"*n"+str(n)+"_T"+str(temp)+"*.npz")
        filename = filenames[0]
        data_T = np.load(filename)['miniBatchs_images'][0]
        data_T_inv = np.linalg.pinv(data_T)
        """
        alphas = [0]*len(filenames)
        l_0s = [0]*len(filenames)
        for i, filename in enumerate(filenames):
            data = np.load(filename)
            data_M = data['M']
            data_L = data['L']

            alphas[i] = (data_M@data_T_inv)[:, :, 0]

            l_0s[i] = 0.5*(data_L[:, :, 1] - data_L[:, :, 4])
            print(i)
        
        np.save(data_dir+"n"+str(n)+"_T"+str(temp)+"_alphas.npy", alphas)
        np.save(data_dir+"n"+str(n)+"_T"+str(temp)+"_l_0s.npy", l_0s)
        exit()
        """
        alphas = np.load(data_dir+"n"+str(n)+"_T"+str(temp)+"_alphas.npy")
        l_0s = np.load(data_dir+"n"+str(n)+"_T"+str(temp)+"_l_0s.npy")
        
        
        GNC = GatherNullClines(753, 494, 719, n, temp/(2.0**(1.0/n)), +1)
        
        alpha_range = np.linspace(0, 1, 1000); l_range = np.linspace(-0.7, 0.7, 1000)
        l_0_mesh, alpha_mesh = np.meshgrid(l_range, alpha_range)

        alpha_nullcline = GNC.alpha_nullcline(alpha_mesh, l_0_mesh)
        l_nullcline = GNC.l_0_nullcline(alpha_mesh, l_0_mesh)
        
        
        p0 = [0]*len(alphas)
        t0 = [0]*len(alphas)
        
        p1 = [0]*len(alphas)
        t1 = [0]*len(alphas)

        t_range = [0, 1200, 2500, 3499]
        for t_i, t in enumerate(t_range):
            i = n_i
            j = t_i
            t_min = t_range[t_i-1]

                       
            ax[i, j].contour(l_0_mesh, alpha_mesh, alpha_nullcline, [0], colors="purple", linewidths=7, alpha=0.5)
            ax[i, j].contour(l_0_mesh, alpha_mesh, l_nullcline, [0], colors="orange", linewidths=7, alpha=0.5)
            
            for k in np.arange(0, len(alphas), 1):
                p0[i], = ax[i, j].plot(l_0s[k, t, 0], alphas[k, t, 0], linestyle="", marker=matplotlib.markers.CARETDOWN, color="red", ms=7)
                t0[i], = ax[i, j].plot(l_0s[k, t_min:t, 0], alphas[k, t_min:t, 0], linestyle="-", marker="", color="red", lw=1.5)
                
                p1[i], = ax[i, j].plot(l_0s[k, t, 1], alphas[k, t, 1], linestyle="", marker=matplotlib.markers.CARETUP, color="black", ms=7)
                t1[i], = ax[i, j].plot(l_0s[k, t_min:t, 1], alphas[k, t_min:t, 1], linestyle="-", marker="", color="black", lw=1.5)


            props = dict(boxstyle='round', facecolor='whitesmoke', alpha=0.5)
            text = ax[i, j].text(0.77, 0.97, r"$t=$"+str(t), transform=ax[i, j].transAxes, fontsize=18, verticalalignment='top', bbox=props)

    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color="purple", linewidth=7, alpha=0.5),
                    Line2D([0], [0], color="orange", linewidth=7, alpha=0.5),
                    (Line2D([0], [0], color="black", marker=matplotlib.markers.CARETUP, linestyle="", ms=10),
                     Line2D([0], [0], color="red", marker=matplotlib.markers.CARETDOWN, linestyle="",  ms=10)),
                    Line2D([0], [0], color="black", marker="", linestyle="-",  lw=2)]
    
    
    fig.legend(custom_lines, ["Memory nullcline", "Label nullcline", "Memory pair", "Trajectory/trail"], loc='upper center', ncol=4)
    plt.subplots_adjust(top=0.89, bottom=0.075, left=0.04, right=0.99, hspace=0.175, wspace=0.04)
    plt.figtext(2*(0.99-0.04)/4+0.0395, 0.91, r"$(n, T) = (15, 700)$", va="center", ha="center", size=18)
    plt.figtext(2*(0.99-0.04)/4+0.040, 0.47, r"$(n, T) = (30, 700)$", va="center", ha="center", size=18)
    
    plt.show()
        
        
        

exit()
