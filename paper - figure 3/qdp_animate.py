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



for temp in [670]:
    for n in [30]:        
        filenames = glob.glob(data_dir+"*n"+str(n)+"_T"+str(temp)+"*.npz")
        filename = filenames[0]
        data_T = np.load(filename)['miniBatchs_images'][0]
        data_T_inv = np.linalg.pinv(data_T)

        alphas = np.load(data_dir+"n"+str(n)+"_T"+str(temp)+"_alphas.npy")
        l_0s = np.load(data_dir+"n"+str(n)+"_T"+str(temp)+"_l_0s.npy")

        
        fig, ax = plt.subplots(1, 1)

        GNC = GatherNullClines(753, 494, 719, n, temp/(2.0**(1.0/n)), +1)
        
        alpha_range = np.linspace(0, 1, 500); l_range = np.linspace(-0.25, 0.25, 500)
        l_0_mesh, alpha_mesh = np.meshgrid(l_range, alpha_range)

        alpha_nullcline = GNC.alpha_nullcline(alpha_mesh, l_0_mesh)
        l_nullcline = GNC.l_0_nullcline(alpha_mesh, l_0_mesh)

        p0 = [0]*len(alphas)
        t0 = [0]*len(alphas)
        
        p1 = [0]*len(alphas)
        t1 = [0]*len(alphas)

        ax.contour(l_0_mesh, alpha_mesh, alpha_nullcline, [0], colors="purple", linewidths=7, alpha=0.5)
        ax.contour(l_0_mesh, alpha_mesh, l_nullcline, [0], colors="orange", linewidths=7, alpha=0.5)

        for k in np.arange(0, len(alphas), 1):
            p0[k], = ax.plot(l_0s[k, 0, 0], alphas[k, 0, 0], linestyle="", marker=matplotlib.markers.CARETDOWN, color="red", ms=7)
            t0[k], = ax.plot(l_0s[k, 0, 0], alphas[k, 0, 0], linestyle="-", marker="", color="red", lw=1.5)
            
            p1[k], = ax.plot(l_0s[k, 0, 1], alphas[k, 0, 1], linestyle="", marker=matplotlib.markers.CARETUP, color="black", ms=7)
            t1[k], = ax.plot(l_0s[k, 0, 1], alphas[k, 0, 1], linestyle="-", marker="", color="black", lw=1.5)

        props = dict(boxstyle='round', facecolor='whitesmoke', alpha=0.5)
        text = ax.text(0.83, 0.97, r"$t=$"+str(0), transform=ax.transAxes, fontsize=18, verticalalignment='top', bbox=props)
            
        def update(t_):
            t = t_*20
            t_min = np.maximum(t-1000, 0)
            
            for i in range(len(alphas)):
                p0[i].set_data(l_0s[i, t, 0], alphas[i, t, 0])
                t0[i].set_data(l_0s[i, t_min:t, 0], alphas[i, t_min:t, 0])
                
                p1[i].set_data(l_0s[i, t, 1], alphas[i, t, 1])
                t1[i].set_data(l_0s[i, t_min:t, 1], alphas[i, t_min:t, 1])

            print(t)
            text.set_text(t)
            return *p0, *p1, text, *t0, *t1, 

        
        
        ani = anim.FuncAnimation(fig, update, frames=3500//20, interval=10, blit=True, repeat=False)
        #ani.save(data_dir+"trajectories_n"+str(n)+"_T"+str(temp)+".mp4", writer="ffmpeg")
        plt.show()
        
