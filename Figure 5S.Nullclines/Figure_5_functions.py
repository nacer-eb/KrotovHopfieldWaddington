import sys
sys.path.append('../')

import numpy as np

import matplotlib
font = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 18}
matplotlib.rc('font', **font)

import matplotlib.pyplot as plt

from nullcline_gather.GatherNullClines import *

dataset = "../defaults/miniBatchs_images_Fig_5.npy"

class Figure_5:
    def __init__(self, saving_dir, data_dir = "data/", C_data_dir = "C_Code_FPs/", selected_digits=[1, 4]):

        data = np.loadtxt(C_data_dir+"save0.dat", delimiter=",", skiprows=1)
        p_mask = (np.abs(data[:, -2]) > 0.001) * (np.abs(data[:, -3]) > 0.001 ) * (data[:, 1] <= 7) + (np.abs(data[:, -2]) > 0.03) * (np.abs(data[:, -3]) > 0.03 ) * (data[:, 1] > 7)
        self.FP_data = data[p_mask, :]


        data = np.load(data_dir+saving_dir)

        data_T = np.load(dataset)[0]
        
        data_T_inv = np.linalg.pinv(data_T)
        data_L = data['L']
        data_M = data['M']
        
        self.l_0 = 0.5*(data_L[:, :, selected_digits[0]] - data_L[:, :, selected_digits[1]])
        self.coefs = data_M @ data_T_inv
            
        self.A, self.B = data_T[0], data_T[1]

        alpha_range = np.linspace(0, 1, 1000); l_range = np.linspace(-1, 1, 1000)
        self.l_0_mesh, self.alpha_mesh = np.meshgrid(l_range, alpha_range)

        self.props = dict(boxstyle='round', facecolor='whitesmoke', alpha=0.5)


        
    def plot_nullclines(self, ax, n, temp=700, t_0=0, t=0, plotDynamics=False, density=1):

        GNC = GatherNullClines(self.A@self.A, self.A@self.B, self.B@self.B, n, temp/(2.0**(1.0/n)), +1)  
        d_alpha_dt, norm_condition = GNC.calc_d_alpha_dt(self.alpha_mesh, self.l_0_mesh)
        d_ell_dt, d_ell_dt_p_sat, d_ell_dt_m_sat = GNC.calc_d_ell_dt(self.alpha_mesh, self.l_0_mesh)

        ax.contour(self.l_0_mesh, self.alpha_mesh, d_alpha_dt, [0], colors="purple", linewidths=4, alpha=0.5)
        ax.contour(self.l_0_mesh, self.alpha_mesh, d_ell_dt, [0], colors="orange", linewidths=4, alpha=0.5)
          
        textbox = ax.text(0.07, 0.94, r"$n=$"+str(n), transform=ax.transAxes, fontsize=22, verticalalignment='top', horizontalalignment='left', bbox=self.props)

        # Fetching all FPs relevant to that n-value
        n_mask = self.FP_data[:, 1] == n

        # Sort by l_0
        l_0s = self.FP_data[n_mask, -1]
        index_sort = np.argsort(l_0s)

        # Then pick Leftmost, middle and rightmost
        l_0s = l_0s[index_sort][[0, np.sum(n_mask)//2, -1]]
        alphas = self.FP_data[n_mask, -3][index_sort][[0, np.sum(n_mask)//2, -1]]
        betas = self.FP_data[n_mask, -2][index_sort][[0, np.sum(n_mask)//2, -2]]
    
    
        # Plotting 3 FPs
        ax.scatter(l_0s, alphas, s=60, facecolor=['green', 'red', 'green'], edgecolor="k", linewidths=1, zorder=10)

        # This mainly for supplemental figures where you plot the memories as they split
        if plotDynamics:
            ax.streamplot(self.l_0_mesh, self.alpha_mesh, d_ell_dt*norm_condition, d_alpha_dt*norm_condition, color="grey", density=density)
            textbox.set_text("t = "+str(t))
           
            ax.plot(self.l_0[t_0:t, 0], self.coefs[t_0:t, 0, 0], color="red", lw=2)
            ax.scatter(self.l_0[t, 0], self.coefs[t, 0, 0], color="red", marker=matplotlib.markers.CARETUP, s=100)
            
            ax.plot(self.l_0[t_0:t, 1], self.coefs[t_0:t, 1, 0], color="black", lw=2)
            ax.scatter(self.l_0[t, 1], self.coefs[t, 1, 0], color="black", marker=matplotlib.markers.CARETDOWN, s=100)

            return self.coefs[t, :, 0], self.coefs[t, :, 1]
            
        return alphas, betas



    def plot_snapshot(self, ax, alpha, beta, isStable=True, hasStabilityTitle=True, fontsize=14):
        sample_mem = alpha * self.A + beta * self.B
        ax.imshow(sample_mem.reshape(28, 28), cmap="bwr")

        if hasStabilityTitle:
            ax.set_title("Stable", color="green", fontsize=fontsize)
            if not isStable:
                ax.set_title("Unstable", color="red", fontsize=fontsize)


