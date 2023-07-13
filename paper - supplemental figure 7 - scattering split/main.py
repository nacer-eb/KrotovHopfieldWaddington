import sys
sys.path.append('../')

from nullcline_gather.GatherNullClines import *

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import matplotlib.animation as anim

from main_module.KrotovV2 import *


n, T = 20, 700 #7, 1200 Good


data_dir = "data_2_2_2/"

selected_digits = [1, 4]
l_0, alpha_0 = 0.095, 0.561 
filename = "test20.npz"
delta_range = 0.2 #0.1

miniBatchs_images = np.load(data_dir+"miniBatchs_images.npy")# np.load(data_dir+"miniBatchs_images_14.npy")
A, B = miniBatchs_images[0, 0], miniBatchs_images[0, 1]

#d_AA, d_AB, d_BB = 729, 586, 723
d_AA, d_AB, d_BB = A@A, A@B, B@B
d_BA = d_AB


def calculateSplit(dl, dalpha, l_0, alpha_0, n, T, d_AA, d_AB, d_BB):

    d_A = (d_AB + alpha_0*(d_AA - d_AB))/T
    d_B = (d_BB - alpha_0*(d_BB - d_AB))/T
    
    delta_d_A = (d_AA-d_AB)/T
    delta_d_B = (d_BB - d_AB)/T

    l_A_o_A = np.tanh(2*l_0*d_A**n)
    l_B_o_A = -np.tanh(2*l_0*d_A**n)
    l_gamma_o_A = -np.tanh(2*d_A**n)

    l_A_o_B = np.tanh(2*l_0*d_B**n)
    l_B_o_B = -np.tanh(2*l_0*d_B**n)
    l_gamma_o_B = -np.tanh(2*d_B**n)


    # label quantities

    D_l_A_0 = (1 - l_A_o_A)**(2*n-1) * (1 - l_A_o_A**2) * d_A**n - (1 + l_A_o_B)**(2*n-1) * (1 - l_A_o_B**2) * d_B**n
    D_l_A_alpha = n * ( (1 - l_A_o_A)**(2*n-1) * (1 - l_A_o_A**2) * d_A**(n-1) * delta_d_A  +  (1 + l_A_o_B)**(2*n-1) * (1 - l_A_o_B**2) * d_B**(n-1) * delta_d_B )

    dt_tilde_l_A = D_l_A_0 + D_l_A_alpha * dalpha
    dt_tilde_l_gamma = - (1 + l_gamma_o_A)**(2*n-1) * (1 - l_gamma_o_A**2) * d_A**n - (1 + l_gamma_o_B)**(2*n-1) * (1 - l_gamma_o_B**2) * d_B**n

    k_L = np.maximum(np.abs(dt_tilde_l_A), np.abs(dt_tilde_l_gamma))
    
    # memory quantities
    
    D_alpha_0 = 2 * (1 - l_A_o_A)**(2*n-1) * (1 - l_A_o_A**2) * l_0 * d_A**(n-1) + 8 * (1 + l_gamma_o_A)**(2*n-1) * (1 - l_gamma_o_A**2) * d_A**(n-1)
    D_alpha_l = 2 * (1 - l_A_o_A)**(2*n-1) * (1 - l_A_o_A**2) * d_A**(n-1)
    D_alpha_alpha = (n-1) * D_alpha_0 * delta_d_A / d_A

    dt_tilde_alpha_1 = D_alpha_0 + D_alpha_l*dl + D_alpha_alpha*dalpha
    dt_tilde_alpha_2 = D_alpha_0 - D_alpha_l*dl - D_alpha_alpha*dalpha

    D_beta_0 = -2 * (1 + l_A_o_B)**(2*n-1) * (1 - l_A_o_B**2) * l_0 * d_B**(n-1) + 8 * (1 + l_gamma_o_B)**(2*n-1) * (1 - l_gamma_o_B**2) * d_B**(n-1)
    D_beta_l = -2 * (1 + l_A_o_B)**(2*n-1) * (1 - l_A_o_B**2) * d_B**(n-1)
    D_beta_alpha = (n-1) * D_beta_0 * delta_d_B / d_B

    dt_tilde_beta_1 = D_beta_0 + D_beta_l * dl - D_beta_alpha * dalpha
    dt_tilde_beta_2 = D_beta_0 - D_beta_l * dl + D_beta_alpha * dalpha

    k_M_1 = np.abs(dt_tilde_alpha_1) + np.abs(dt_tilde_beta_1)
    k_M_2 = np.abs(dt_tilde_alpha_2) + np.abs(dt_tilde_beta_2)

    k_M = np.minimum(k_M_1, k_M_2)
    k_M = k_M_1

    #D_alpha_l * dl + D_alpha_alpha*dalpha - alpha_0*((D_alpha_l + D_beta_l)*dl + (D_alpha_alpha - D_beta_alpha)*dalpha) - dalpha * (D_alpha_0 + D_beta_0)
    return D_l_A_alpha * dalpha / k_L, ( dt_tilde_alpha_1 - (alpha_0 + dalpha)*(dt_tilde_alpha_1 + dt_tilde_beta_1) )/(2*k_M_1) + ( dt_tilde_beta_2 - (1 - alpha_0 + dalpha)*(dt_tilde_alpha_2 + dt_tilde_beta_2) )/(2*k_M_2)


#"""


fig, ax = plt.subplots(1, 2, figsize=(16, 9))
sample_range = np.linspace(0, 1, 1000)
l_0_mesh, alpha_mesh = np.meshgrid(sample_range*2-1, sample_range)

GNC = GatherNullClines(d_AA, d_AB, d_BB, n, T/(2**(1.0/n)), +1)
ax[0].contour(l_0_mesh, alpha_mesh, GNC.alpha_nullcline(alpha_mesh, l_0_mesh), [0], colors="orange", linewidths=7, alpha=0.5)
ax[0].contour(l_0_mesh, alpha_mesh, GNC.l_0_nullcline(alpha_mesh, l_0_mesh), [0], colors="purple", linewidths=7, alpha=0.5)
 
sample_range = np.linspace(-1, 1, 1000)*delta_range
dl_0_mesh, dalpha_mesh = np.meshgrid(sample_range, sample_range)
uv = calculateSplit(dl_0_mesh, dalpha_mesh, l_0, alpha_0, n, T, d_AA, d_AB, d_BB)
ax[1].streamplot(dl_0_mesh, dalpha_mesh, *uv,  color=(0, 0, 0, 0.2), density=3)

#plt.show(); exit()

A_p = (d_BB*A - d_AB*B)/(d_AA*d_BB-d_AB**2)
B_p = -(d_AB*A - d_AA*B)/(d_AA*d_BB-d_AB**2)


N = 1
tmax = 2000-1 #4000

lineA = [0]*N
lineB = [0]*N

alphas = np.zeros((N, 4000, 2))
l = np.zeros((N, 4000, 2))
for i in range(0, 1):
    data = np.load(data_dir+filename)

    alphas[i, :, :] = data['M'] @ A_p


    l[i, :, :] = data['L'][:, :, selected_digits[0]]


    ax[0].plot(l[i, :tmax, 0], alphas[i, :tmax, 0], marker="", color="blue")
    lineA[i], = ax[0].plot(l[i, tmax, 0], alphas[i, tmax, 0], marker=matplotlib.markers.CARETUP, linestyle="", ms=10, color="blue")

    ax[0].plot(l[i, :tmax, 1], alphas[i, :tmax, 1], marker="", color="red")
    lineB[i], = ax[0].plot(l[i, tmax, 1], alphas[i, tmax, 1], marker=matplotlib.markers.CARETDOWN, linestyle="", ms=10, color="red")


    

#np.clip(dl_0_mesh*uv[0]*r, -1, 1) , np.clip(dl_0_mesh*uv[1]*r + dalpha_mesh*uv[2]*r, -1, 1), color="black", density=3)

lineC, = ax[1].plot(0.5*(l[0, :tmax, 0] - l[0, :tmax, 1]), 0.5*(alphas[0, :tmax, 0] - alphas[0, :tmax, 1]), marker="", color="purple", ms=5)
#lineCp, = ax[1].plot(0.5*(l[0, tmax, 0] - l[0, tmax, 1]), 0.5*(alphas[0, tmax, 0] - alphas[0, tmax, 1]), marker=".", color="purple", linestyle="", ms=5)
lineCp, = ax[1].plot(0.5*(l[0, tmax, 0] - l[0, tmax, 1]), 0.5*(alphas[0, tmax, 0] - alphas[0, tmax, 1]), marker=matplotlib.markers.CARETUP, color="blue", linestyle="", ms=5)
lineCp, = ax[1].plot(0.5*(l[0, tmax, 0] - l[0, tmax, 1]), 0.5*(alphas[0, tmax, 0] - alphas[0, tmax, 1]), marker=matplotlib.markers.CARETDOWN, color="red", linestyle="", ms=5)


from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], marker=matplotlib.markers.CARETUP, linestyle="", ms=10, color="blue"),
                Line2D([0], [0], marker=matplotlib.markers.CARETDOWN, linestyle="", ms=10, color="red"),
                Line2D([0], [0], marker="", color="red")]

fig.legend(custom_lines, ['Memories', '', 'Trajectories'], loc='upper center', ncols=3)

ax[0].set_xlabel(r"$l_0$"); ax[0].set_ylabel(r"$\alpha_0$")
ax[1].set_xlabel(r"$\delta l$"); ax[1].set_ylabel(r"$\delta \alpha$")

plt.subplots_adjust(top=0.88, bottom=0.1, left=0.05, right=0.95, hspace=0.2, wspace=0.25)
plt.show()
