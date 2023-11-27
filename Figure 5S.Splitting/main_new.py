import argparse
import sys
sys.path.append('../')

from main_module.KrotovV2 import *
from splittingDynamics import *

from nullcline_gather.GatherNullClines import *

import numpy as np
import matplotlib.pyplot as plt
import matplotlib



data_dir = "data/"


dataset = "../defaults/miniBatchs_images_Fig_5.npy"
data_T = np.load(dataset)[0]
data_T_inv = np.linalg.pinv(data_T)

selected_digits = [1, 4]

A, B = data_T
A_dot_A, A_dot_B, B_dot_B = A@A, A@B, B@B


# Initial conditions
n = 12
temp = 800

da, dl = 0.031, -0.2 # Initial perturbation
max_epoch = 700

parser = argparse.ArgumentParser(description="This program runs the splitting simulations.")
parser.add_argument('--n', help="The hyperparameter n, the power on the dot product. [DEFAULT=12]", default=12, type=int)
parser.add_argument('--temp', help="The hyperparameter T, the 'renormalizes' the dot product. [DEFAULT=800]", default=800, type=int)
parser.add_argument('--da', help="The initial perturbation on the memories. [DEFAULT=0.031]", default=0.031, type=float)
parser.add_argument('--dl', help="The initial perturbation on the labels. [DEFAULT=-0.2]", default=-0.2, type=float)
parser.add_argument('--maxepoch', help="The maxmimum epoch for which to plot and run the 2 splitting dynamics. [DEFAULT=700]", default=700, type=int)
parse_args = parser.parse_args()

n, temp, da, dl, max_epoch = parse_args.n, parse_args.temp, parse_args.da, parse_args.dl, parse_args.maxepoch

T_tilde = temp/(2.0**(1.0/n))

fig, ax = plt.subplots(1, 2, figsize=(16, 9))


############# --- SIMULATIONS ---- ####################

# 1-memory sys
net_K1 = KrotovNet(Kx=1, Ky=1, n_deg=n, m_deg=n, M=2, nbMiniBatchs=1, momentum=0*0.6, rate=0.001, temp=T_tilde, rand_init_mean=-0.03, rand_init_std=0.03, selected_digits=[1, 4])
net_K1.miniBatchs_images[0] = data_T
net_K1.visibleDetectors[:] = 0.6*A #A
net_K1.hiddenDetectors[:, :] = -1
net_K1.hiddenDetectors[:, selected_digits[0]] = 1

net_K1.train_plot_update(4000, isPlotting=False, isSaving=True, saving_dir=data_dir+"K1.npz")

ell, alpha = net_K1.hiddenDetectors[0, selected_digits[0]], net_K1.visibleDetectors[0]@data_T_inv[:, 0]

# Plotting the fixed point
ax[0].scatter(ell, alpha, s=20, color="k", alpha=1)



# 2-memory sys
net_K2 = KrotovNet(Kx=2, Ky=1, n_deg=n, m_deg=n, M=2, nbMiniBatchs=1, momentum=0, rate=0.001, temp=temp, rand_init_mean=-0.03, rand_init_std=0.03, selected_digits=[1, 4])
net_K2.miniBatchs_images[0] = data_T

net_K2.visibleDetectors[0, :] = (alpha + da)*A + (1-alpha - da)*B
net_K2.visibleDetectors[1, :] = (alpha - da)*A + (1-alpha + da)*B

net_K2.hiddenDetectors[:, :] = -1

net_K2.hiddenDetectors[0, selected_digits[0]] = ell + dl
net_K2.hiddenDetectors[0, selected_digits[1]] = -ell - dl

net_K2.hiddenDetectors[1, selected_digits[0]] = ell - dl
net_K2.hiddenDetectors[1, selected_digits[1]] = -ell + dl

net_K2.train_plot_update(max_epoch, isPlotting=False, isSaving=True, saving_dir=data_dir+"K2.npz")


# Loading saved data and plotting
data = np.load(data_dir+"K2.npz")

data_M = data['M']
data_L = data['L']

alphas = data_M @ data_T_inv[:, 0]
ells = data_L[:, :, selected_digits[0]]

colors = ['k', 'red']

for m_i in range(2):
    ax[0].plot(ells[:, m_i], alphas[:, m_i], linewidth=3, color=colors[m_i], alpha=0.5)
    ax[0].scatter(ells[-1, m_i], alphas[-1, m_i], s=10, color=colors[m_i])

ax[1].plot((ells[:, 0] - ells[:, 1])/2.0, (alphas[:, 0] - alphas[:, 1])/2.0, linewidth=3, color="green", alpha=0.5)
ax[1].scatter((ells[-1, 0] - ells[-1, 1])/2.0, (alphas[-1, 0] - alphas[-1, 1])/2.0, s=20, color="green")



######### ------- ANALYTICAL / NUMERICAL ---------- ##########


# The delta alpha and delta ell meshgrid
delta_range = np.linspace(-0.1, 0.1, 1000)
delta_ell, delta_alpha = np.meshgrid(delta_range, delta_range) # Annoyingly the order matters one the LFS
sd = splittingDynamics(delta_alpha, delta_ell, n, T_tilde, alpha, ell, A_dot_A, A_dot_B, B_dot_B)
ax[1].streamplot(delta_ell, delta_alpha, sd.d_delta_ell_dt, sd.d_delta_alpha_dt, color=(0, 0, 0, 0.2), density=3)
ax[1].set_xlim(-0.1, 0.1)
ax[1].set_ylim(-0.1, 0.1)


# Plotting the nullclines
sample_range = np.linspace(0, 1, 1000)
l_0_mesh, alpha_mesh = np.meshgrid(sample_range*2-1, sample_range)

GNC = GatherNullClines(A_dot_A, A_dot_B, B_dot_B, n, T_tilde, +1)
d_alpha_dt, norm_condition = GNC.calc_d_alpha_dt(alpha_mesh, l_0_mesh)
d_ell_dt, d_ell_dt_p_sat, d_ell_dt_m_sat = GNC.calc_d_ell_dt(alpha_mesh, l_0_mesh)

ax[0].contour(l_0_mesh, alpha_mesh, d_alpha_dt, [0], colors="purple", linewidths=7, alpha=0.5)
ax[0].contour(l_0_mesh, alpha_mesh, d_ell_dt, [0], colors="orange", linewidths=7, alpha=0.5)

ax[0].set_ylabel(r"$\alpha$")
ax[0].set_xlabel(r"$\ell$")

ax[1].set_ylabel(r"$\delta\alpha$")
ax[1].set_xlabel(r"$\delta\ell$")

plt.subplots_adjust(top=0.952, bottom=0.114, left=0.071, right=0.963, hspace=0.2, wspace=0.4)
plt.show()


