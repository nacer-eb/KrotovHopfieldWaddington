import sys
sys.path.append('../')

from main_module.KrotovV2 import *
from nullcline_gather.GatherNullClines import *

import numpy as np
import matplotlib.pyplot as plt
import matplotlib


dataset = "../defaults/miniBatchs_images_Fig_5.npy"

data_T = np.load(dataset)[0]
data_T_inv = np.linalg.pinv(data_T)

selected_digits = [1, 4]

A, B = data_T

d_AA, d_AB, d_BB = A@A, A@B, B@B

data_dir = "data/"

n, temp = 12, 800

da, dl = 0.031, -0.2
tmax = 700

#da, dl = 0.031, -0.2
#tmax = 650

# 1-memory sys
net_K1 = KrotovNet(Kx=1, Ky=1, n_deg=n, m_deg=n, M=2, nbMiniBatchs=1, momentum=0*0.6, rate=0.01, temp=temp/(2.0**(1.0/n)), rand_init_mean=-0.03, rand_init_std=0.03, selected_digits=[1, 4])
net_K1.miniBatchs_images[0] = data_T

net_K1.visibleDetectors[:] = 0.6*A #A

net_K1.hiddenDetectors[:, :] = -1
net_K1.hiddenDetectors[:, selected_digits[0]] = 1

net_K1.train_plot_update(1000, isPlotting=False, isSaving=True, saving_dir=data_dir+"K1.npz")


l_0, alpha_0 = net_K1.hiddenDetectors[0, selected_digits[0]], net_K1.visibleDetectors[0]@data_T_inv[:, 0]



# 2-memory sys
net_K2 = KrotovNet(Kx=2, Ky=1, n_deg=n, m_deg=n, M=2, nbMiniBatchs=1, momentum=0*0.6, rate=0.001, temp=temp, rand_init_mean=-0.03, rand_init_std=0.03, selected_digits=[1, 4])
net_K2.miniBatchs_images[0] = data_T

net_K2.visibleDetectors[0, :] = (alpha_0 + da)*A + (1-alpha_0 - da)*B
net_K2.visibleDetectors[1, :] = (alpha_0 - da)*A + (1-alpha_0 + da)*B

net_K2.hiddenDetectors[:, :] = -1

net_K2.hiddenDetectors[0, selected_digits[0]] = l_0 + dl
net_K2.hiddenDetectors[0, selected_digits[1]] = -l_0 - dl

net_K2.hiddenDetectors[1, selected_digits[0]] = l_0 - dl
net_K2.hiddenDetectors[1, selected_digits[1]] = -l_0 + dl

net_K2.train_plot_update(tmax, isPlotting=False, isSaving=True, saving_dir=data_dir+"K2.npz")




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

    return D_l_A_alpha * dalpha / k_L, ( dt_tilde_alpha_1 - (alpha_0 + dalpha)*(dt_tilde_alpha_1 + dt_tilde_beta_1) )/(2*k_M_1) + ( dt_tilde_beta_2 - (1 - alpha_0 + dalpha)*(dt_tilde_alpha_2 + dt_tilde_beta_2) )/(2*k_M_2)






fig, ax = plt.subplots(1, 2, figsize=(16, 9))
sample_range = np.linspace(0, 1, 1000)
l_0_mesh, alpha_mesh = np.meshgrid(sample_range*2-1, sample_range)

GNC = GatherNullClines(d_AA, d_AB, d_BB, n, temp/(2**(1.0/n)), +1)
ax[0].contour(l_0_mesh, alpha_mesh, GNC.alpha_nullcline(alpha_mesh, l_0_mesh), [0], colors="purple", linewidths=7, alpha=0.5)
ax[0].contour(l_0_mesh, alpha_mesh, GNC.l_0_nullcline(alpha_mesh, l_0_mesh), [0], colors="orange", linewidths=7, alpha=0.5)

delta_range = 0.2
sample_range = np.linspace(-1, 1, 1000)*delta_range
dl_0_mesh, dalpha_mesh = np.meshgrid(sample_range, sample_range)
uv = calculateSplit(dl_0_mesh, dalpha_mesh, l_0, alpha_0, n, temp, d_AA, d_AB, d_BB)
ax[1].streamplot(dl_0_mesh, dalpha_mesh, *uv,  color=(0, 0, 0, 0.2), density=3)


data_K2 = np.load(data_dir+"/K2.npz")
L_K2 = data_K2['L']
M_K2 = data_K2['M']

ax[0].plot(l_0, alpha_0, marker=".", markerfacecolor="green", markeredgecolor="k", markeredgewidth=2, ms=10)

ax[0].plot(L_K2[:, 0, selected_digits[0]], M_K2[:, 0]@data_T_inv[:, 0], lw=3, alpha=0.5, color="k")
ax[0].plot(L_K2[:, 1, selected_digits[0]], M_K2[:, 1]@data_T_inv[:, 0], lw=3, alpha=0.5, color="red")

ax[0].plot(L_K2[-1, 0, selected_digits[0]], M_K2[-1, 0]@data_T_inv[:, 0], ms=10, marker=matplotlib.markers.CARETUPBASE, color="k")
ax[0].plot(L_K2[-1, 1, selected_digits[0]], M_K2[-1, 1]@data_T_inv[:, 0], ms=10, marker=matplotlib.markers.CARETDOWNBASE, color="red")


ax[1].plot(0.5*(L_K2[:, 0, selected_digits[0]] - L_K2[:, 1, selected_digits[0]]), 0.5*(M_K2[:, 0]@data_T_inv[:, 0] - M_K2[:, 1]@data_T_inv[:, 0]), lw=2, color="green")
ax[1].plot(0.5*(L_K2[-1, 0, selected_digits[0]] - L_K2[-1, 1, selected_digits[0]]), 0.5*(M_K2[-1, 0]@data_T_inv[:, 0] - M_K2[-1, 1]@data_T_inv[:, 0]), ms=10, marker=".", color="darkgreen")


ax[1].set_xlim(-delta_range, delta_range)
ax[1].set_ylim(-delta_range, delta_range)

ax[0].set_title(r"$n=$"+str(n)+", "+r"$T_r=$"+'{0:.2f}'.format(temp/784))

ax[0].set_ylabel(r"$\alpha_1$")
ax[0].set_xlabel(r"$\ell$")

ax[1].set_ylabel(r"$\delta \alpha$")
ax[1].set_xlabel(r"$\delta \ell$")

plt.subplots_adjust(wspace=0.4)
plt.savefig("Figure_5S_Splitting_tmp_2.png")
