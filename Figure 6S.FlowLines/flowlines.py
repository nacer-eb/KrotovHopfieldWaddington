import sys
sys.path.append("../single_memory_dynamics_module/")

from single_memory_dynamics import *

import numpy as np

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(16, 9), dpi=300)
ax = fig.add_subplot(projection='3d')

ax.view_init(elev=20, azim=-20, roll=0) #ax.view_init(elev=0, azim=-0, roll=0)
ax.set_box_aspect((1, 1, 0.4))

# l_0, alpha, beta

ax.set_xlabel(r'$\ell$'); ax.set_xlim(-1, 1)
ax.set_ylabel(r"$\alpha_A$"); ax.set_ylim(0, 1)
ax.set_zlabel(r"$\alpha_A + \alpha_B$"); ax.set_zlim(0.8, 1)

ell_flat = np.linspace(-1, 1, 100)
alpha_flat = np.linspace(0, 1, 100)

ell, alpha = np.meshgrid(ell_flat, alpha_flat)

"""
ax.plot_surface(ell, alpha, 1 - alpha, color="k", alpha=0.2)
ax.plot(ell[0], alpha[0], 1 - alpha[0], color="grey", alpha=0.8)
ax.plot(ell[-1], alpha[-1], 1 - alpha[-1], color="grey", alpha=0.8)
ax.plot(ell[:, 0], alpha[:, 0], 1 - alpha[:, 0], color="grey", alpha=0.8)
ax.plot(ell[:, -1], alpha[:, -1], 1 - alpha[:, -1], color="grey", alpha=0.8)
"""
ax.plot_surface(ell, alpha, np.zeros_like(ell)+1, color="k", alpha=0.2)
ax.plot(ell[0], alpha[0], 1, color="grey", alpha=0.8)
ax.plot(ell[-1], alpha[-1], 1, color="grey", alpha=0.8)
ax.plot(ell[:, 0], alpha[:, 0], 1, color="grey", alpha=0.8)
ax.plot(ell[:, -1], alpha[:, -1], 1, color="grey", alpha=0.8)

dataset = "../defaults/miniBatchs_images_Fig_6.npy"
A, B = np.load(dataset)[0]
smd = single_memory_dynamics(A@A, A@B, B@B, 15, 700)

size = 0.05
ell_range = np.arange(-1, 1, size)
for ell_ in ell_range:
    R = 1

    alpha_A_range = np.arange(0, R, size)
    for alpha_A in alpha_A_range:
        
        alpha_B = R - alpha_A
        print(ell_, alpha_A, alpha_B)
        
        ell_t, alpha_A_t, alpha_B_t = smd.simulate_and_save(alpha_A, alpha_B, ell_, 1000, 0.001, "tmp_sv")
        
        print(ell_t[0], alpha_A_t[0], alpha_B_t[0])
            
            
        ax.plot(ell_t, alpha_A_t, alpha_A_t + alpha_B_t, c="k", alpha=0.1, lw=1)

        if ell_ == ell_range[-1] and alpha_A == alpha_A_range[1]:
            ax.plot(ell_t, alpha_A_t, alpha_A_t + alpha_B_t, c="blue", alpha=0.8, lw=1)

        if ell_ == ell_range[1] and alpha_A == alpha_A_range[1]:
            ax.plot(ell_t, alpha_A_t, alpha_A_t + alpha_B_t, c="red", alpha=0.8, lw=1)

        

#ax.computed_zorder = False
plt.savefig("tmp.png")


"""
d_alpha_A_dt, d_alpha_B_dt = smd.calc_d_alphas_dt(alpha_A, alpha_B, ell_)
d_ell_dt = smd.calc_d_ell_dt(alpha_A, alpha_B, ell_)

n1 = np.abs(d_alpha_A_dt) + np.abs(d_alpha_B_dt)

d_alpha_A_dt /= n1
d_alpha_B_dt /= n1
            
n2 = np.abs(d_ell_dt)
d_ell_dt /= n2

lr = 0.05
"""
        
