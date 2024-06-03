import sys
sys.path.append('../')

from nullcline_gather.GatherNullClines import *


import matplotlib
font = {'size' : 14}
matplotlib.rc('font', **font)

import matplotlib.pyplot as plt

import numpy as np

data_dir="data/"


temp = 700
n = 20 # 20

alpha = 0.4
l_0 = 0.5

def get_data(alpha, l_0):
    saving_dir = data_dir + "run_" + str([1, 4]) \
        + "_n" + str(n) \
        + "_T" + str(temp) \
        + "_alpha" + str(alpha) \
        + "_l_0" + str(l_0) \
        + ".npz"
    

    data = np.load(saving_dir)
    
    data_Ms = data['M']
    data_Ls = data['L']
    data_T = data['miniBatchs_images'][0]
    data_T_inv = np.linalg.pinv(data_T)
    
    tmax = len(data_Ms)
    
    alphas = data_Ms@data_T_inv

    alphas_0 = (alphas[:, 0] + alphas[:, 1])/2.0

    delta_alphas = (alphas[:, 0] - alphas[:, 1])/2.0

    
    ell  = (data_Ls[:, 0, [1, 4]]+data_Ls[:, 1, [1, 4]])/2
    delta_ell  = (data_Ls[:, 0, [1, 4]]-data_Ls[:, 1, [1, 4]])/2

    return data, data_Ms, data_Ls, data_T, data_T_inv, alphas, alphas_0, delta_alphas, ell, delta_ell

data, data_Ms, data_Ls, data_T, data_T_inv, alphas, alphas_0, delta_alphas, ell, delta_ell = get_data(alpha, l_0)



fig = plt.figure(figsize=(16, 9), dpi=300)
ax = fig.add_subplot(projection='3d')
ax.computed_zorder = False

# Y is Ell
ymin, ymax = -0.25, 0.5

if n == 39:
    ymin, ymax = -1, 1

ax.set_ylim(ymin, ymax)

# Z is delta alpha
zmax = 0.5*1E-1
zmin = -zmax
ax.set_zlim(zmin, zmax)

# x is alpha
xmin, xmax = 0, 1
ax.set_xlim(xmin, xmax)

ax.view_init(elev=12, azim=-20)
ax._focal_length = 100

# Plot edges

edges_kw = dict(color='0.8', linewidth=1, zorder=1)
ax.plot([xmin, xmax], [ymin, ymin], 0, **edges_kw)
ax.plot([xmin, xmax], [ymax, ymax], 0, **edges_kw)
ax.plot([xmin, xmin], [ymin, ymax], 0, **edges_kw)
ax.plot([xmax, xmax], [ymin, ymax], 0, **edges_kw)


X = np.linspace(xmin, xmax, 10)
Y = np.linspace(ymin, ymax, 10)
X, Y = np.meshgrid(X, Y)


ax.contourf(X, Y, np.zeros((10, 10)), levels=0, colors="w", alpha=0.5, zorder=4.5)


alphas = np.arange(0.1, 0.9, 0.1)
ells = np.arange(-0.2, 0.5, 0.1)

alphas_ells1 = np.asarray([alphas.ravel(), [-0.2]*len(alphas)]).T
alphas_ells2 = np.asarray([alphas.ravel(), [0.5]*len(alphas)]).T
#alphas_ells3 = np.asarray([ [0.1]*len(ells) , ells.ravel()]).T
alphas_ells4 = np.asarray([ [0.9]*len(ells) , ells.ravel()]).T

alphas_ells = np.concatenate((alphas_ells1, alphas_ells2, alphas_ells4))
alphas_ells = [[0.4, 0.5],
               [0.01, -0.04]] # [0.6, -0.2],

for alpha_ell in alphas_ells: # alphas_ells[[11, 5]]
    print(alpha_ell)
    alpha, l_0 = alpha_ell
    print(alpha)
    data, data_Ms, data_Ls, data_T, data_T_inv, alphas, alphas_0, delta_alphas, ell, delta_ell = get_data(alpha, l_0)
    
    # PLOTTING TRAJECTORIES

    x, z = alphas_0[:], delta_alphas[:]
    y = ell[:]



    t_start = 0
    t_1 = 400

    dt = 200

    if alpha == 0.01:
        t_start = 10 #80
    dt = 400

    color_A, color_B = "red", "darkred"

    if alpha == 0.6:
        color_A, color_B = "black", "gray"


    if alpha == 0.01:
        color_A, color_B = "#5283E4", "#151D6F"


    #matplotlib.markers.CARETUPBASE
    #matplotlib.markers.CARETDOWNBASE
    ax.plot(x[t_start:t_1:dt, 0], y[t_start:t_1:dt, 0], z[t_start:t_1:dt, 0], marker="", ms=10, linestyle="", color=color_A, zorder=6, alpha=0.9)
    ax.plot(x[t_start:t_1:dt, 0], y[t_start:t_1:dt, 0], -z[t_start:t_1:dt, 0], marker="", ms=10, linestyle="", color=color_B, zorder=4, alpha=0.9)


    t_2 = 650
    if alpha == 0.6:
        t_2 = 500

    if alpha == 0.01:
        t_2=800

    t_2 -= 100
    
    ax.plot(x[t_2, 0], y[t_2, 0], z[t_2, 0], marker=matplotlib.markers.CARETUPBASE, ms=10, linestyle="", color=color_A, zorder=6, alpha=0.9)
    ax.plot(x[t_2, 0], y[t_2, 0], -z[t_2, 0], marker=matplotlib.markers.CARETDOWNBASE, ms=10, linestyle="", color=color_B, zorder=4, alpha=0.9)


    ax.plot(x[t_start:t_2, 0], y[t_start:t_2, 0], z[t_start:t_2, 0],  color=color_A, zorder=5, alpha=0.8)
    ax.plot(x[t_start:t_2, 0], y[t_start:t_2, 0], -z[t_start:t_2, 0],  color=color_B, zorder=4, alpha=0.8)


    
A, B = data_T

alpha_mesh, l_0_mesh = np.linspace(xmin+0.02, xmax-0.02, 500), np.linspace(ymin+0.02, ymax-0.02, 500)
alpha_mesh, l_0_mesh = np.meshgrid(alpha_mesh, l_0_mesh)

GNC = GatherNullClines(A@A, A@B, B@B, n, temp/(2.0**(1.0/n)), +1)
d_alpha_dt, norm_condition = GNC.calc_d_alpha_dt(alpha_mesh, l_0_mesh)
d_ell_dt, d_ell_dt_p_sat, d_ell_dt_m_sat = GNC.calc_d_ell_dt(alpha_mesh, l_0_mesh)

ax.contour(alpha_mesh, l_0_mesh, d_alpha_dt, [0], colors="purple", linewidths=5, alpha=0.9, zorder=4.5)
ax.contour(alpha_mesh, l_0_mesh, d_ell_dt, [0], colors="orange", linewidths=5, alpha=0.9, zorder=4.5)



C_data_dir = "../Figure 6/C_Code_FPs/"
data = np.loadtxt(C_data_dir+"save0.dat", delimiter=",", skiprows=1)
p_mask = (np.abs(data[:, -2]) > 0.001) * (np.abs(data[:, -3]) > 0.001 ) * (data[:, 1] <= 7) + (np.abs(data[:, -2]) > 0.03) * (np.abs(data[:, -3]) > 0.03 ) * (data[:, 1] > 7)
FP_data = data[p_mask, :]

# Fetching all FPs relevant to that n-value
n_mask = FP_data[:, 1] == n

# Sort by l_0
l_0s = FP_data[n_mask, -1]
index_sort = np.argsort(l_0s)

# Then pick Leftmost, middle and rightmost
l_0s = l_0s[index_sort][[0, np.sum(n_mask)//2, -1]]
alphas = FP_data[n_mask, -3][index_sort][[0, np.sum(n_mask)//2, -1]]
betas = FP_data[n_mask, -2][index_sort][[0, np.sum(n_mask)//2, -2]]


from matplotlib.patches import Circle
import mpl_toolkits.mplot3d.art3d as art3d

colors = ['green', 'red', 'green']
for i in range(3):
    p = Circle((alphas[i], l_0s[i]), 0.018, color="k")
    ax.add_patch(p)
    art3d.pathpatch_2d_to_3d(p, z=0, zdir="z")
    p.set(zorder=4.6)

    p = Circle((alphas[i], l_0s[i]), 0.015, color=colors[i])
    ax.add_patch(p)
    art3d.pathpatch_2d_to_3d(p, z=0, zdir="z")
    p.set(zorder=4.6)
    ## Plotting 3 FPs
    #ax.scatter(alphas, l_0s, 0, s=40, facecolor=['green', 'red', 'green'], edgecolor="k", linewidths=1, zorder=6)



ax.set_box_aspect((1, 1, 0.4))

ax.set_zticks([zmin/2, 0, zmax/2])

edges_kw = dict(color='0.8', linewidth=1, zorder=10)

xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()
zmin, zmax = ax.get_zlim()

#ax.plot([xmax, xmax], [ymin, ymax], zmax, **edges_kw)
#ax.plot([xmin, xmax], [ymin, ymin], zmax, **edges_kw)
#ax.plot([xmax, xmax], [ymin, ymin], [zmin, zmax], **edges_kw)


ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.set_zlim(zmin, zmax)



ax.set_xlabel(r"$\alpha$", labelpad=20)
ax.set_ylabel(r"$\ell$", labelpad=20)
ax.set_zlabel(r"$\delta\alpha$", labelpad=15)
ax.tick_params(axis='z', which='major', pad=7)

ax.grid(False)
plt.savefig("Figure_6_3D_Split.png", transparent=True)


"""
plt.plot(delta_alphas[:, 1])
plt.savefig("tmp.png")
"""
