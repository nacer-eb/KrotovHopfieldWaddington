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




dataset = "../defaults/miniBatchs_images_Fig_6.npy"
A, B = np.load(dataset)[0]

alpha_mesh, l_0_mesh = np.linspace(0, 1, 500), np.linspace(-1, 1, 500)
alpha_mesh, l_0_mesh = np.meshgrid(alpha_mesh, l_0_mesh)

GNC = GatherNullClines(A@A, A@B, B@B, n, temp/(2.0**(1.0/n)), +1)
d_alpha_dt, norm_condition = GNC.calc_d_alpha_dt(alpha_mesh, l_0_mesh)
d_ell_dt, d_ell_dt_p_sat, d_ell_dt_m_sat = GNC.calc_d_ell_dt(alpha_mesh, l_0_mesh)

fig, ax = plt.subplots(1, 1, dpi=600)

data_, data_Ms_, data_Ls_, data_T_, data_T_inv_, alphas_, alphas_0_, delta_alphas_, ell_, delta_ell_ = get_data(0.01, -0.04)


tmin, tmax = 405, 668 #640
print(alphas_0_[tmin:tmax, 0])

ax.contour(alpha_mesh, l_0_mesh, d_alpha_dt, [0], colors="purple", linewidths=5, alpha=0.7)
cs = ax.contour(alpha_mesh, l_0_mesh, d_ell_dt, [0], colors="orange", linewidths=5, alpha=0.2, zorder=4.5)

p = cs.collections[0].get_paths()[0]
nullcline_al = p.vertices
ax.plot(nullcline_al[:, 0], nullcline_al[:, 1], ".", color="orange", ms=1) # Testing
ax.plot(alphas_[tmin:tmax, 0, 0], data_Ls_[tmin:tmax, 0, 1])
ax.plot(alphas_[tmin:tmax, 1, 0], data_Ls_[tmin:tmax, 1, 1])
plt.savefig("tmp_.png")

fig, ax = plt.subplots(1, 1)
ax.plot(data_Ls_[tmin:tmax, 0, 0], color="k")
ax.plot(data_Ls_[tmin:tmax, 1, 0])
plt.savefig("tmp__.png")

print(delta_ell_[tmin:tmax, 0]/delta_alphas_[tmin:tmax, 0])


fig = plt.figure(figsize=(16, 9), dpi=200)
ax = fig.add_subplot(projection='3d')
ax.computed_zorder = False

"""
# Z is E
zmin, zmax = 0, 1
ax.set_zlim(zmin, zmax)
"""
# x is delta alpha
xmax = 2E-1
xmin = -xmax
ax.set_xlim(xmin, xmax)
ax.set_xticks([xmin/4, 0, xmax/4])

# y is alpha
#ymin, ymax = 0.1, 0.65
#ax.set_ylim(ymin, ymax)

ax.view_init(elev=15, azim=20)#-20 #12
ax._focal_length = 100


# Plot edges
edges_kw = dict(color='0.8', linewidth=1, zorder=5)


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



ax.set_box_aspect((1, 1, 0.4))

ax.grid(False)
plt.savefig("tmp_2.png")


def calc_E(M_A, M_B, L_A, L_B, A, B, n, temp):
    O_A = np.tanh( L_A*np.expand_dims(M_A@A/temp, axis=-1).repeat(10, axis=-1)**n + L_B*np.expand_dims(M_B@A/temp, axis=-1).repeat(10, axis=-1)**n )
    O_B = np.tanh( L_A*np.expand_dims(M_A@B/temp, axis=-1).repeat(10, axis=-1)**n + L_B*np.expand_dims(M_B@B/temp, axis=-1).repeat(10, axis=-1)**n )

    t_A = np.zeros((10))
    t_A[:] = -1
    t_A[1] = 1

    t_B = np.zeros((10))
    t_B[:] = -1
    t_B[4] = 1

    E = np.sum((t_A - O_A)**(2*n) + (t_B - O_B)**(2*n), axis=-1)

    return E


#fig, ax = plt.subplots(1, 1)
#ax.scatter(alphas, l_0s)


dataset = "../defaults/miniBatchs_images_Fig_6.npy"
A, B = np.load(dataset)[0]


from matplotlib.colors import LightSource


import matplotlib.colors as colors
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100, c_str="darkorange", prop=0.5):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        (1-prop)*cmap(np.linspace(minval, maxval, n)) + np.asarray(matplotlib.colors.to_rgba(c_str))*prop )
    return new_cmap


p = np.polyfit(nullcline_al[:, 0], nullcline_al[:, 1], deg=100)
p_delta = np.polyfit(delta_alphas_[tmin:tmax+100, 0], delta_ell_[tmin:tmax+100, 0], deg=100) # 100 is a bit overkill, 1 works too... i.e. not very sensitive to choice of this


for dr_i in range(4):
    size = 200
    alphas_detailed = np.linspace(min(alphas)-0.07, max(alphas)+0.07, size) #max(alphas)+0.05

    dr_max = 0.03
    dr_center = 0.005
    if dr_i == 0:
        dr = np.linspace(dr_center, dr_max, size)

    if dr_i == 1:
        dr = np.linspace(-dr_max, -dr_center, size)

    
    if dr_i == 2:
        dr = np.linspace(0, dr_center, size)

    if dr_i == 3:
        dr = np.linspace(-dr_center, 0, size)
    
    
    betas_detailed = 1-alphas_detailed
    ells_detailed = np.polyval(p, alphas_detailed)

    
    da = dr
    dl = np.polyval(p_delta, da)

    dl[da<0] = -np.polyval(p_delta, -da[da<0])
    
    M_A = np.zeros((size, size, 784))
    M_B = np.zeros((size, size, 784))
    
    L_A = np.zeros((size, size, 10))
    L_B = np.zeros((size, size, 10))
    L_A[:, :, :] = -1
    L_B[:, :, :] = -1
    
    for i in range(size):
        for j in range(size):
            M_A[i, j] = (alphas_detailed[i]+da[j])*A + (betas_detailed[i]-da[j])*B
            M_B[i, j] = (alphas_detailed[i]-da[j])*A + (betas_detailed[i]+da[j])*B
            
            L_A[i, j, 1] = (ells_detailed[i]+dl[j])
            L_B[i, j, 1] = (ells_detailed[i]-dl[j])
            
            L_A[i, j, 4] = (-ells_detailed[i]-dl[j]) # -(ells_detailed[i]+dl[j])
            L_B[i, j, 4] = (-ells_detailed[i]+dl[j]) # -(ells_detailed[i]-dl[j])
            
        
        
    E = calc_E(M_A, M_B, L_A, L_B, A, B, n, temp)

    ls = LightSource(azdeg=120)
    illuminated_surface = ls.shade(-E, cmap=truncate_colormap(plt.get_cmap("Greys"), minval=0.5, maxval=0.8, prop=0.5, c_str="Grey")) 

    dr, alphas_detailed = np.meshgrid(dr, alphas_detailed)

    color="grey"
    alpha=1.0
    if dr_i>1:
        color="orange"

    if dr_i == 0:
        im1 = ax.plot_surface(dr, alphas_detailed, E, zorder=6, linewidth=0, shade=True, color=color, rstride=1, cstride=1, alpha=alpha, antialiased=False)
        
    if dr_i == 1:
        im2 = ax.plot_surface(dr, alphas_detailed, E, zorder=3, linewidth=0, shade=True, color=color, rstride=1, cstride=1, alpha=alpha, antialiased=False) # facecolors=illuminated_surface


    if dr_i == 2:
        im1 = ax.plot_surface(dr, alphas_detailed, E, zorder=6, linewidth=0, color=color, rstride=1, cstride=1, alpha=alpha, antialiased=False)
        
    if dr_i == 3:
        im2 = ax.plot_surface(dr, alphas_detailed, E, zorder=3, linewidth=0,  color=color, rstride=1, cstride=1, alpha=alpha, antialiased=False) # facecolors=illuminated_surface
        




# Assume delta ell is 0?
"""
fig, ax = plt.subplots(1 , 1)
ax.plot(delta_alphas_[tmin:tmax, 0], delta_ell_[tmin:tmax, 0])
print(p)

ax.plot(delta_alphas_[tmin:tmax, 0], np.polyval(p_delta, delta_alphas_[tmin:tmax, 0]), c="k")
plt.savefig("debug.png")
exit()
"""

print(np.polyval(p_delta, delta_alphas_[tmin:tmax, 0])-delta_ell_[tmin:tmax, 0])

data_Ls_r = np.copy(data_Ls_[tmin:tmax])
"""

data_Ls_r[:, 0, 1] = (data_Ls_[tmin:tmax, 0, 1] + data_Ls_[tmin:tmax, 1, 1])/2.0
data_Ls_r[:, 0, 4] = (data_Ls_[tmin:tmax, 0, 4] + data_Ls_[tmin:tmax, 1, 4])/2.0
data_Ls_r[:, 1, 1] = data_Ls_r[:, 0, 1]
data_Ls_r[:, 1, 4] = data_Ls_r[:, 0, 4]
"""

color_A, color_B = "#5283E4", "#151D6F"
ax.plot(delta_alphas_[tmin:tmax, 0], alphas_0_[tmin:tmax, 0], calc_E(data_Ms_[tmin:tmax, 0], data_Ms_[tmin:tmax, 1], data_Ls_r[:, 0], data_Ls_r[:, 1], A, B, n, temp),
        color=color_A, zorder=6, alpha=0.9, lw=3)
ax.plot(-delta_alphas_[tmin:tmax, 0], alphas_0_[tmin:tmax, 0], calc_E(data_Ms_[tmin:tmax, 0], data_Ms_[tmin:tmax, 1], data_Ls_r[:, 0], data_Ls_r[:, 1], A, B, n, temp),
        color=color_B, zorder=3.5, alpha=0.9, lw=3)




#ax.plot(dr[:, -1], alphas_detailed[:, -1], E[:, -1], color="orange", alpha=0.5, zorder=5)

#t = 23
#ax.plot(dr[t, :], alphas_detailed[t, :], E[t, :], color="k", alpha=0.5, zorder=5)


alphas_detailed = np.linspace(min(alphas)-0.1, max(alphas)+0.05, size)
E_range = np.linspace(np.min(E)-0.2, np.max(E)+0.2, 100)
alphas_detailed, E_range = np.meshgrid(alphas_detailed, E_range)



#ax.plot_surface(np.zeros_like(alphas_detailed), alphas_detailed, E_range,  zorder=4.5, facecolor="grey", alpha=0.2, rstride=1, cstride=1) #alpha=0.5,




ax.set_xlabel(r"$\delta\alpha$", labelpad=20)
ax.set_ylabel(r"$\alpha$", labelpad=20)

ax.zaxis.set_rotate_label(False)
ax.set_zlabel(r"$C(\alpha, \ell, \delta\alpha, \delta\ell)$", rotation=90)


edges_kw = dict(color='0.8', linewidth=1, zorder=5)

xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()
zmin, zmax = ax.get_zlim()

#ax.plot([xmax, xmax], [ymin, ymax], zmax, **edges_kw)
#ax.plot([xmin, xmax], [ymin, ymin], zmax, **edges_kw)
#ax.plot([xmax, xmax], [ymin, ymin], [zmin, zmax], **edges_kw)


ax.set_xlim(-0.05, 0.05)
ax.set_ylim(ymin, ymax)
ax.set_zlim(zmin, zmax)

plt.savefig("Figure_6_EnergyLandscape.png", transparent=True)




"""
plt.plot(delta_alphas[:, 1])
plt.savefig("tmp.png")
"""
