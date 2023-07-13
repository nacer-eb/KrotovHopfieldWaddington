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


data_dir = "data_100_10_200/"

n, temp = 30, 670

run = 0
selected_digits = [1, 4, 7]
saving_dir = data_dir+"run_"+str(selected_digits)+"_n"+str(n)+"_T"+str(temp)+".npz"
data = np.load(saving_dir)
data_L = data['L']; data_L_key = np.argmax(data_L[-1], axis=-1)
data_M = data['M']
data_T = data['miniBatchs_images'][0]; data_T_inv = np.linalg.pinv(data_T)


coefs = np.sum((data_M@data_T_inv).reshape(3500, 100, 3, 20), axis=-1)

p = [0]*100

tab_10 = matplotlib.cm.tab10
normalizer = matplotlib.colors.Normalize(vmin=0, vmax=9)

props = dict(boxstyle='round', facecolor='whitesmoke', alpha=0.5)
fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
t_samples = [430, 537,  594, 725]

trailsize = 20
for i, t in enumerate(t_samples):
    for j in range(0, 100):
        ax[i//2, i%2].plot(coefs[t, j, 0], coefs[t, j, 1], color=tab_10(normalizer(data_L_key[j])), linestyle="", marker=".", ms=3)
        ax[i//2, i%2].plot(coefs[t-trailsize:t, j, 0], coefs[t-trailsize:t, j, 1], color=tab_10(normalizer(data_L_key[j])), linestyle="-", marker="", lw=1)

    ax[i//2, i%2].text(0.83, 0.97, s=r"$t=$"+str(t), fontsize=14, transform=ax[i//2, i%2].transAxes, verticalalignment='top', bbox=props)
    ax[i//2, 0].set_ylabel(r"$\alpha_4$"); ax[-1, i%2].set_xlabel(r"$\alpha_1$")

from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color=plt.cm.tab10(0.11), marker="o", linestyle="", ms=7),
                Line2D([0], [0], color=plt.cm.tab10(0.11), linestyle="-", lw=4, marker="")]

fig.legend(custom_lines, ['Memory (Visible layer)', 'Memory trajectory/trail'], loc='upper center', ncol=4)
plt.subplots_adjust(top=0.93, bottom=0.09, left=0.052, right=0.92, hspace=0.02, wspace=0.02)

cbar_ax = fig.add_axes([0.925, 0.09, 0.02, 0.84])
cb = matplotlib.colorbar.ColorbarBase(cbar_ax, cmap=matplotlib.cm.tab10, norm=matplotlib.colors.Normalize(vmin=0, vmax=9))
cb.ax.set_ylabel("Memory digit class post-training.")
plt.show()

# animate
"""
fig = plt.figure()
ax = plt.axes()

for j in range(0, 100):
    p[j], = ax.plot(coefs[0, j, 0], coefs[0, j, 1], color=tab_10(normalizer(data_L_key[j])), linestyle="-", marker=".", ms=3)


ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_xlabel(r"$\alpha_1$", labelpad=10); ax.set_ylabel(r"$\alpha_4$", labelpad=10)

props = dict(boxstyle='round', facecolor='whitesmoke', alpha=0.5)
text = ax.text(0.8, 0.95, s=r"$t=$"+str(0), fontsize=14, verticalalignment='top', bbox=props)

manager = plt.get_current_fig_manager()
manager.full_screen_toggle()

def update(t_):
    t = t_ + 430
    for j in range(0, 100):
        p[j].set_data(coefs[t-20:t, j, 0], coefs[t-20:t, j, 1])

    text.set_text(r"$t=$"+str(t))
        
    return *p,

ani = anim.FuncAnimation(fig, update, frames=300, interval=10, blit=False)
ani.save('visible_layer_split[1, 4, 7]_2d.mp4', writer='ffmpeg')
#plt.show()

"""
