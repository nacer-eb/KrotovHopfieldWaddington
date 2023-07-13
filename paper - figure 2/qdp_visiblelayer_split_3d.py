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

n, temp = 33, 800 #30, 670

run = 0
selected_digits = [1, 9, 4]
saving_dir = data_dir+"run_"+str(selected_digits)+"_n"+str(n)+"_T"+str(temp)+".npz"
data = np.load(saving_dir)

data_L = data['L']
data_L_key = np.argmax(data_L[-1], axis=-1)

data_M = data['M']


data_T = data['miniBatchs_images'][0]
data_T_inv = np.linalg.pinv(data_T)


coefs = np.sum((data_M@data_T_inv).reshape(len(data_M), 100, 3, 20), axis=-1)

p = [0]*100

tab_10 = matplotlib.cm.tab10
normalizer = matplotlib.colors.Normalize(vmin=0, vmax=9)

fig = plt.figure()
ax = plt.axes(projection='3d')


for j in range(0, 100):
    p[j], = ax.plot(coefs[0, j, 0], coefs[0, j, 1], coefs[0, j, 2], color=tab_10(normalizer(data_L_key[j])), linestyle="-", marker=".", ms=1, lw=1)


ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_zlim(0, 1)  
ax.view_init(azim=39.166347386082904, elev=25.006011282715246)

ax.xaxis.set_rotate_label(False); ax.yaxis.set_rotate_label(False); ax.zaxis.set_rotate_label(False)
ax.set_xlabel(r"$\alpha_"+str(selected_digits[0])+"$", rotation=0, labelpad=10); ax.set_ylabel(r"$\alpha_"+str(selected_digits[1])+"$", rotation=0, labelpad=10); ax.set_zlabel(r"$\alpha_"+str(selected_digits[2])+"$", rotation=0, labelpad=10)

props = dict(boxstyle='round', facecolor='whitesmoke', alpha=0.5)
text = ax.text(-0.9, -0.7, 0.95, s=r"$t=$"+str(0), fontsize=14, verticalalignment='top', bbox=props)

manager = plt.get_current_fig_manager()
manager.full_screen_toggle()

def update(t_):
    t = t_*5 + 1500
    for j in range(0, 100):
        p[j].set_data(coefs[1500:t, j, 0], coefs[1500:t, j, 1])
        p[j].set_3d_properties(coefs[1500:t, j, 2])

    text.set_text(r"$t=$"+str(t))
    print(ax.azim, ax.elev)
        
    return *p,

ani = anim.FuncAnimation(fig, update, frames=3500//5, interval=10, blit=True)
ani.save('visible_layer_split[1, 9, 4]_tmp.mp4', writer='ffmpeg')
#ani.save('visible_layer_split[1, 4, 7].mp4', writer='ffmpeg')
#plt.show()


