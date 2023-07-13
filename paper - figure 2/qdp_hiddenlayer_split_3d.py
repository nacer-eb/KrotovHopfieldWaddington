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

n, temp = 30, 800#40, 670

run = 0
selected_digits = [1, 9, 4]
saving_dir = data_dir+"run_"+str(selected_digits)+"_n"+str(n)+"_T"+str(temp)+".npz"
data = np.load(saving_dir)

data_L = data['L']
data_L_key = np.argmax(data_L[-1], axis=-1)


p = [0]*100

tab_10 = matplotlib.cm.tab10
normalizer = matplotlib.colors.Normalize(vmin=0, vmax=9)

fig = plt.figure()
ax = plt.axes(projection='3d')


for j in range(0, 100):
    p[j], = ax.plot(data_L[0, j, selected_digits[0]], data_L[0, j, selected_digits[1]], data_L[0, j, selected_digits[2]], color=tab_10(normalizer(data_L_key[j])), linestyle="-", marker=".", ms=1, linewidth=1)


ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)  
ax.view_init(azim=34.249657956965116, elev=47.35459959688673)

ax.xaxis.set_rotate_label(False); ax.yaxis.set_rotate_label(False); ax.zaxis.set_rotate_label(False)
ax.set_xlabel(r"$L_"+str(selected_digits[0])+"$", rotation=0, labelpad=10); ax.set_ylabel(r"$L_"+str(selected_digits[1])+"$", rotation=0, labelpad=10); ax.set_zlabel(r"$L_"+str(selected_digits[2])+"$", rotation=0, labelpad=10)

props = dict(boxstyle='round', facecolor='whitesmoke', alpha=0.5)
text = ax.text(-0.9, -0.7, 0.95, s=r"$t=$"+str(0), fontsize=14, verticalalignment='top', bbox=props)

manager = plt.get_current_fig_manager()
manager.full_screen_toggle()

def update(t_):
    t = t_*5 #+ 430
    for j in range(0, 100):
        p[j].set_data(data_L[:t, j, selected_digits[0]], data_L[:t, j, selected_digits[1]])
        p[j].set_3d_properties(data_L[:t, j, selected_digits[2]])

    text.set_text(r"$t=$"+str(t))
    print(ax.azim, ax.elev)
        
    return *p, text,

ani = anim.FuncAnimation(fig, update, frames=3500//5, interval=10, blit=True, repeat=True)
#ani.save('hidden_layer_split[1, 4, 7].mp4', writer='ffmpeg')
plt.show()
