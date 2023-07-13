import sys
sys.path.append('../')

import numpy as np

from main_module.KrotovV2 import *

data_dir = "data_100_10_200_supplemental/" #"data_100_10_200/"


data_Mf = np.zeros((3, 3, 100, 784))
data_Lf = np.zeros((3, 3, 100, 10))

"""
for i, temp in enumerate([400, 550, 670]):#[800, 850]:
    for j, n in enumerate([3, 15, 30]):
        run = 0
        saving_dir=data_dir+"run_"+str(run)+"_n"+str(n)+"_T"+str(temp)+".npz"
        
        data_tmp = np.load(saving_dir)

        data_Mf[i, j] = data_tmp['M'][-1]
        data_Lf[i, j] = data_tmp['L'][-1]

        print(j)
        
np.save(data_dir+"/data_Mf.npy", data_Mf)
np.save(data_dir+"/data_Lf.npy", data_Lf)
"""

data_Mf = np.load(data_dir+"/data_Mf.npy")
data_Lf = np.load(data_dir+"/data_Lf.npy")

data_T = np.load(data_dir+"/miniBatchs_images.npy")[0]
data_T_inv = np.linalg.pinv(data_T)




all_coefs = np.sum((data_Mf@data_T_inv).reshape(3, 3, 100, 10, 20), axis=-1)

digit_to_observe = 4
index = data_Lf[:, :, :, digit_to_observe] >=0.99 #data_Lf[:, :, :, 4] == 1 #- label-based index
#index *= (np.argmax(all_coefs, axis=-1) == digit_to_observe)


choices = np.zeros((3, 3, 4))
choices[0, 0] = np.asarray([7, 8, 9, 10]); choices[0, 1] = np.asarray([1, 2, 3, 4]); choices[0, 2] = np.asarray([-4, -3, -2, -1])
choices[1, 0] = np.asarray([0, 1, 2, 3]); choices[1, 1] = np.asarray([9, 10, 11, 12]); choices[1, 2] = np.asarray([0, 1, 2, 3])
choices[2, 0] = np.asarray([0, 1, 2, 3]); choices[2, 1] = np.asarray([4, 5, 6, 7]); choices[2, 2] = np.asarray([0, 1, 2, 3])

choices = np.asarray(choices, dtype=int)

#"""
fig, ax = plt.subplots(3, 3, sharex=True)
for i in range(0, 3):
    for j in range(0, 3):
        coefs = (data_Mf[i, j, index[i, j]]@data_T_inv)[choices[i, j, 0]] 
        coefs = coefs.reshape(10, 20)

        ax[i, j].axhline(y=0, color="r", alpha=0.5)
        
        for d in range(0, 10):
            sorted_array = sorted(coefs[d], reverse=True, key=abs)
            
            symmetric_array = sorted_array[::-2]
            symmetric_array = np.concatenate((symmetric_array, [sorted_array[0]]), axis=0)
            symmetric_array = np.concatenate((symmetric_array, sorted_array[2::2]), axis=0)
            
                        
            ax[i, j].plot(np.arange(20*d, 20*(d+1), 1), symmetric_array, marker=".", linewidth=2, ms=3)
                
            ax[i, 0].set_ylabel(str(np.asarray([400, 550, 670])[i]) + "\n"*2 + "Coefficient value")
            ax[-1, j].set_xlabel("Training example (reordered)" + "\n"*2 + str(np.asarray([3, 15, 30])[j]))

            
        ax[i, 0].set_ylim(-0.15, 0.15); ax[i, 1].set_ylim(-0.1, 0.4); ax[i, -1].set_ylim(-0.1, 0.5); 


cbar_ax = fig.add_axes([0.93, 0.168, 0.02, 0.91-0.168])
cb = matplotlib.colorbar.ColorbarBase(cbar_ax, cmap=matplotlib.cm.tab10, norm=matplotlib.colors.Normalize(vmin=0, vmax=9))
cb.ax.set_ylabel("Digit class")

plt.subplots_adjust(top=0.91, bottom=0.168, left=0.1, right=0.92, hspace=0.1, wspace=0.169) 

plt.show()

#exit()
# Top left panel figure
fig, ax = plt.subplots(3, 3, sharex=True, sharey=True)
for i in range(0, 3):
    for j in range(0, 3):
        print(np.asarray([400, 550, 670])[i], np.asarray([3, 15, 30])[j], data_Lf[i, j, index[i, j]][0])
        sample_memories = data_Mf[i, j, index[i, j]][choices[i, j]]
        ax[i, j].imshow(merge_data(sample_memories, 2, 2), cmap="bwr")
        ax[i, j].set_xticks([])
        ax[i, j].set_yticks([])

        ax[i, 0].set_ylabel(str(np.asarray([400, 550, 670])[i]) + "\n")
        ax[-1, j].set_xlabel("\n"*1 + str(np.asarray([3, 15, 30])[j]))

plt.show()
#"""


# Top right panel figure
label_range = np.arange(0, 10, 1)
fig, ax = plt.subplots(3, 3, figsize=(5, 4), sharex=True, sharey=True)
for i in range(0, 3):
    for j in range(0, 3):

        Ls = data_Lf[i, j, index[i, j]]
        
        ax[i, j].plot(label_range, Ls[choices[i, j, 0]], marker="", color="k", linewidth=1)
            
        ax[i, j].scatter(label_range, Ls[choices[i, j, 0]], marker=".", c=label_range, cmap="tab10", s=100, alpha=1)
        
        
        ax[i, j].set_xticks([])
        ax[-1, j].set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        ax[i, j].set_yticks([-1, 1])

        ax[i, 0].set_ylabel(str(np.asarray([400, 550, 670])[i]) + "\n"*2 + "Label value")
        ax[-1, j].set_xlabel("Label element" + "\n"*2 + str(np.asarray([3, 15, 30])[j]))




from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color="k", linewidth=2, alpha=0.3),
                Line2D([0], [0], marker=".", ms=13, color="k")]

fig.legend(custom_lines, ['Labels', 'Mean label'], loc='upper center', ncol=4)
        
cbar_ax = fig.add_axes([0.91, 0.168, 0.02, 0.91-0.168])
cb = matplotlib.colorbar.ColorbarBase(cbar_ax, cmap=matplotlib.cm.tab10, norm=matplotlib.colors.Normalize(vmin=0, vmax=9))
cb.ax.set_ylabel("Label element/class")

plt.subplots_adjust(top=0.91, bottom=0.168, left=0.089, right=0.9, hspace=0.05, wspace=0.05)
plt.show()
#plt.savefig("image.png",bbox_inches='tight',dpi=100)

    
