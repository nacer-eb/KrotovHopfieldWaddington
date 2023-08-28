import sys
sys.path.append('../')

import numpy as np

from main_module.KrotovV2 import *

import matplotlib
font = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 25}

matplotlib.rc('font', **font)

data_dir = "data_100_10_200/"


data_Mf = np.zeros((3, 3, 100, 784))
data_Lf = np.zeros((3, 3, 100, 10))

isFirstRun = False

# Fetch all data
if isFirstRun:
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


data_Mf = np.load(data_dir+"/data_Mf.npy")
data_Lf = np.load(data_dir+"/data_Lf.npy")

data_T = np.load(data_dir+"/miniBatchs_images.npy")[0]
data_T_inv = np.linalg.pinv(data_T)

# Use matrix inverse to obtain coefficients from memories
all_coefs = np.sum((data_Mf@data_T_inv).reshape(3, 3, 100, 10, 20), axis=-1)

# Important: Which digit class we are observing. 4 is chosen because of it UMAP nearness to 7,9. This can be ill-posed for low-n
digit_to_observe = 4

# Getting an index list of all memories with 4 labels (can be ill-posed for low-n)
index = data_Lf[:, :, :, digit_to_observe] >=0.99 

# Manually picking the fours... This turned out to be better, but the code can [Should?] be amended and made automatic.
choices = np.zeros((3, 3, 4))
choices[0, 0] = np.asarray([7, 8, 9, 10]); choices[0, 1] = np.asarray([1, 2, 3, 4]); choices[0, 2] = np.asarray([-4, -3, -2, -1])
choices[1, 0] = np.asarray([0, 1, 2, 3]); choices[1, 1] = np.asarray([9, 10, 11, 12]); choices[1, 2] = np.asarray([0, 1, 2, 3])
choices[2, 0] = np.asarray([0, 1, 2, 3]); choices[2, 1] = np.asarray([4, 5, 6, 7]); choices[2, 2] = np.asarray([0, 1, 2, 3])
choices = np.asarray(choices, dtype=int)



# The sub-figure showing the actually memories picked.
def memory_samples_fig(ax, isLeftAxis=True, isBotAxis=True):   
    for i in range(0, 3):
        for j in range(0, 3):
            print(np.asarray([400, 550, 670])[i], np.asarray([3, 15, 30])[j], data_Lf[i, j, index[i, j]][0])
            sample_memories = data_Mf[i, j, index[i, j]][choices[i, j]]
            ax[i, j].imshow(merge_data(sample_memories, 2, 2), cmap="bwr")
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])

            if isBotAxis:
                ax[-1, j].set_xlabel(str(np.asarray([3, 15, 30])[j]), labelpad=20)
            else:
                ax[0, j].xaxis.set_label_position("top")
                ax[0, j].set_xlabel(str(np.asarray([3, 15, 30])[j]), labelpad=20)

            if isLeftAxis:
                ax[i, 0].set_ylabel(str(np.asarray([400, 550, 670])[i]), labelpad=20)
            else:
                ax[i, -1].yaxis.set_label_position("right")
                ax[i, -1].set_ylabel(str(np.asarray([400, 550, 670])[i]), labelpad=20)

            



# The sub-figure showing the memory coefficients for the selected 4s.
def memory_coefs_fig(ax):
    for i in range(0, 3):
        for j in range(0, 3):
            coefs = (data_Mf[i, j, index[i, j]]@data_T_inv)[choices[i, j, 0]] 
            coefs = coefs.reshape(10, 20)

            ax[i, j].axhline(y=0, color="r", alpha=0.5)

            
            ax[i, j].xaxis.set_label_position("top")
            ax[i, j].xaxis.tick_top()
            ax[0, j].set_xlabel("Training \n Example #", rotation=60)
            if i != 0:
                ax[i, j].set_xticks([])

            
            ax[i, j].yaxis.set_label_position("right")
            ax[i, j].yaxis.tick_right()
            ax[i, -1].set_ylabel(r"$\alpha_\#$", rotation=60, labelpad=20)
            if j != 2: 
                ax[i, j].set_yticks([])
            
            for d in range(0, 10):
                sorted_array = sorted(coefs[d], reverse=True, key=abs)
                
                symmetric_array = sorted_array[::-2]
                symmetric_array = np.concatenate((symmetric_array, [sorted_array[0]]), axis=0)
                symmetric_array = np.concatenate((symmetric_array, sorted_array[2::2]), axis=0)
            
                        
                ax[i, j].plot(np.arange(20*d, 20*(d+1), 1), symmetric_array, marker=".", linewidth=2, ms=3)

            ax[i, j].set_ylim(-0.15, 0.4)

            

            
# The sub-figure showing the label
def label_coefs_fig(ax):
    label_range = np.arange(0, 10, 1)
    for i in range(0, 3):
        for j in range(0, 3):
            
            Ls = data_Lf[i, j, index[i, j]]
            
            ax[i, j].plot(label_range, Ls[choices[i, j, 0]], marker="", color="k", linewidth=1)
            ax[i, j].scatter(label_range, Ls[choices[i, j, 0]], marker=".", c=label_range, cmap="tab10", s=100, alpha=1)

            ax[-1, j].set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
            ax[i, j].set_yticks([-1, 1])

            ax[-1, j].set_xlabel("Training \n Class", rotation=0, labelpad=20)
            
            ax[i, 0].set_ylabel("Label \n Coefficients", rotation=90, labelpad=20)
            
            if i != 2:
                ax[i, j].set_xticks([])

            if j != 0: 
                ax[i, j].set_yticks([])
            





fig = plt.figure(figsize=(19, 19))
axs = fig.subplot_mosaic(
    """
    AABBCC..KKLLMM
    AABBCC..KKLLMM
    DDEEFF..NNOOPP
    DDEEFF..NNOOPP
    HHIIJJ..QQRRSS
    HHIIJJ..QQRRSS
    ..............
    ..............
    TTUUVV..aabbcc
    TTUUVV..aabbcc
    WWXXYY..ddeeff
    WWXXYY..ddeeff
    ZZ1122..hhiijj
    ZZ1122..hhiijj
    """
)


mem_ax_tl = [[axs['A'], axs['B'], axs['C']], [axs['D'], axs['E'], axs['F']], [axs['H'], axs['I'], axs['J']]]
mem_ax_tl = np.asarray(mem_ax_tl)
memory_samples_fig(mem_ax_tl, isBotAxis=False)


mem_ax_br = [[axs['a'], axs['b'], axs['c']], [axs['d'], axs['e'], axs['f']], [axs['h'], axs['i'], axs['j']]]
mem_ax_br = np.asarray(mem_ax_br)
memory_samples_fig(mem_ax_br, isLeftAxis=False)

mem_coef_ax_tr = [[axs['K'], axs['L'], axs['M']], [axs['N'], axs['O'], axs['P']], [axs['Q'], axs['R'], axs['S']]]
mem_coef_ax_tr = np.asarray(mem_coef_ax_tr)
memory_coefs_fig(mem_coef_ax_tr)

label_coef_ax_bl = [[axs['T'], axs['U'], axs['V']], [axs['W'], axs['X'], axs['Y']], [axs['Z'], axs['1'], axs['2']]] 
label_coef_ax_bl= np.asarray(label_coef_ax_bl)
label_coefs_fig(label_coef_ax_bl)


mem_ax_tl[0, 1].text(28, -30, "n-power", fontsize=30, ha='center', family='Times New Roman')
mem_ax_tl[1, 0].text(-32, 2*28, "Temperature", rotation=90, fontsize=30, ha='center', family='Times New Roman')

mem_ax_br[-1, 1].text(28, 29*2+30, "n-power", fontsize=30, ha='center', family='Times New Roman')
mem_ax_br[1, -1].text(28*2 + 32, 2*28, "Temperature", rotation=90, fontsize=30, ha='center', family='Times New Roman')


plt.subplots_adjust(hspace=0.4, wspace=0.4)
plt.savefig("tmp_fig0.png", transparent=True)


