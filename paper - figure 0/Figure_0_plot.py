import sys
sys.path.append('../')

import numpy as np

from main_module.KrotovV2 import *

import matplotlib
fontsize = 50
font = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : fontsize}

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

# Choices from the above indices. As of now this is done manually, but you can do random selection or [0, 1, 2, 3] if you want.
choices = np.zeros((3, 3, 4)) 
choices[0, 0] = np.asarray([7, 8, 9, 10]); choices[0, 1] = np.asarray([1, 2, 3, 4]); choices[0, 2] = np.asarray([-4, -3, -2, -1])
choices[1, 0] = np.asarray([0, 1, 2, 3]); choices[1, 1] = np.asarray([9, 10, 11, 12]); choices[1, 2] = np.asarray([0, 1, 2, 3])
choices[2, 0] = np.asarray([0, 1, 2, 3]); choices[2, 1] = np.asarray([4, 5, 6, 7]); choices[2, 2] = np.asarray([0, 1, 2, 3])
choices = np.asarray(choices, dtype=int)



# The sub-figure showing the actually memories picked.
def memory_samples_fig(ax, isLeftAxis=True, isBotAxis=True):   
    for i in range(0, 3):
        for j in range(0, 3):
            sample_memories = data_Mf[i, j, index[i, j]][choices[i, j]] # Picking the relevant 4 memories from the complete memory set

            ax[i, j].imshow(merge_data(sample_memories, 2, 2), cmap="bwr") # Plotting using the merge-data tool (KrotovV2_utils)

            # Cosmetics
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

            # Picking relevant memories and calculating their coefficient
            coefs = (data_Mf[i, j, index[i, j]]@data_T_inv)[choices[i, j, 0]]
            coefs = coefs.reshape(10, 20)

            # Plotting a zero-line for reference
            ax[i, j].axhline(y=0, color="r", alpha=0.5)

            # Cosmetics
            ax[2, j].set_xlabel("Training \n Sample #", rotation=0)
            if i != 2:
                ax[i, j].set_xticks([])


            # Cosmetics
            ax[i, 0].set_ylabel(r"$\alpha_\#$", rotation=90, labelpad=20)
            if j != 0: 
                ax[i, j].set_yticks([])
            
            for d in range(0, 10):
                sorted_array = sorted(coefs[d], reverse=True, key=abs)
                
                symmetric_array = sorted_array[::-2]
                symmetric_array = np.concatenate((symmetric_array, [sorted_array[0]]), axis=0)
                symmetric_array = np.concatenate((symmetric_array, sorted_array[2::2]), axis=0)
            
                # The main plot
                ax[i, j].plot(np.arange(20*d, 20*(d+1), 1), symmetric_array, marker=".", linewidth=2, ms=3)

            # For comparison's sake a sensible fixed yaxis range.
            ax[i, j].set_ylim(-0.15, 0.4)

            

            
# The sub-figure showing the label
def label_coefs_fig(ax):
    label_range = np.arange(0, 10, 1) # List of digit classes
    for i in range(0, 3):
        for j in range(0, 3):
            
            Ls = data_Lf[i, j, index[i, j]] # Fetching the relevant digit labels

            # Main plot, the line and the colored points
            ax[i, j].plot(label_range, Ls[choices[i, j, 0]], marker="", color="k", linewidth=1)
            ax[i, j].scatter(label_range, Ls[choices[i, j, 0]], marker=".", c=label_range, cmap="tab10", s=200, alpha=1)

            # Cosmetics
            ax[-1, j].set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
            ax[i, j].set_yticks([-1, 1])

            ax[-1, j].set_xlabel("Training \n Class", rotation=00, labelpad=20)          
            ax[i, 0].set_ylabel("Label \n Coefficients", rotation=90, labelpad=20)
            
            if i != 2:
                ax[i, j].set_xticks([])

            if j != 0: 
                ax[i, j].set_yticks([])
            


"""

AAAABBBBCCCC....KKKKLLLLMMMM#
AAAABBBBCCCC....KKKKLLLLMMMM#
AAAABBBBCCCC....KKKKLLLLMMMM#
AAAABBBBCCCC....KKKKLLLLMMMM#
DDDDEEEEFFFF....NNNNOOOOPPPP#
DDDDEEEEFFFF....NNNNOOOOPPPP#
DDDDEEEEFFFF....NNNNOOOOPPPP#
DDDDEEEEFFFF....NNNNOOOOPPPP#
HHHHIIIIJJJJ....QQQQRRRRSSSS#
HHHHIIIIJJJJ....QQQQRRRRSSSS#
HHHHIIIIJJJJ....QQQQRRRRSSSS#
HHHHIIIIJJJJ....QQQQRRRRSSSS#
.............................
.............................
.............................
TTTTUUUUVVVV$................
TTTTUUUUVVVV$................
TTTTUUUUVVVV$................
TTTTUUUUVVVV$................
WWWWXXXXYYYY$................
WWWWXXXXYYYY$................
WWWWXXXXYYYY$................
WWWWXXXXYYYY$................
ZZZZ11112222$................
ZZZZ11112222$................
ZZZZ11112222$................
ZZZZ11112222$................
"""

# Putting it all together
         
fig = plt.figure(figsize=(50, 242-195+1))
axs = fig.subplot_mosaic(
"""   
............................AAAAAAABBBBBBBCCCCCCC%
............................AAAAAAABBBBBBBCCCCCCC%
............................AAAAAAABBBBBBBCCCCCCC%
............................AAAAAAABBBBBBBCCCCCCC%
............................AAAAAAABBBBBBBCCCCCCC%
............................AAAAAAABBBBBBBCCCCCCC%
............................AAAAAAABBBBBBBCCCCCCC%
............................DDDDDDDEEEEEEEFFFFFFF%
............................DDDDDDDEEEEEEEFFFFFFF%
............................DDDDDDDEEEEEEEFFFFFFF%
............................DDDDDDDEEEEEEEFFFFFFF%
............................DDDDDDDEEEEEEEFFFFFFF%
............................DDDDDDDEEEEEEEFFFFFFF%
............................DDDDDDDEEEEEEEFFFFFFF%
............................HHHHHHHIIIIIIIJJJJJJJ%
............................HHHHHHHIIIIIIIJJJJJJJ%
............................HHHHHHHIIIIIIIJJJJJJJ%
............................HHHHHHHIIIIIIIJJJJJJJ%
............................HHHHHHHIIIIIIIJJJJJJJ%
............................HHHHHHHIIIIIIIJJJJJJJ%
............................HHHHHHHIIIIIIIJJJJJJJ%
..................................................
..................................................
..................................................
..................................................
..................................................
..................................................
KKKKKKKLLLLLLLMMMMMMM#......TTTTTTTUUUUUUUVVVVVVV$
KKKKKKKLLLLLLLMMMMMMM#......TTTTTTTUUUUUUUVVVVVVV$
KKKKKKKLLLLLLLMMMMMMM#......TTTTTTTUUUUUUUVVVVVVV$
KKKKKKKLLLLLLLMMMMMMM#......TTTTTTTUUUUUUUVVVVVVV$
KKKKKKKLLLLLLLMMMMMMM#......TTTTTTTUUUUUUUVVVVVVV$
KKKKKKKLLLLLLLMMMMMMM#......TTTTTTTUUUUUUUVVVVVVV$
KKKKKKKLLLLLLLMMMMMMM#......TTTTTTTUUUUUUUVVVVVVV$
NNNNNNNOOOOOOOPPPPPPP#......WWWWWWWXXXXXXXYYYYYYY$
NNNNNNNOOOOOOOPPPPPPP#......WWWWWWWXXXXXXXYYYYYYY$
NNNNNNNOOOOOOOPPPPPPP#......WWWWWWWXXXXXXXYYYYYYY$
NNNNNNNOOOOOOOPPPPPPP#......WWWWWWWXXXXXXXYYYYYYY$
NNNNNNNOOOOOOOPPPPPPP#......WWWWWWWXXXXXXXYYYYYYY$
NNNNNNNOOOOOOOPPPPPPP#......WWWWWWWXXXXXXXYYYYYYY$
NNNNNNNOOOOOOOPPPPPPP#......WWWWWWWXXXXXXXYYYYYYY$
QQQQQQQRRRRRRRSSSSSSS#......ZZZZZZZ11111112222222$
QQQQQQQRRRRRRRSSSSSSS#......ZZZZZZZ11111112222222$
QQQQQQQRRRRRRRSSSSSSS#......ZZZZZZZ11111112222222$
QQQQQQQRRRRRRRSSSSSSS#......ZZZZZZZ11111112222222$
QQQQQQQRRRRRRRSSSSSSS#......ZZZZZZZ11111112222222$
QQQQQQQRRRRRRRSSSSSSS#......ZZZZZZZ11111112222222$
QQQQQQQRRRRRRRSSSSSSS#......ZZZZZZZ11111112222222$

"""
)

# The top left sub-plot
mem_ax_tl = [[axs['A'], axs['B'], axs['C']], [axs['D'], axs['E'], axs['F']], [axs['H'], axs['I'], axs['J']]]
mem_ax_tl = np.asarray(mem_ax_tl)
memory_samples_fig(mem_ax_tl, isBotAxis=False)

# The top right mem-coefs subplot
mem_coef_ax_tr = [[axs['K'], axs['L'], axs['M']], [axs['N'], axs['O'], axs['P']], [axs['Q'], axs['R'], axs['S']]]
mem_coef_ax_tr = np.asarray(mem_coef_ax_tr)
memory_coefs_fig(mem_coef_ax_tr)

# The bot left label-coefs subplot
label_coef_ax_bl = [[axs['T'], axs['U'], axs['V']], [axs['W'], axs['X'], axs['Y']], [axs['Z'], axs['1'], axs['2']]] 
label_coef_ax_bl= np.asarray(label_coef_ax_bl)
label_coefs_fig(label_coef_ax_bl)

# Column and row labels
mem_ax_tl[0, 1].text(28, -25, "n-power", fontsize=fontsize, ha='center', family='Times New Roman')
mem_ax_tl[1, 0].text(-35, 2*28, "Temperature", rotation=90, fontsize=fontsize, ha='center', family='Times New Roman')

wpad = 0.21




for char in ['#', '$']:
    ax_cb_class = axs[char]
    tab10_cmap = matplotlib.cm.tab10
    tab10_norm = matplotlib.colors.Normalize(vmin=0, vmax=10)
    cb_class = matplotlib.colorbar.ColorbarBase(ax_cb_class, cmap=tab10_cmap, norm=tab10_norm, orientation='vertical')
    cb_class.set_ticks(np.arange(0, 10, 1) + 0.5) # Finally found how to center these things 
    cb_class.set_ticklabels(np.arange(0, 10, 1))
    cb_class.set_label("Digit class", labelpad=20)

for char in ['%']:
    ax_cb_class = axs[char]
    bwr_cmap = matplotlib.cm.bwr
    bwr_norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
    cb_class = matplotlib.colorbar.ColorbarBase(ax_cb_class, cmap=bwr_cmap, norm=bwr_norm, orientation='vertical')
    cb_class.set_label("Pixel value", labelpad=20)


# Might remove the letters all together
alphabet = ['B', 'C', 'D', 'E', 'F', 'G', 'H']
for i, char in enumerate(['A', 'K', 'T']):
    axs[char].text(-0.35, 1.2, alphabet[i], transform=axs[char].transAxes, fontsize=83, verticalalignment='bottom', ha='right', fontfamily='Times New Roman', fontweight='bold')




# Cosmetics and saving with transparency.
plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.savefig("Figure-0.png")#, transparent=True)


