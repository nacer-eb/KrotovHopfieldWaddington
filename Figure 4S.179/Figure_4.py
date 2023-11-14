import sys
sys.path.append('../')

import numpy as np

import matplotlib.pyplot as plt
import matplotlib

import pickle
import umap

from main_module.KrotovV2_utils import *

import matplotlib
fontsize = 35
font = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : fontsize}
matplotlib.rc('font', **font)


dataset = "../defaults/miniBatchs_images.npy"
umap_model_path = "../defaults/umap_model_correlation.sav"

class Figure_4_panel:
    def __init__(self, n, temp=800, data_dir="data/", selected_digits = [1, 7, 9], t_s = [20, 200, 250, 300, 344] ):

        self.n = n
        self.temp = temp
        self.data_dir = data_dir
        self.selected_digits = selected_digits

        self.subdir = str(selected_digits) + "/" 

        self.saving_dir = self.data_dir+self.subdir+"trained_net_n"+str(self.n)+"_T"+str(self.temp)+".npz"


        # Loading data - will improve dir struct soon..
        self.data_M = np.load(self.saving_dir)['M']
        self.data_L = np.load(self.saving_dir)['L']
        self.data_T = np.load(dataset)[0]


        self.mapper = pickle.load((open(umap_model_path, 'rb')))

        self.embedding = self.mapper.transform(self.data_T)
        self.M_embedding = np.load(self.data_dir+self.subdir+"/memory_umap_embed_correlation_n"+str(self.n)+"_T"+str(self.temp)+".npy")
    
        
        self.indices = np.zeros(5*3)
        for i in range(len(self.indices)):
            strictness = 0.99
            all_indices = np.argwhere(self.data_L[-1, :, self.selected_digits[i//5]] >= strictness )
            while len(all_indices) == 0:
                strictness -= 0.1
                all_indices = np.argwhere(self.data_L[-1, :, self.selected_digits[i//5]] >= strictness )
                print(self.selected_digits[i//5], len(all_indices), strictness)
        
        
    
            self.indices[i] =  all_indices[np.random.randint(len(all_indices))] # -> Pick randomly when Label is mostly # digit class i//2 $
        self.indices = np.asarray(self.indices, dtype=int)

        # Use the time stamps file or manually get timestamps from UMAP movies.
        # Handpicked, notice this is /10 because of the UMAP timesteps
        self.t_s = t_s
        self.t_samples = np.asarray(self.t_s)*10

        self.props = dict(boxstyle='round', facecolor='whitesmoke', alpha=0.5)


    def memory_sample_plot(self, ax):
        tmax = len(self.data_M)
            
        for t_i, t in enumerate(self.t_samples):
            im = ax[t_i].imshow(merge_data(self.data_M[t, self.indices], 5, 3), cmap="bwr", vmin=-1, vmax=1, aspect=5.0/4.0) # Plotting the selected memory samples 
            ax[t_i].set_title(r"Epoch "+str(t), pad=30, fontsize=fontsize, bbox=self.props) # Time stamps / Cosmetics
            ax[t_i].axis('off')



    def UMAP_plot(self, ax):


        M = len(self.data_T)
        keys = np.zeros((M))
        for d in range(0, 10):
            keys[d*M//10:(d+1)*M//10] = d
            
        tmax, N_mem = np.shape(self.M_embedding)[0], np.shape(self.M_embedding)[1]
        
        for t_i in range(0, len(self.t_s)-1):
            im = ax[t_i].scatter(self.embedding[:, 0], self.embedding[:, 1], c=keys, cmap="tab10", s=10, marker="*") # Plotting the UMAP training data
            
            for i in range(0, N_mem-1):
                data_pnts = self.M_embedding[self.t_s[0]:self.t_s[t_i+1], i, :]
                ax[t_i].plot(data_pnts[:, 0], data_pnts[:, 1], linewidth=1, alpha=0.3, color="k") # Plotting the trajectories of the memories on UMAP
                
                
            # Time stamps / Cosmetics
            ax[t_i].text(0.95, 0.95, r"Epoch "+str(self.t_s[t_i+1]*10), transform=ax[t_i].transAxes, fontsize=fontsize, verticalalignment='top', ha='right', bbox=self.props)
            
            # Plotting the memories as white points for each time point.
            ax[t_i].plot(self.M_embedding[self.t_s[t_i+1], :, 0], self.M_embedding[self.t_s[t_i+1], :, 1], marker="o",
                         linestyle="", alpha=1, markeredgecolor="k", markerfacecolor="white", markeredgewidth=1, markersize=4)
            
            # Cosmetics
            if t_i != 0 and t_i != 2:
                ax[t_i].set_yticks([])
                
            if t_i < 2:
                ax[t_i].set_xticks([])

            # Labels / cosmetics
            ax[0].set_ylabel("UMAP 2"); ax[2].set_ylabel("UMAP 2"); ax[2].set_xlabel("UMAP 1"); ax[3].set_xlabel("UMAP 1")



    def split_plot(self, ax, t_start, t_stop):
        tmax, N_max, tmp = np.shape(self.data_M)
            
        # Invert using 1, 7, 9 only 
        train_mask = np.zeros(200)
        for d in self.selected_digits:
            train_mask[d*20: (d+1)*20] = 1
        train_mask = np.asarray(train_mask, dtype=bool)

        tmax, N_max, tmp = np.shape(self.data_M)
        data_coefs = np.sum((self.data_M@np.linalg.pinv(self.data_T[train_mask])).reshape(tmax, N_max, len(self.selected_digits), 20), axis=-1)

    
        tab10_cmap = matplotlib.cm.tab10
        tab10_norm = matplotlib.colors.Normalize(vmin=0, vmax=10)
        
        for n in range(N_max):
            ax[0].plot(data_coefs[t_start:t_stop, n, 0], data_coefs[t_start:t_stop, n, 1], c=tab10_cmap(tab10_norm(np.argmax(self.data_L[-1, n]))), lw=2)
            ax[0].scatter(data_coefs[t_start, n, 0], data_coefs[t_start, n, 1], marker=".", color="k", s=400)
            
            
            ax[1].plot(data_coefs[t_start:t_stop, n, 0], data_coefs[t_start:t_stop, n, 2], c=tab10_cmap(tab10_norm(np.argmax(self.data_L[-1, n]))), lw=2)
            ax[1].scatter(data_coefs[t_start, n, 0], data_coefs[t_start, n, 2], marker=".", color="k", s=400)        
            
            
            
        ax[0].set_xlabel(r"$\bar{\alpha}_"+str(self.selected_digits[0])+"$"); ax[1].set_xlabel(r"$\bar{\alpha}_"+str(self.selected_digits[0])+"$");
        ax[0].set_ylabel(r"$\bar{\alpha}_"+str(self.selected_digits[1])+"$"); ax[1].set_ylabel(r"$\bar{\alpha}_"+str(self.selected_digits[2])+"$");

        
fig = plt.figure(figsize=(31, 38))#(figsize=(25, 15.7*2))
axs = fig.subplot_mosaic("""
...............................
...............................
AAAAAAAA.BBBBBBBB.1111112222220
AAAAAAAA.BBBBBBBB.1111112222220
AAAAAAAA.BBBBBBBB.1111112222220
AAAAAAAA.BBBBBBBB.1111112222220
AAAAAAAA.BBBBBBBB.3333334444440
AAAAAAAA.BBBBBBBB.3333334444440
AAAAAAAA.BBBBBBBB.3333334444440
AAAAAAAA.BBBBBBBB.3333334444440
...............................
...............................
aaaaaabbbbbbccccccddddddeeeeeex
aaaaaabbbbbbccccccddddddeeeeeex
aaaaaabbbbbbccccccddddddeeeeeex
...............................
...............................
CCCCCCCC.DDDDDDDD.555555666666O
CCCCCCCC.DDDDDDDD.555555666666O
CCCCCCCC.DDDDDDDD.555555666666O
CCCCCCCC.DDDDDDDD.555555666666O
CCCCCCCC.DDDDDDDD.777777888888O
CCCCCCCC.DDDDDDDD.777777888888O
CCCCCCCC.DDDDDDDD.777777888888O
CCCCCCCC.DDDDDDDD.777777888888O
...............................
...............................
ffffffgggggghhhhhhiiiiiijjjjjjX
ffffffgggggghhhhhhiiiiiijjjjjjX
ffffffgggggghhhhhhiiiiiijjjjjjX
...............................
""")

panel_top = Figure_4_panel(30, t_s=[20, 44, 107, 239, 343]) # use the umap movies to select the time samples

ax_mem_1 = np.asarray([axs['a'], axs['b'], axs['c'], axs['d'], axs['e']])
panel_top.memory_sample_plot(ax_mem_1)

ax_UMAP_1 = np.asarray([axs['1'], axs['2'], axs['3'], axs['4']])
panel_top.UMAP_plot(ax_UMAP_1)

ax_split_1 = np.asarray([axs['A'], axs['B']])
panel_top.split_plot(ax_split_1, 440, 2000)

ax_cb_UMAP = axs['0']
tab10_cmap = matplotlib.cm.tab10
tab10_norm = matplotlib.colors.Normalize(vmin=0, vmax=10)
cb_UMAP = matplotlib.colorbar.ColorbarBase(ax_cb_UMAP, cmap=tab10_cmap, norm=tab10_norm, orientation='vertical')
cb_UMAP.set_ticks(np.arange(0, 10, 1) + 0.5) # Finally found how to center these things 
cb_UMAP.set_ticklabels(np.arange(0, 10, 1))
cb_UMAP.set_label("Digit class")


ax_cb_mem = axs['x']
bwr_cmap = matplotlib.cm.bwr
bwr_norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
cb_mem = matplotlib.colorbar.ColorbarBase(ax_cb_mem, cmap=bwr_cmap, norm=bwr_norm, orientation='vertical')
cb_mem.set_label("Pixel value")


# The n text
ax_split_1[1].text(0.5, 1.1, r"n = 30", transform=ax_split_1[1].transAxes, fontsize=fontsize*1.3, rotation=0, verticalalignment='bottom', ha='center', bbox=panel_top.props)

########################################

panel_bot = Figure_4_panel(3, t_s=[20, 22, 30, 50, 344]) # 147

ax_mem_2 = np.asarray([axs['f'], axs['g'], axs['h'], axs['i'], axs['j']])
panel_bot.memory_sample_plot(ax_mem_2)

ax_UMAP_2 = np.asarray([axs['5'], axs['6'], axs['7'], axs['8']])
panel_bot.UMAP_plot(ax_UMAP_2)

ax_split_2 = np.asarray([axs['C'], axs['D']])
panel_bot.split_plot(ax_split_2, 0, 1400)

ax_cb_UMAP_2 = axs['O']
cb_UMAP_2 = matplotlib.colorbar.ColorbarBase(ax_cb_UMAP_2, cmap=tab10_cmap, norm=tab10_norm, orientation='vertical')
cb_UMAP_2.set_ticks(np.arange(0, 10, 1) + 0.5) # Finally found how to center these things 
cb_UMAP_2.set_ticklabels(np.arange(0, 10, 1))
cb_UMAP_2.set_label("Digit class")


ax_cb_mem_2 = axs['X']
cb_mem_2 = matplotlib.colorbar.ColorbarBase(ax_cb_mem_2, cmap=bwr_cmap, norm=bwr_norm, orientation='vertical')
cb_mem_2.set_label("Pixel value")

# The n text
ax_split_2[1].text(0.5, 1.1, r"n = 3", transform=ax_split_2[1].transAxes, fontsize=fontsize*1.3, rotation=0, verticalalignment='bottom', ha='center', bbox=panel_top.props)


alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
rx = [5.0/8.0, 1.0, 5.0/8.0, 1.0]
ry = [3.0/8.0, 1.0, 3.0/8.0, 1.0]
for i, char in enumerate(['A', 'a', 'C', 'f']):
    axs[char].text(-0.3*rx[i], 1.0+0.1*ry[i], alphabet[i], transform=axs[char].transAxes, fontsize=83, verticalalignment='bottom', ha='right', fontfamily='Times New Roman', fontweight='bold')


rx = 32.0/25.0
plt.subplots_adjust(top=0.99, bottom=0.01, wspace=2.0, hspace=0.8)
plt.savefig("Figure_4S_tmp.png")
