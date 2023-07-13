import sys
sys.path.append('../')

import numpy as np

import matplotlib

from main_module.KrotovV2 import *
from main_module.KrotovV2_utils import *

data_dir = "data_1_10_200_179_long/"#"data_1_10_200_47_900_precise/"

for noise_r in [5]:
    for temp in [800]:#np.arange(400, 680, 20):

        #fig, ax = plt.subplots(1, 1)
        """
            EEFF.AAAAAAAA
            EEFF.AAAAAAAA
            .....AAAAAAAA
            .DD..AAAAAAAA
            .DD..AAAAAAAA
            .....AAAAAAAA
            BBCC.AAAAAAAA
            BBCC.AAAAAAAA
        """
        fig = plt.figure()
        axs = fig.subplot_mosaic(
            """
            FFFGGG.AAAAAAAA
            FFFGGG.AAAAAAAA
            ..EEE..AAAAAAAA
            ..EEE..AAAAAAAA
            .DDD...AAAAAAAA
            .DDD...AAAAAAAA
            BBBCCC.AAAAAAAA
            BBBCCC.AAAAAAAA
            """
        )
        
        ax_main = axs['A']

        ax_L = np.asarray([axs['B'], axs['C'], axs['D'], axs['E'], axs['F'], axs['G']])
        

        

        
        for run in [1, 7]:

            n_range = np.arange(2, 61, 0.1)[:]
            data_Ms = np.zeros((len(n_range), 784))

            alphas = np.zeros((len(n_range)))
            
            for n_i, n in enumerate(n_range):
                saving_dir = data_dir+"run_"+str(run)+"_n"+str(n)+"_T"+str(temp)+".npz"
                
                data = np.load(saving_dir)
                data_M = data['M']
                
                data_Ms[n_i] = data_M
                
                data_L = data['L']
                data_T = data['miniBatchs_images'][0]
                data_T_inv = np.linalg.pinv(data_T)
                coefs = np.sum(((data_M@data_T_inv).reshape(len(data_M), 3, 20)), axis=-1)

                alphas[n_i] = coefs[0, 0]/np.sum(np.abs(coefs[0]))
                print(n_i, n)
                
            norm = matplotlib.colors.Normalize(vmin=0, vmax=9)
            tab10 = matplotlib.cm.get_cmap('tab10')
                       
            
            ax_main.scatter(alphas, n_range, s=20, color=tab10(norm(run)), alpha=0.4)

            ax_main.set_xlabel(r"$\alpha_1$"); ax_main.set_ylabel(r"$n$")
            
            
            print(n_range[20], n_range[210], n_range[360], n_range[480])
            if run == 7:
                ax_L[0].imshow(data_Ms[20].reshape(28, 28), cmap="bwr", vmin=-1, vmax=1)
                ax_L[2].imshow(data_Ms[210].reshape(28, 28), cmap="bwr", vmin=-1, vmax=1)
                ax_L[3].imshow(data_Ms[360].reshape(28, 28), cmap="bwr", vmin=-1, vmax=1)
                ax_L[4].imshow(data_Ms[480].reshape(28, 28), cmap="bwr", vmin=-1, vmax=1)

            if run == 1:
                ax_L[1].imshow(data_Ms[20].reshape(28, 28), cmap="bwr", vmin=-1, vmax=1)
                ax_L[5].imshow(data_Ms[480].reshape(28, 28), cmap="bwr", vmin=-1, vmax=1)

                
            
        for ax in ax_L:
            ax.set_xticks([]); ax.set_yticks([])
        

        
        from matplotlib.lines import Line2D
        custom_lines = [Line2D([0], [0], color=plt.cm.tab10(0.7), marker=".", linestyle="", ms=13, alpha=0.4),
                        Line2D([0], [0], color=plt.cm.tab10(0.1), marker=".", linestyle="", ms=13, alpha=0.4)]
        
        fig.legend(custom_lines, ['Initialized near 7', 'Initialized near 1'], loc='upper right', ncol=4)
            
        plt.subplots_adjust(top=0.9, bottom=0.11, left=0.16, right=0.84, hspace=0.2, wspace=0.2)
        plt.show()

