import argparse
import sys
sys.path.append('../')

import numpy as np

import matplotlib.pyplot as plt
import matplotlib

from main_module.KrotovV2 import *
from main_module.KrotovV2_utils import *

import matplotlib
fontsize = 20
font = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : fontsize}
matplotlib.rc('font', **font)


data_dir = "data/"

# The digit classes to include in the training
selected_digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] 
prefix = str(selected_digits)+"/"


n, temp = 30, 670

fig = plt.figure(layout="constrained", figsize=(18, 10))
axs = fig.subplots(3, 3, sharex=True, sharey=True)


for ax in axs[-1]:
    ax.set_xlabel("Training epoch")

for ax in axs[:, 0]:
    ax.set_ylabel("Classification accuracy")
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels(['0', '50%', '100%'])



tab10_cmap = matplotlib.cm.tab10
tab10_norm = matplotlib.colors.Normalize(vmin=0, vmax=10)

saving_dir = data_dir+prefix+"trained_net_n"+str(n)+"_T"+str(temp)+"_run"+str(0)+".npz"
data = np.load(saving_dir)
data_M = data['M']
data_T = data['miniBatchs_images'][0]

max_epoch = data_M.shape[0]
epoch_range = np.arange(1, max_epoch, 10)

net = KrotovNet(M=len(data_T), nbMiniBatchs=1) # The rest will be filled in by the next line load-net
net.load_net(saving_dir, epoch=0)
for i in range(9):
    classification_score = np.zeros((len(epoch_range), 10))

    ax = axs.ravel()[i]
    ax.text(0.02, 0.95, chr(ord('A')+i), transform=ax.transAxes, fontsize=30, verticalalignment='top', ha='left')

    data = np.load(data_dir+prefix+"trained_net_n"+str(n)+"_T"+str(temp)+"_run"+str(i)+".npz")
    data_M = data['M']
    data_L = data['L']
    data_T = data['miniBatchs_images'][0]

    for t_i, epoch in enumerate(epoch_range):
        print(i, epoch)
        net.visibleDetectors = data_M[epoch]
        net.hiddenDetectors = data_L[epoch]

        for m_i in range(len(data_T)):
            output = net.compute(data_T[m_i])

            if np.max(output) > -1.0:
                classification_score[t_i, m_i//20] += np.argmax(output)==(m_i//20) # 20 examples per class, if it's right add one else don't

    ax.stackplot(epoch_range, classification_score.T/200.0, colors=tab10_cmap(tab10_norm(np.arange(0, 10, 1))), alpha=0.7)

    ax.set_yticks([0, 0.5, 1.0])
    ax.set_xlim(1, max(epoch_range))

plt.savefig("layout_3F.pdf")
