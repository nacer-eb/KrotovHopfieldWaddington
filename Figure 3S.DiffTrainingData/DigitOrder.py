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

isFirstRun = False
if isFirstRun:
    saving_dir = data_dir+prefix+"trained_net_n"+str(n)+"_T"+str(temp)+"_run"+str(0)+".npz"
    data = np.load(saving_dir)
    data_M = data['M']
    data_T = data['miniBatchs_images'][0]

    max_epoch = data_M.shape[0]
    epoch_range = np.arange(1, max_epoch, 10)

    
    classification_scores = np.zeros((100, len(epoch_range), 10))
    net = KrotovNet(M=len(data_T), nbMiniBatchs=1) # The rest will be filled in by the next line load-net
    net.load_net(saving_dir, epoch=0)

    for i in range(100):
        print("Simulation", i)
        classification_score = classification_scores[i]
        
        data = np.load(data_dir+prefix+"trained_net_n"+str(n)+"_T"+str(temp)+"_run"+str(i)+".npz")
        data_M = data['M']
        data_L = data['L']
        data_T = data['miniBatchs_images'][0]

        for t_i, epoch in enumerate(epoch_range):
            print("Simulation", i, "Epoch", epoch)
            net.visibleDetectors = data_M[epoch]
            net.hiddenDetectors = data_L[epoch]

            for m_i in range(len(data_T)):
                output = net.compute(data_T[m_i])
                
                if np.max(output) > -1.0:
                    classification_score[t_i, m_i//20] += np.argmax(output)==(m_i//20) # 20 examples per class, if it's right add one else don't

    print("Saving...")
    np.save("classification_scores.npy", classification_scores)


classification_scores = np.load("classification_scores.npy")

threshold = 1*(classification_scores > 4)
crossing_point = threshold[:, 1:, :] - threshold[:, :-1, :]
crossing_times = np.argmax(crossing_point, axis=1)


digit_order = np.argsort(crossing_times, axis=-1)
print(digit_order)

for i in range(10):

    digits = np.asarray(list(set(digit_order[:, i])))
    print("1st digit is one of", digits, "(", end= ' ')

    for element in digits:
        frequency = 100*np.sum(digit_order[:, i]==element)/100.0
        print('{0:.2f}'.format(frequency), "%, ", end=' ')

    print(")")




mean_order = np.zeros((10))
std_order = np.zeros((10))
for d in range(10):
    mean_order[d] = np.mean(np.argmax(digit_order==d, axis=-1)+1)
    std_order[d] = np.std(np.argmax(digit_order==d, axis=-1))


indices = np.argsort(mean_order)

## v1
norm = matplotlib.colors.Normalize(vmin=0, vmax=10)
cmap = plt.get_cmap("tab10")

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
for d in indices:
    print("Digit", d, "appears", mean_order[d], r"$\pm$", std_order[d])

    ax.scatter(d, mean_order[d], c=cmap(norm(d)), s=35)
    ax.errorbar(d, mean_order[d], std_order[d], 0, ecolor=cmap(norm(d)), markersize=0, capsize=5, elinewidth=5, alpha=0.5)


ax.set_yticks(np.arange(0, 10, 1))
ax.set_xticks(np.arange(0, 10, 1, dtype=int))
ax.set_xticklabels(np.arange(0, 10, 1, dtype=int), weight = 'bold')

for xtick_i, xtick in enumerate(ax.get_xticklabels()):
    xtick.set_color(cmap(norm(xtick_i)))

ax.set_xlabel("Digit")
ax.set_ylabel("Order")

plt.savefig("DigitOrder.png")



for i in range(0, 16): #(100):
    print("Simulation", i)
    classification_score = classification_scores[i]
    
    data = np.load(data_dir+prefix+"trained_net_n"+str(n)+"_T"+str(temp)+"_run"+str(i)+".npz")
    data_M = data['M']
    data_L = data['L']
    data_T = data['miniBatchs_images'][0]

    M, N = data_T.shape

    indices = np.arange(0, 200, 1, dtype=int)
    indices_order = indices.reshape(10, 20)[digit_order[i], :].ravel()

    dot_products = data_T[indices_order]@data_T[indices_order].T

    fig, ax = plt.subplots(1, 1, layout="constrained")
    plt.imshow(dot_products, cmap="bwr")
    plt.colorbar()
    
    for d in range(10):
        for d_2 in range(10):
            plt.axhline(20*d)
            plt.axvline(20*d)


    ax.set_xticks(np.arange(0, 10, 1, dtype=int)*20 + 10)
    ax.set_xticklabels(digit_order[i])
    ax.set_yticks(np.arange(0, 10, 1, dtype=int)*20 + 10)
    ax.set_yticklabels(digit_order[i])
    plt.savefig("tmp"+str(i)+".png")
