# This is used to import modules in other directories
import sys
sys.path.append('../')


import numpy as np

# Importing the main code for the KrotovHopfield network
from main_module.KrotovV2 import *

import matplotlib

# Network main parameters
n, temp = 30, 670
selected_digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Generally I save data in ./data/selected_digits/trained_net_n$n$_T$T$.npz
# Hence I create the relevant directories if they don't exist
data_dir = "data/"
prefix = str(selected_digits) + "/"



# networks can be generated as:
net = KrotovNet(Kx=5, Ky=5, # The number of memories in your network is Kx*Ky (Kx and Ky are mostly for plotting, their multiplication is what matters)
                n_deg=n, m_deg=n, # The n hyperparameter - n_deg is the power on the dot product, m_deg is the power on the cost function (m_deg = n_deg usually)
                nbMiniBatchs=1, # You can divide your training data into miniBatchs if so specify the number of miniBatchs (here I chose only 1)
                M=20, # This is the number of training samples per miniBatch
                momentum=0*0.6, # This is the usual ML momentum, not really earthshattering, we don't use it in the paper
                rate=0.005, # The training rate; I recommend less than 0.1 (at the very most) - miniBatchs also increase this artificially (more training steps)
                temp=temp, # The temperature of the system
                rand_init_mean=-0.003, rand_init_std=1.0/10.0**5, # noise, gaussian based
                selected_digits=selected_digits) # The digits you want to include in training


# This shows the training data generated
net.show_minibatchs()

# Training the system
net.train_plot_update(3500, # Number of training epochs (the optimal number will depend on n and training rate)
                      isPlotting=False, # Do you want to plot while training (as a preview)
                      isSaving=True, # Do you want to save
                      saving_dir=data_dir+prefix+"trained_net_n"+str(n)+"_T"+str(temp)+".npz", # Where do you want to save?
                      testFreq=500) # Originally this was the frequency at which you wanted to compute the accuracy, it might have changed
