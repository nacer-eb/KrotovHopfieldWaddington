import sys
sys.path.append('../')

import numpy as np

from main_module.KrotovV2 import *

import matplotlib.animation as anim

r = 1.0/10.0**(5)
temp = 800


data_dir_1 = "FirstDegree/"
data_dir_2 = "SecondDegree/"


isFirstRun = True
    
tmax = 5000
n = 30

data_1 = np.load(data_dir_1+"FPs.npy")
data_2 = np.load(data_dir_2+"FPs.npy")

print(np.shape(data_1))
print(np.shape(data_2))

fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot()

ax.scatter(data_1[:, :, 0].ravel(), data_1[:, :, 1].ravel(), alpha=(data_1[:, :, 0].ravel() + data_1[:, :, 1].ravel() + data_1[:, :, 2].ravel()), s=30, c="black")
ax.scatter(data_2[:, :, 0, 0].ravel(), data_2[:, :, 0, 1].ravel(), alpha=(data_2[:, :, 0, 0].ravel() + data_2[:, :, 0, 1].ravel() + data_2[:, :, 0, 2].ravel()), s=30, c="red")



plt.show()

