import sys
sys.path.append('../')

import numpy as np

from main_module.KrotovV2 import *
from main_module.KrotovV2_utils import *

import umap
import pickle


data_T = get_MNIST_train_images()
keys = get_MNIST_train_labels()
M = len(data_T)

"""
test code - not useful

print(M)

size = 50000
index = np.zeros((size), dtype=int)
for i in range(0, 10):
    for d in range(10):
        index[d*size//10:(d+1)*size//10] = np.random.randint(d*6000, (d+1)*6000, size//10)
    print(index)
    
    coefs = np.random.randn(size)

    M_const = (coefs@data_T[index])

    print(np.max(np.abs(M_const))/np.sum((coefs)))
    
    M_const /= np.max(np.abs(M_const))

    

    fig, ax = plt.subplots(1, 2, figsize=(16, 9))
    ax[0].set_title(np.linalg.matrix_rank(data_T[index]))

    ax[0].imshow(M_const.reshape(28, 28), cmap="bwr", vmin=-1, vmax=1)
    ax[1].imshow(((M_const@np.linalg.pinv(data_T[index], rcond=1E-25))@data_T[index]).reshape(28, 28), cmap="bwr", vmin=-1, vmax=1)
    plt.show()

    print(np.mean(np.abs(M_const - ((M_const@np.linalg.pinv(data_T[index]))@data_T[index]))))
    print(np.std(np.abs(M_const - ((M_const@np.linalg.pinv(data_T[index]))@data_T[index]))))

exit()

"""

reducer = umap.UMAP(random_state=4, n_neighbors=55, min_dist=0.05, metric='correlation')
mapper = reducer.fit(data_T)

pickle.dump(reducer, open("umap_model_correlation.sav", 'wb'))
