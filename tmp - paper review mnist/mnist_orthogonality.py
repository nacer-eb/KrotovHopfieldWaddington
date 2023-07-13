import sys
sys.path.append('../')

import numpy as np
import matplotlib.pyplot as plt

from main_module.KrotovV2_utils import *

from sklearn.decomposition import PCA
import numpy.linalg as linalg

N = 200
data_images, data_labels = get_MNIST_train_partitionned(N,
                                                        get_MNIST_train_images(),
                                                        get_MNIST_train_labels(),
                                                        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])




data_images = (data_images+0.0)/1.0
data_labels = (data_labels+0.0)/1.0

plt.imshow(merge_data(data_images, 5, 4), cmap="bwr")
plt.show()

n_i = 784
#pca = PCA(whiten=True, n_components=n_i)

#data_images = pca.fit_transform(data_images)
"""

for i in range(0, n_i):
    x = np.zeros(n_i)-1
    x[i]=1
    plt.imshow(pca.inverse_transform(x).reshape(28, 28), cmap="bwr")
    plt.show()
"""

sigma_11 = np.zeros((n_i, n_i))
sigma_31 = np.zeros((10, n_i))

for i in range(0, N):
    if i % 100 == 0:
        print(i)
    x = np.expand_dims(data_images[i], -1)
    y = np.expand_dims(data_labels[i], -1)
    sigma_11 += x@x.T
    sigma_31 += y@x.T

u, s, v = linalg.svd(sigma_31)

print(np.shape(sigma_31))
print(np.shape(u))
print(np.shape(s))
print(np.shape(v))

plt.imshow(u, cmap="bwr")
plt.colorbar()
plt.show()

for i in range(0, 40):
    plt.imshow(v[:, i].reshape(28, 28), cmap="bwr")
    plt.colorbar()
    plt.show()

plt.imshow(v@sigma_11@v.T, cmap="bwr", vmax=200)
plt.colorbar()
plt.show()


