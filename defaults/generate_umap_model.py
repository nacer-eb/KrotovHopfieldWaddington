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

reducer = umap.UMAP(random_state=4, n_neighbors=55, min_dist=0.05, metric='correlation')
mapper = reducer.fit(data_T)

pickle.dump(reducer, open("umap_model_correlation.sav", 'wb'))
