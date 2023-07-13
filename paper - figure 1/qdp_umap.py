import sys
sys.path.append('../')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim

import pickle


import umap
import umap.plot

from main_module.KrotovV2_utils import *

data_dir = "data_100_10_200/"
n, temp = 30, 670




"""
data_T = get_MNIST_train_images() #np.load(data_dir+"miniBatchs_images.npy")[0]
keys = get_MNIST_train_labels()
M = len(data_T)

reducer = umap.UMAP(random_state=4, n_neighbors=55, min_dist=0.05, metric='correlation')
mapper = reducer.fit(data_T)

fig, ax = plt.subplots(1, 1, figsize=(16, 9))

plt.scatter(mapper.embedding_[:, 0], mapper.embedding_[:, 1], c=keys, cmap="tab10", s=1)
plt.colorbar()
plt.show()


pickle.dump(reducer, open(data_dir+"umap_model_4.sav", 'wb'))

#exit()
"""
"""

umap.plot.connectivity(mapper, show_points=True,  labels=keys, cmap="tab10", width=1920, height=1080)#, edge_bundling='hammer')edge_cmap="tab10", 
umap.plot.plt.show()
exit()

"""


data_T = np.load(data_dir+"miniBatchs_images.npy")[0]
mapper = pickle.load((open(data_dir+"/umap_model_correlation.sav", 'rb')))


M = len(data_T)
keys = np.zeros((M))
for d in range(0, 10):
    keys[d*M//10:(d+1)*M//10] = d


embedding = mapper.transform(data_T)


fig, ax = plt.subplots(1, 1, figsize=(16, 9))

plt.scatter(embedding[:, 0], embedding[:, 1], c=keys, cmap="tab10", s=10, marker="*")
plt.colorbar()




TIME_SKIP = 10
run = 0
saving_dir = data_dir+"momentum_run_"+str(run)+"_n"+str(n)+"_T"+str(temp)+".npz"
data_M = np.load(saving_dir)['M'][::TIME_SKIP]
tmax, N_mem = np.shape(data_M)[0], np.shape(data_M)[1]

print(tmax, N_mem)
#"""

t = 0
t_slice_size = 10
M_embedding = mapper.transform(data_M[t*t_slice_size:(t+1)*t_slice_size].reshape(t_slice_size*N_mem, 784)).reshape(t_slice_size, N_mem, 2)
                                                                                       
for t in range(1, tmax//t_slice_size):
    tmp_embed = mapper.transform(data_M[t*t_slice_size:(t+1)*t_slice_size].reshape(t_slice_size*N_mem, 784)).reshape(t_slice_size, N_mem, 2)
    M_embedding = np.concatenate((M_embedding, tmp_embed), axis=0)
    print(t, "out of", tmax//t_slice_size)
np.save(data_dir+"/memory_umap_embed_correlation_n"+str(n)+"_momentum", M_embedding)
#"""

M_embedding = np.load(data_dir+"/memory_umap_embed_correlation_n"+str(n)+"_momentum.npy")

t = 0 
pts, = ax.plot(M_embedding[t, :, 0], M_embedding[t, :, 1], marker=".", markersize=5, linestyle="", color="k")
text = ax.text(0.05, 0.95, "t=", transform=ax.transAxes, fontsize=14, verticalalignment='top')
def update(t):
    print(t)

    t_sub = t
    
    pts.set_data(M_embedding[t+1, :, 0], M_embedding[t+1, :, 1])

    text.set_text("t="+str(t)) 
    return pts, text,

ani = anim.FuncAnimation(fig, update, frames=(len(M_embedding)-1), interval=100, blit=True, repeat=False)
ani.save(data_dir+"tmp"+str(n)+"_momentum.mp4", writer="ffmpeg")

exit()



    
