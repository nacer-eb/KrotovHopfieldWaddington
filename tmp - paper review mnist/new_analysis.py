import numpy as np
import matplotlib.pyplot as plt

import matplotlib.animation as anim

data_images = (np.load("data_images.npy")+1.0)/2.00
data_labels = (np.load("data_labels.npy")+1.0)/2.0

data_dir = "data_long_2/"

A, B = data_images[0], data_images[1]
print(A@A, A@B, B@B)


A_p = (B@B * A - A@B * B)*1.0/(A@A*B@B-(A@B)**2)
B_p = -(A@B * A - A@A * B)*1.0/(A@A*B@B-(A@B)**2)


n_i = 784
tmax = 1000

alpha_range = np.arange(-2, 2, 0.02)

n_alpha = len(alpha_range)

M_alpha = np.zeros((n_alpha, tmax))
M_beta = np.zeros((n_alpha, tmax))

factors = np.zeros((n_alpha, tmax))

#ax.set_xlim(-2, 2)
#ax.set_ylim(-2, 2)
for alpha_i, alpha in enumerate(alpha_range):
    print(alpha_i, alpha)
    beta = np.sqrt(4.0 - alpha*alpha)
    
    save_file = data_dir + "alpha_"+str(alpha) + ".npz"
    data = np.load(save_file)
    
    W_21_all_t = data['W_21_all_t']
    W_32_all_t = data['W_32_all_t']
    
    
    factors[alpha_i] = np.sqrt(W_32_all_t[:, 0, 0]**2 + W_32_all_t[:, 1, 0]**2)
    
    
    M_alpha[alpha_i] = A_p@W_21_all_t[:, 0, :].T#*factor
    M_beta[alpha_i] = B_p@W_21_all_t[:, 0, :].T#*factor
    

    print(alpha_i)

    
fig, ax = plt.subplots(1, 1, figsize=(16, 9))
graph, = ax.plot(M_alpha[:, -1].reshape((n_alpha))*factors[:, -1].reshape((n_alpha)), M_beta[:, -1].reshape((n_alpha))*factors[:, -1].reshape((n_alpha)), linestyle="", marker="o", ms=1)


def update_graph(t_):
    t = t_
    print(t)
    graph.set_data(M_alpha[:, t].reshape((n_alpha))*factors[:, t].reshape((n_alpha)), M_beta[:, t].reshape((n_alpha))*factors[:, t].reshape((n_alpha)))

    return graph,

ani = anim.FuncAnimation(fig, update_graph, tmax//1, interval=10, blit=True)

plt.show()



