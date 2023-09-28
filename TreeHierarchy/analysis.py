import numpy as np
import matplotlib.pyplot as plt

data_dir = "data/[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]/"


data = np.load(data_dir + "trained_net_n30_T650.npz")

threshold = 0.5
timestep = 200

tmax, N_mem, tmp = np.shape(data['M'])

tmax=2000

state = np.zeros((tmax//timestep, N_mem)) + 1

fig, ax = plt.subplots(1, 1, figsize=(16, 9))

t_range = np.arange(1, tmax, timestep)
for t_i, t in enumerate(t_range):
    print(t)
    for m in range(N_mem):
        if max(data['L'][t, m]) > threshold:
            state[t_i, m] = np.argmax(data['L'][t, m])
        else:
            state[t_i, m] = state[t_i-1, m]

        ax.plot([state[t_i-1, m], state[t_i, m]], [-(t_i - 1), -(t_i)], color="black", alpha=0.1)

    ax.scatter(state[t_i], [-t_i]*N_mem, c=state[t_i], cmap="tab10", vmin=0, vmax=10)

plt.savefig("tmp.png")

