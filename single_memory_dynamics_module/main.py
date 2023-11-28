import sys
sys.path.append("../")

from single_memory_dynamics import *

from main_module.KrotovV2 import *



training_datafile = "../defaults/miniBatchs_images_Fig_5.npy"

data_T = np.load(training_datafile)[0]
data_T_inv = np.linalg.pinv(data_T)
A, B = data_T

selected_digits=[1, 4]

## Simulation 
n, temp = 15, 640
tmax, dt = 3500, 0.001

net = KrotovNet(Kx=1, Ky=1, n_deg=n, m_deg=n, M=2, nbMiniBatchs=1, momentum=0, rate=dt, temp=temp, rand_init_std=0.0001, selected_digits=selected_digits)
net.miniBatchs_images[0] = data_T
net.hiddenDetectors[:, [0, 2, 3, 5, 6, 7, 8, 9]] = -1


isRandomInit = False
if isRandomInit:
    alpha_A_init = net.visibleDetectors[0]@data_T_inv[:, 0]
    alpha_B_init = net.visibleDetectors[0]@data_T_inv[:, 1]
    ell_init = (net.hiddenDetectors[0, selected_digits[0]] + net.hiddenDetectors[0, selected_digits[1]])/2.0


if not isRandomInit:

    # Change this by hand
    alpha_A_init = 0.1
    alpha_B_init = 0.2
    ell_init = 0.1

    net.visibleDetectors[0] = alpha_A_init*A + alpha_B_init*B
    net.hiddenDetectors[0, selected_digits[0]] = ell_init
    net.hiddenDetectors[0, selected_digits[1]] = -ell_init


net.train_plot_update(tmax, isPlotting=False, isSaving=True, saving_dir="./simulation.npz")


## Analytics Verification
smd = single_memory_dynamics(A@A, A@B, B@B, n, temp)
smd.simulate_and_save(alpha_A_init, alpha_B_init, ell_init, tmax, dt, "./analytics.npz")


# Plotting
data_simulation = np.load("simulation.npz")
data_analytics = np.load("analytics.npz")


fig, ax = plt.subplots(1, 2, figsize=(20, 9), dpi=200)
ax[0].plot(data_simulation['L'][:, 0, selected_digits[0]], color="orange", linewidth=10, alpha=0.7)
ax[0].plot(data_analytics['ells'][:], color="darkorange", linewidth=3)

ax[0].set_ylabel(r"$\ell$")
ax[0].set_xlabel("epoch")


ax[1].plot(data_simulation['M'][:, 0]@data_T_inv[:, 0], color="cyan", linewidth=10, alpha=0.7)
ax[1].plot(data_analytics['alpha_As'][:], color="blue", linewidth=3)

ax[1].plot(data_simulation['M'][:, 0]@data_T_inv[:, 1], color="red", linewidth=10, alpha=0.7)
ax[1].plot(data_analytics['alpha_Bs'][:], color="darkred", linewidth=3)

ax[1].set_ylabel(r"$\alpha$")
ax[1].set_xlabel("epoch")


plt.subplots_adjust(left=0.055, bottom=0.096, right=0.96, top=0.913, wspace=0.272, hspace=0.215)
plt.savefig("AnalyticsVerification.png")

