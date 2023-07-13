import numpy as np
import matplotlib.pyplot as plt

import matplotlib.animation as anim

import matplotlib
font = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 14}

matplotlib.rc('font', **font)


def E(l_0, alpha, n, temp):
    d_AA, d_AB, d_BB = 753/temp, 494/temp, 719/temp

    l_0 = np.clip(l_0, -1, 1)
    alpha = np.clip(alpha, -1, 1)
    
    
    beta=1-np.abs(alpha)
    
    d_A = alpha * d_AA + beta * d_AB
    d_B = alpha * d_AB + beta * d_BB

    l_A_o_A = np.tanh(l_0 * d_A**n)
    l_A_o_B = np.tanh(l_0 * d_B**n)

    l_gamma_o_A = -np.tanh(d_A**n)
    l_gamma_o_B = -np.tanh(d_B**n)

    E = 2 * np.abs(1 - l_A_o_A)**(2*n) + 2 * np.abs(1 + l_A_o_B)**(2*n) + 8 * np.abs(1 + l_gamma_o_A)**(2*n) + 8 * np.abs(1 + l_gamma_o_B)**(2*n)

    return E


# Get data
data_dir = "C_code/"
data_files = ["low", "mid", "high"]
data_file = data_dir + data_files[0] + ".dat"
data = np.loadtxt(data_file, skiprows=1, delimiter=",")   
p_mask = (np.abs(data[:, -2]) > 0.001) * (np.abs(data[:, -3]) > 0.001 ) * (data[:, 1] <= 7) + (np.abs(data[:, -2]) > 0.03) * (np.abs(data[:, -3]) > 0.03 ) * (data[:, 1] > 7)
data = data[p_mask, :]


fig, ax = plt.subplots(1, 4)
for n_i, n in enumerate([20, 23, 24.2, 25.5]):
    n_mask = np.abs((data[:, 1]-n)) < 0.1

    #ax[n_i, 0].plot(data[n_mask, -1], data[n_mask, -3], linestyle="", marker=".", ms=10, c="k")

    if n_i == 0:
        pnts_x, pnts_y = data[n_mask, -1], data[n_mask, -3]

        
    pnts_x, pnts_y = data[n_mask, -1], data[n_mask, -3]
    
    p = np.polyfit(pnts_x, pnts_y, 2)
    pnts_x_detailed = np.linspace(np.min(pnts_x)-0.15, np.max(pnts_x)+0.15, 1000)
    pnts_y_detailed = np.polyval(p, pnts_x_detailed)
    #ax[n_i, 0].plot(pnts_x_detailed, pnts_y_detailed, lw=1, c="cyan")
    
    log_E = np.log(E(pnts_x_detailed, pnts_y_detailed, n, 700.0/(2.0**(1.0/n)) ))
    ax[n_i].plot(pnts_x_detailed, log_E, c="red", linestyle="", marker=".", ms=1)
    ax[n_i].set_ylim([1.25, 1.38, 1.4, 1.45][n_i], 1.5)

plt.subplots_adjust(top=0.7, bottom=0.3, left=0.05, right=0.95)
plt.show()
    

