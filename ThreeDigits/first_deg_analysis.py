import sys
sys.path.append('../')

import numpy as np

from main_module.KrotovV2 import *

import matplotlib.animation as anim

r = 1.0/10.0**(5)
temp = 800


data_dir = "FirstDegree/"


# Makes sure the data_dir exits else creates it.
if not path.exists(data_dir):
    print(data_dir, "Does not exist. It will be created ...")
    os.mkdir(data_dir)
    print(data_dir, "Created!")


isFirstRun = True
    
tmax = 5000
n = 30
data = np.zeros((10, 10, 5000, 1, 3))
if isFirstRun:
    for a_i, a in enumerate(np.arange(0, 1, 0.1)):
        for b_i, b in enumerate(np.arange(0, 1-a, 0.1)):           
            data_T = np.load("miniBatchs_images.npy")[0];
            
            # Initial Conditions
            A, B, C = data_T
            
            saving_dir=data_dir+"n"+str(n)+"_a"+str(a_i)+"_b"+str(b_i)+"_save.npz"
            data[a_i, b_i] = np.load(saving_dir)['M']@np.linalg.pinv(data_T)
        print(a_i)        
    np.save(data_dir + "FPs.npy", data[:, :, -1, 0, :])
    
#exit()
fig, ax = plt.subplots(1, 1, figsize=(16, 9))

t = -1
im, = ax.plot(data[:, :, t, 0, 0].ravel(), data[:, :, t, 0, 1].ravel(), linestyle="", marker=".", ms=20)

def update(t):
    im.set_data(data[:, :, t, 0, 0].ravel(), data[:, :, t, 0, 1].ravel())

    return im,

ani = anim.FuncAnimation(fig, update, frames=tmax, blit=True, interval=20)
plt.show()

exit()


