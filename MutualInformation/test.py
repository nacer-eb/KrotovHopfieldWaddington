import numpy as np
import time
from multiprocessing import Pool

def f(a):
    n, T = a
    time.sleep(3)
    print(n, T)


if __name__ == '__main__':

    n_range = np.arange(1, 5)
    temp_range = np.asarray([550, 650, 750])

    n, T = np.meshgrid(n_range, temp_range)
    print(np.shape(n))

    nT_merge = np.asarray([n.flatten(), T.flatten()]).T
    
    t1 = time.time()
    with Pool(5) as p:
        r = p.map_async(f, nT_merge)
        r.wait()
    t2 = time.time()

    print(t2-t1)
