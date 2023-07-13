import sys
sys.path.append('../')

from nullcline_gather.GatherNullClines import *

import matplotlib.pyplot as plt


GNC = GatherNullClines(729, 586, 723, 10, 800, +1)
sample_range = np.linspace(0, 1, 100)
l_0_mesh, alpha_mesh = np.meshgrid(sample_range*2-1, sample_range)


plt.contour(l_0_mesh, alpha_mesh, GNC.alpha_nullcline(alpha_mesh, l_0_mesh), [0], colors="orange", linewidths=7, alpha=0.5)
plt.contour(l_0_mesh, alpha_mesh, GNC.l_0_nullcline(alpha_mesh, l_0_mesh), [0], colors="purple", linewidths=7, alpha=0.5)
plt.show()
