import sys
sys.path.append('../')

from nullcline_gather.GatherNullClines import *

import matplotlib.pyplot as plt


GNC = GatherNullClines(753, 494, 719, 15, 700, +1)
sample_range = np.linspace(0, 1, 1000)
l_0_mesh, alpha_mesh = np.meshgrid(sample_range*2-1, sample_range)


dt_ell, dt_ell_sat_p, dt_ell_sat_m = GNC.calc_d_ell_dt(alpha_mesh, l_0_mesh)

plt.contour(l_0_mesh, alpha_mesh, GNC.calc_d_alpha_dt(alpha_mesh, l_0_mesh), [0], colors="purple", linewidths=3, alpha=0.5)

plt.contour(l_0_mesh, alpha_mesh, dt_ell, [0], colors="orange", linewidths=3, alpha=0.5)
plt.contour(l_0_mesh, alpha_mesh, dt_ell_sat_p, [0], colors="orange", linewidths=3, alpha=0.5)
plt.contour(l_0_mesh, alpha_mesh, dt_ell_sat_m, [0], colors="orange", linewidths=3, alpha=0.5)
plt.show()
