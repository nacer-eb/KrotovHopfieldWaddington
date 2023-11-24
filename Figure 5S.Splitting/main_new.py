import sys
sys.path.append('../')

from main_module.KrotovV2 import *
from nullcline_gather.GatherNullClines import *

import numpy as np
import matplotlib.pyplot as plt
import matplotlib


def calc_nabla_A(delta_alpha, delta_ell, n, T, alpha, ell, pm):
    beta = 1 - alpha

    M_dot_A = (alpha*A_dot_A + beta*A_dot_B)/T

    d_M_dot_A_d_alpha = (A_dot_A - A_dot_B)/T
    
    l_A_o_A = np.tanh(ell * ( M_dot_A)**n )
    l_gamma_o_A = -np.tanh(( M_dot_A)**n )

    nabla_A_pm = ( (1 - l_A_o_A)**(2*n - 1)  *  (1 - l_A_o_A**2)  *  M_dot_A**(n-1)  *  ell

                   + 4 * (1 + l_gamma_o_A)**(2*n-1)  *  (1 - l_gamma_o_A**2)  *  M_dot_A**(n-1)

                   + pm * (n-1) * (1 - l_A_o_A)**(2*n - 1)  *  (1 - l_A_o_A**2)  *  M_dot_A**(n-2) * d_M_dot_A_d_alpha * ell * delta_alpha

                   + pm * 4 * (n-1) * (1 + l_gamma_o_A)**(2*n - 1)  *  (1 - l_gamma_o_A**2)  *  M_dot_A**(n-2) * d_M_dot_A_d_alpha * delta_alpha

                   + pm * (1 - l_A_o_A)**(2*n - 1)  *  (1 - l_A_o_A**2)  *  M_dot_A**(n-1) * delta_ell
    )

    return nabla_A_pm
    


def calc_nabla_B(delta_alpha, delta_ell, n, T, alpha, ell, pm):
    beta = 1 - alpha

    M_dot_B = (alpha*A_dot_B + beta*B_dot_B)/T

    d_M_dot_B_d_alpha = (A_dot_B - B_dot_B)/T
    
    l_A_o_B = np.tanh(ell * ( M_dot_B)**n )
    l_gamma_o_B = -np.tanh(( M_dot_B)**n )

    nabla_B_pm = ( - (1 + l_A_o_B)**(2*n - 1)  *  (1 - l_A_o_B**2)  *  M_dot_B**(n-1)  *  ell

                   + 4 * (1 + l_gamma_o_B)**(2*n-1)  *  (1 - l_gamma_o_B**2)  *  M_dot_B**(n-1)

                   - pm * (n-1) * (1 - l_A_o_B)**(2*n - 1)  *  (1 - l_A_o_B**2)  *  M_dot_B**(n-2) * d_M_dot_B_d_alpha * ell * delta_alpha

                   + pm * 4 * (n-1) * (1 + l_gamma_o_B)**(2*n - 1)  *  (1 - l_gamma_o_B**2)  *  M_dot_B**(n-2) * d_M_dot_B_d_alpha * delta_alpha

                   - pm * (1 + l_A_o_B)**(2*n - 1)  *  (1 - l_A_o_B**2)  *  M_dot_B**(n-1) * delta_ell
    )

    return nabla_B_pm
    
    

def calc_nabla_ell(delta_alpha, delta_ell, n, T, alpha, ell, pm):
    beta = 1 - alpha

    M_dot_A = (alpha*A_dot_A + beta*A_dot_B)/T
    M_dot_B = (alpha*A_dot_B + beta*B_dot_B)/T

    
    d_M_dot_A_d_alpha = (A_dot_A - A_dot_B)/T
    d_M_dot_B_d_alpha = (A_dot_B - B_dot_B)/T

        
    l_A_o_A = np.tanh(ell * ( M_dot_A)**n )
    l_gamma_o_A = -np.tanh(( M_dot_A)**n )
    
    l_A_o_B = np.tanh(ell * ( M_dot_B)**n )
    l_gamma_o_B = -np.tanh(( M_dot_B)**n )


    nabla_ell_pm = ( (1 - l_A_o_A)**(2*n - 1)   *   (1 - l_A_o_A**2) * M_dot_A**n

                     - (1 + l_A_o_B)**(2*n - 1)   *   (1 - l_A_o_B**2) * M_dot_B**n

                     + pm * n * (1 - l_A_o_A)**(2*n - 1)   *   (1 - l_A_o_A**2) * M_dot_A**(n-1) * d_M_dot_A_d_alpha * delta_alpha

                     - pm * n * (1 + l_A_o_B)**(2*n - 1)   *   (1 - l_A_o_B**2) * M_dot_B**(n-1) * d_M_dot_B_d_alpha * delta_alpha
        
    )

    return nabla_ell_pm


def calc_nabla_gamma(n, T, alpha, ell):
    beta = 1 - alpha

    M_dot_A = (alpha*A_dot_A + beta*A_dot_B)/T
    M_dot_B = (alpha*A_dot_B + beta*B_dot_B)/T
    
    l_gamma_o_A = -np.tanh(( M_dot_A)**n )
    l_gamma_o_B = -np.tanh(( M_dot_B)**n )


    nabla_gamma = ( - (1 + l_gamma_o_A)**(2*n - 1)  *  (1 - l_gamma_o_A**2) * M_dot_A**n
                    - (1 + l_gamma_o_B)**(2*n - 1)  *  (1 - l_gamma_o_B**2) * M_dot_B**n
    )


    return nabla_gamma




def calc_d_alpha_pm_delta_alpha_dt(delta_alpha, delta_ell, n, T, alpha, ell, pm):
    nabla_A_pm = calc_nabla_A(delta_alpha, delta_ell, n, T, alpha, ell, pm)
    nabla_B_pm = calc_nabla_B(delta_alpha, delta_ell, n, T, alpha, ell, pm)

    nabla_A = calc_nabla_A(delta_alpha, delta_ell, n, T, alpha, ell, 0)
    nabla_B = calc_nabla_B(delta_alpha, delta_ell, n, T, alpha, ell, 0)
        
    d_alpha_pm_delta_alpha_dt = (  nabla_A_pm - alpha*(nabla_A_pm + nabla_B_pm) - pm * delta_alpha * (nabla_A + nabla_B)  ) / ( np.abs(nabla_A_pm) + np.abs(nabla_B_pm) )

    return d_alpha_pm_delta_alpha_dt


def calc_d_ell_pm_delta_ell_dt(delta_alpha, delta_ell, n, T, alpha, ell, pm):

    nabla_ell_pm = calc_nabla_ell(delta_alpha, delta_ell, n, T, alpha, ell, pm)
    nabla_gamma = calc_nabla_gamma(n, T, alpha, ell)
    
    d_ell_pm_delta_ell_dt = nabla_ell_pm / np.maximum( np.abs(nabla_ell_pm), np.abs(nabla_gamma) )

    return d_ell_pm_delta_ell_dt
    

ell, alpha = 0.038, 0.715

delta_alpha_1d = np.linspace(0, 1, 500)
delta_ell_1d = np.linspace(-0.5, 0.5, 500)
ell, alpha = np.meshgrid(delta_ell_1d, delta_alpha_1d)

delta_alpha, delta_ell = 0, 0


A_dot_A, A_dot_B, B_dot_B = 753, 494, 719
n = 15
T = 670/(2.0**(1.0/n))


d_alpha_p_delta_alpha_dt = calc_d_alpha_pm_delta_alpha_dt(delta_alpha, delta_ell, n, T, alpha, ell, 0)
d_alpha_m_delta_alpha_dt = calc_d_alpha_pm_delta_alpha_dt(delta_alpha, delta_ell, n, T, alpha, ell, 0) 

d_delta_alpha_dt = (d_alpha_p_delta_alpha_dt + 0*d_alpha_m_delta_alpha_dt)/1.0


d_ell_p_delta_ell_dt = calc_d_ell_pm_delta_ell_dt(delta_alpha, delta_ell, n, T, alpha, ell, 0)
d_ell_m_delta_ell_dt = calc_d_ell_pm_delta_ell_dt(delta_alpha, delta_ell, n, T, alpha, ell, 0)

d_delta_ell_dt = (d_ell_p_delta_ell_dt + 0*d_ell_m_delta_ell_dt)/1.0



fig, ax = plt.subplots(1, 1, figsize=(16, 9))


ax.streamplot(ell, alpha, d_delta_ell_dt, d_delta_alpha_dt, color=(0, 0, 0, 0.2), density=3)




sample_range = np.linspace(0, 1, 1000)
l_0_mesh, alpha_mesh = np.meshgrid(sample_range*2-1, sample_range)

GNC = GatherNullClines(A_dot_A, A_dot_B, B_dot_B, n, T, +1)
ax.contour(l_0_mesh, alpha_mesh, GNC.alpha_nullcline(alpha_mesh, l_0_mesh), [0], colors="purple", linewidths=7, alpha=0.5)
ax.contour(l_0_mesh, alpha_mesh, GNC.l_0_nullcline(alpha_mesh, l_0_mesh), [0], colors="orange", linewidths=7, alpha=0.5)




plt.show()

