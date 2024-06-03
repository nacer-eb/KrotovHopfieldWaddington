import sys
sys.path.append('../')

from main_module.KrotovV2 import *

import numpy as np
import matplotlib.pyplot as plt
import matplotlib


class splittingDynamics:
    def __init__(self, delta_alpha, delta_ell, n, T, alpha, ell, A_dot_A, A_dot_B, B_dot_B):
        self.delta_alpha, self.delta_ell, self.n, self.T, self.alpha, self.ell = delta_alpha, delta_ell, n, T, alpha, ell

        self.A_dot_A, self.A_dot_B, self.B_dot_B = A_dot_A, A_dot_B, B_dot_B


        # Getting quantities d(alpha +- delta_alpha)/dt and d(ell +- delta_ell)/dt
        self.d_alpha_p_delta_alpha_dt = self.calc_d_alpha_pm_delta_alpha_dt(+1)
        self.d_alpha_m_delta_alpha_dt = self.calc_d_alpha_pm_delta_alpha_dt(-1)

        self.d_ell_p_delta_ell_dt = self.calc_d_ell_pm_delta_ell_dt(+1)
        self.d_ell_m_delta_ell_dt = self.calc_d_ell_pm_delta_ell_dt(-1)


        # Getting d(alpha)/dt, d(ell)/dt as well as d(delta_alpha)/dt and d(delta_ell)/dt
        self.d_alpha_dt = ( self.d_alpha_p_delta_alpha_dt + self.d_alpha_m_delta_alpha_dt  )/2.0
        self.d_ell_dt = ( self.d_ell_p_delta_ell_dt + self.d_ell_m_delta_ell_dt  )/2.0

        self.d_delta_alpha_dt = ( self.d_alpha_p_delta_alpha_dt - self.d_alpha_m_delta_alpha_dt  )/2.0
        self.d_delta_ell_dt = ( self.d_ell_p_delta_ell_dt - self.d_ell_m_delta_ell_dt  )/2.0
        

    # Near a non-trivial FP, ReLU is a bit useless...
    def ReLU(self, x):
        return (np.abs(x) + x)/2.0
        
    def calc_nabla_A(self, pm):

        # Fetching class data
        delta_alpha, delta_ell, n, T, alpha, ell = self.delta_alpha, self.delta_ell, self.n, self.T, self.alpha, self.ell
        A_dot_A, A_dot_B, B_dot_B = self.A_dot_A, self.A_dot_B, self.B_dot_B
        
        beta = 1 - alpha
        
        M_dot_A = (alpha*A_dot_A + beta*A_dot_B)
        
        d_M_dot_A_d_alpha = (A_dot_A - A_dot_B)
        
        l_A_o_A = np.tanh(ell * ( M_dot_A/T)**n )
        l_gamma_o_A = -np.tanh(( M_dot_A/T)**n )
        
        nabla_A_pm = ( (1 - l_A_o_A)**(2*n - 1)  *  (1 - l_A_o_A**2)  *  M_dot_A**(n-1)  *  ell
                       
                       + 4 * (1 + l_gamma_o_A)**(2*n-1)  *  (1 - l_gamma_o_A**2)  *  M_dot_A**(n-1)
                       
                       + pm * (n-1) * (1 - l_A_o_A)**(2*n - 1)  *  (1 - l_A_o_A**2)  *  M_dot_A**(n-2) * d_M_dot_A_d_alpha * ell * delta_alpha
                       
                       + pm * 4 * (n-1) * (1 + l_gamma_o_A)**(2*n - 1)  *  (1 - l_gamma_o_A**2)  *  M_dot_A**(n-2) * d_M_dot_A_d_alpha * delta_alpha
                       
                       + pm * (1 - l_A_o_A)**(2*n - 1)  *  (1 - l_A_o_A**2)  *  M_dot_A**(n-1) * delta_ell
        )

        return nabla_A_pm
    


    def calc_nabla_B(self, pm):

        # Fetching class data
        delta_alpha, delta_ell, n, T, alpha, ell = self.delta_alpha, self.delta_ell, self.n, self.T, self.alpha, self.ell
        A_dot_A, A_dot_B, B_dot_B = self.A_dot_A, self.A_dot_B, self.B_dot_B
        
        beta = 1 - alpha
        
        M_dot_B = (alpha*A_dot_B + beta*B_dot_B)
        
        d_M_dot_B_d_alpha = (A_dot_B - B_dot_B)
        
        l_A_o_B = np.tanh(ell * ( M_dot_B/T)**n )
        l_gamma_o_B = -np.tanh(( M_dot_B/T)**n )
        
        nabla_B_pm = ( - (1 + l_A_o_B)**(2*n - 1)  *  (1 - l_A_o_B**2)  *  M_dot_B**(n-1)  *  ell
                       
                       + 4 * (1 + l_gamma_o_B)**(2*n-1)  *  (1 - l_gamma_o_B**2)  *  M_dot_B**(n-1)
                       
                       - pm * (n-1) * (1 - l_A_o_B)**(2*n - 1)  *  (1 - l_A_o_B**2)  *  M_dot_B**(n-2) * d_M_dot_B_d_alpha * ell * delta_alpha
                       
                       + pm * 4 * (n-1) * (1 + l_gamma_o_B)**(2*n - 1)  *  (1 - l_gamma_o_B**2)  *  M_dot_B**(n-2) * d_M_dot_B_d_alpha * delta_alpha
                       
                       - pm * (1 + l_A_o_B)**(2*n - 1)  *  (1 - l_A_o_B**2)  *  M_dot_B**(n-1) * delta_ell
        )
        
        return nabla_B_pm
    
    

    def calc_nabla_ell(self, pm):

        # Fetching class data
        delta_alpha, delta_ell, n, T, alpha, ell = self.delta_alpha, self.delta_ell, self.n, self.T, self.alpha, self.ell
        A_dot_A, A_dot_B, B_dot_B = self.A_dot_A, self.A_dot_B, self.B_dot_B
        
        beta = 1 - alpha
        
        M_dot_A = (alpha*A_dot_A + beta*A_dot_B)
        M_dot_B = (alpha*A_dot_B + beta*B_dot_B)
        
        
        d_M_dot_A_d_alpha = (A_dot_A - A_dot_B)
        d_M_dot_B_d_alpha = (A_dot_B - B_dot_B)
        
        
        l_A_o_A = np.tanh(ell * ( M_dot_A/T)**n )
        l_gamma_o_A = -np.tanh(( M_dot_A/T)**n )
        
        l_A_o_B = np.tanh(ell * ( M_dot_B/T)**n )
        l_gamma_o_B = -np.tanh(( M_dot_B/T)**n )
        
        
        nabla_ell_pm = ( (1 - l_A_o_A)**(2*n - 1)   *   (1 - l_A_o_A**2) * M_dot_A**n
                         
                         - (1 + l_A_o_B)**(2*n - 1)   *   (1 - l_A_o_B**2) * M_dot_B**n
                         
                         + pm * n * (1 - l_A_o_A)**(2*n - 1)   *   (1 - l_A_o_A**2) * M_dot_A**(n-1) * d_M_dot_A_d_alpha * delta_alpha
                         
                         - pm * n * (1 + l_A_o_B)**(2*n - 1)   *   (1 - l_A_o_B**2) * M_dot_B**(n-1) * d_M_dot_B_d_alpha * delta_alpha
                         
        )
        
        return nabla_ell_pm
    

    def calc_nabla_gamma(self):

        # Fetching class data
        n, T, alpha, ell = self.n, self.T, self.alpha, self.ell
        A_dot_A, A_dot_B, B_dot_B = self.A_dot_A, self.A_dot_B, self.B_dot_B
        
        beta = 1 - alpha
        
        M_dot_A = (alpha*A_dot_A + beta*A_dot_B)
        M_dot_B = (alpha*A_dot_B + beta*B_dot_B)
        
        l_gamma_o_A = -np.tanh(( M_dot_A/T)**n )
        l_gamma_o_B = -np.tanh(( M_dot_B/T)**n )
        
        
        nabla_gamma = ( - (1 + l_gamma_o_A)**(2*n - 1)  *  (1 - l_gamma_o_A**2) * M_dot_A**n
                        - (1 + l_gamma_o_B)**(2*n - 1)  *  (1 - l_gamma_o_B**2) * M_dot_B**n
        )
        

        return nabla_gamma




    def calc_d_alpha_pm_delta_alpha_dt(self, pm):

        # Fetching class data
        delta_alpha, delta_ell, n, T, alpha, ell = self.delta_alpha, self.delta_ell, self.n, self.T, self.alpha, self.ell
        A_dot_A, A_dot_B, B_dot_B = self.A_dot_A, self.A_dot_B, self.B_dot_B
        
        nabla_A_pm = self.calc_nabla_A(pm)
        nabla_B_pm = self.calc_nabla_B(pm)
        
        nabla_A = self.calc_nabla_A(0)
        nabla_B = self.calc_nabla_B(0)
        
        d_alpha_pm_delta_alpha_dt = (  nabla_A_pm - alpha*(nabla_A_pm + nabla_B_pm) - pm * delta_alpha * (nabla_A + nabla_B)  ) / ( np.abs(nabla_A_pm) + np.abs(nabla_B_pm) )
        
        return d_alpha_pm_delta_alpha_dt


    def calc_d_ell_pm_delta_ell_dt(self, pm):

        # Fetching class data
        delta_alpha, delta_ell, n, T, alpha, ell = self.delta_alpha, self.delta_ell, self.n, self.T, self.alpha, self.ell
        A_dot_A, A_dot_B, B_dot_B = self.A_dot_A, self.A_dot_B, self.B_dot_B

        
        nabla_ell_pm = self.calc_nabla_ell(pm)
        nabla_gamma = self.calc_nabla_gamma()
        
        d_ell_pm_delta_ell_dt = nabla_ell_pm / np.maximum( np.abs(nabla_ell_pm), np.abs(nabla_gamma) )
        
        return d_ell_pm_delta_ell_dt
    


