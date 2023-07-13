import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../')

from main_module.KrotovV2 import *
from main_module.KrotovLinearAnalysis import *


class simplified_1_2_2_system:
    def __init__(self, temp, n, m, filename=None, filename_minibatchs_images=None):

        T=None
        if filename_minibatchs_images is not None:
            T = np.load(filename_minibatchs_images)
                
        if filename_minibatchs_images is None:
            data = np.load(filename_network)
            T = data["miniBatchs_images"]

        if T is None:
            print("Error, no information about minibatchs")
            exit()

        # The dots <A|A> and <B|B>
        self.d_AA = T[0, 0] @ T[0, 0]
        self.d_AB = T[0, 0] @ T[0, 1]
        self.d_BB = T[0, 1] @ T[0, 1]
        print(self.d_AA, self.d_AB,  self.d_BB)

        
        # Model paramters
        self.temp = temp
        self.n = n
        self.m = m

    def ReLU(self, x):
        return (np.abs(x) + x)/2.0
        
    def calc_d_A(self, alpha, beta):
        return self.ReLU( (alpha*self.d_AA + beta*self.d_AB)/self.temp )

    def calc_d_B(self, alpha, beta):
        return self.ReLU( (alpha*self.d_AB + beta*self.d_BB)/self.temp )

    # This is sometimes useful
    def calc_O_A(self, alpha, beta, l_0):
        d_A = self.calc_d_A(alpha, beta)

        return np.tanh(l_0 * d_A)

    def calc_O_B(self, alpha, beta, l_0):
        d_B = self.calc_d_B(alpha, beta)

        return np.tanh(l_0 * d_B)
    
    
    def tilde_dt_l_0(self, alpha, beta, l_0):
        d_A = self.calc_d_A(alpha, beta)
        d_B = self.calc_d_B(alpha, beta)
            
        A1 = (1 - np.tanh( l_0 * (d_A)**self.n ) )**(2*self.m-1)
        A2 = (1 - (np.tanh( l_0 * (d_A)**self.n ))**2 )
        
        B1 = (1 + np.tanh( l_0*(d_B)**self.n ) )**(2*self.m-1)
        B2 = (1 - (np.tanh( l_0*(d_B)**self.n ))**2 )
        
        return A1 * A2 * d_A**self.n  -  B1 * B2 * d_B**self.n

    def tilde_dt_l_gamma(self, alpha, beta, l_0):
        d_A = self.calc_d_A(alpha, beta)
        d_B = self.calc_d_B(alpha, beta)
            
        A1 = (1 - np.tanh( (d_A)**self.n ) )**(2*self.m-1)
        A2 = (1 - (np.tanh( (d_A)**self.n ))**2 )
        
        B1 = (1 - np.tanh( (d_B)**self.n ) )**(2*self.m-1)
        B2 = (1 - (np.tanh( (d_B)**self.n ))**2 )

        return  -A1 * A2 * d_A**self.n  -  B1 * B2 * d_B**self.n


    def tilde_dt_alpha(self, alpha, beta, l_0):
        d_A = self.calc_d_A(alpha, beta)
        
        A1 = (1 - np.tanh( l_0*(d_A)**self.n ) )**(2*self.m-1)
        A2 = (1 - (np.tanh( l_0*(d_A)**self.n ))**2 )
        
        B1 = (1 - np.tanh( (d_A)**self.n ) )**(2*self.m-1)
        B2 = (1 - (np.tanh( (d_A)**self.n ))**2 )
        
        return l_0 * A1 * A2 * d_A**(self.n-1)  +  4 * B1 * B2 * d_A**(self.n-1)
    

    def tilde_dt_beta(self, alpha, beta, l_0):
        d_B = self.calc_d_B(alpha, beta)
        
        A1 = (1 + np.tanh( l_0*(d_B)**self.n ) )**(2*self.m-1)
        A2 = (1 - (np.tanh( l_0*(d_B)**self.n ))**2 )
        
        B1 = (1 - np.tanh( (d_B)**self.n ) )**(2*self.m-1)
        B2 = (1 - (np.tanh( (d_B)**self.n ))**2 )
        
        return -l_0 * A1 * A2 * d_B**(self.n-1)  +  4 * B1 * B2 * d_B**(self.n-1)
    
    
    def simulate(self, alpha_0, beta_0, l_0, tmax, rate):

        # Initializing the simulation
        # Simluation paramters
        self.tmax = tmax
        self.rate = rate

        # Defining our arrays
        self.alpha, self.beta, self.l_0 = np.zeros((3, tmax))

        # Initial conditions
        self.alpha[0], self.beta[0], self.l_0[0] = alpha_0, beta_0, l_0

        #Starting the simulation
        for t in range(1, self.tmax):
            delta_l_0 = self.tilde_dt_l_0(self.alpha[t-1], self.beta[t-1], self.l_0[t-1])
            delta_l_gamma = self.tilde_dt_l_gamma(self.alpha[t-1], self.beta[t-1], self.l_0[t-1])
            
            self.l_0[t] = self.l_0[t-1] + self.rate * (delta_l_0/np.maximum(np.abs(delta_l_gamma), np.abs(delta_l_0) ))
            self.l_0[t] = np.clip(self.l_0[t], -1, 1)
            
            delta_alpha = self.tilde_dt_alpha(self.alpha[t-1], self.beta[t-1], self.l_0[t-1])
            delta_beta = self.tilde_dt_beta(self.alpha[t-1], self.beta[t-1], self.l_0[t-1])
                
            self.alpha[t] = self.alpha[t-1] + self.rate * ( delta_alpha / ( np.abs(delta_alpha) + np.abs(delta_beta) ) )
            self.beta[t] = self.beta[t-1] + self.rate * ( delta_beta / ( np.abs(delta_alpha) + np.abs(delta_beta) ) )
            
            N = np.maximum( np.abs(self.alpha[t]) + np.abs(self.beta[t]), 1)
            self.alpha[t] /= N
            self.beta[t] /= N

            
    # Can only be called if simulate has been called!!!
    def saveSimulation(self, savefile):
        print("Saving...")
        np.savez_compressed(savefile, l_0=self.l_0, alpha=self.alpha, beta=self.beta)
        
