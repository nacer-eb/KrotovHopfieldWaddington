import numpy as np

class GatherNullClines:
    def __init__(self, D_AA, D_AB, D_BB, n, T, pm):
        self.D_AA = D_AA
        self.D_AB = D_AB
        self.D_BA = self.D_AB
        self.D_BB = D_BB

        self.T = T
        self.n = n
        self.m = n # by default.

        self.pm = pm

    def beta(self, alpha):
        if self.pm == 1:
            return 1 - np.abs(alpha)

        if self.pm == -1:
            return -1 + np.abs(alpha)

        # else
        print("Garbage in, garbage out. pm was set to", pm)
        exit(-1)

    def ReLU(self, x):
        return (np.abs(x) + x)/2.0

    
    def d_A(self, alpha):
        d_A = self.ReLU(alpha*self.D_AA + self.beta(alpha)*self.D_AB)/self.T

        return d_A


    def d_B(self, alpha):
        d_B = self.ReLU(alpha*self.D_BA + self.beta(alpha)*self.D_BB)/self.T

        return d_B


    def O_A(self, alpha, l_0):
        O_A = np.tanh(l_0 * ( self.d_A(alpha) )**self.n )

        return O_A


    def O_B(self, alpha, l_0):
        O_B = np.tanh(l_0 * ( self.d_B(alpha) )**self.n )

        return O_B

    

    def PN_dt_alpha(self, alpha, l_0):
        PN_dt_alpha = (l_0*(1.0 - self.O_A(alpha, l_0))**(2*self.m-1)  * (1.0 - (self.O_A(alpha, l_0))**2) * (self.d_A(alpha))**(self.n-1)
                       + 4.0 * (1.0 + self.O_A(alpha, -1) )**(2*self.m-1) * (1.0 - (self.O_A(alpha, -1) )**2) * (self.d_A(alpha))**(self.n-1) )

        return PN_dt_alpha

    
    def PN_dt_beta(self, alpha, l_0):
        # There might've been a mistake + self.OB should be -self.OB
        PN_dt_beta = (-l_0*(1.0 + self.O_B(alpha, l_0))**(2*self.m-1)  * (1.0 - (self.O_B(alpha, l_0))**2) * (self.d_B(alpha))**(self.n-1)
                       + 4 * (1.0 + self.O_B(alpha, -1) )**(2*self.m-1) * (1.0 - (self.O_B(alpha, -1) )**2) * (self.d_B(alpha))**(self.n-1) )

        return PN_dt_beta


    def PN_dt_l_gamma(self, alpha):
        O_gamma_A = self.O_A(alpha, -1.0)
        O_gamma_B = self.O_B(alpha, -1.0)
        
        return (( -1.0 - O_gamma_A )**(2*self.m-1) * ( 1.0 - O_gamma_A**2 ) * (self.d_A(alpha)) ** (self.n)
                 + ( -1.0 - O_gamma_B )**(2*self.m-1) * ( 1.0 - O_gamma_B**2 ) * (self.d_B(alpha)) ** (self.n))
    
    
    def PN_dt_l_0(self, alpha, l_0):
        PN_dt_l_0 = ((1.0 - self.O_A(alpha, l_0))**(2*self.m-1)  * (1.0 - (self.O_A(alpha, l_0))**2) * (self.d_A(alpha))**(self.n)
                     - (1.0 + self.O_B(alpha, l_0))**(2*self.m-1)  * (1.0 - (self.O_B(alpha, l_0))**2) * (self.d_B(alpha))**(self.n))
        
        return PN_dt_l_0


    def get_dt(self, alpha, l_0):
        PN_dt_alpha = self.PN_dt_alpha(alpha, l_0)
        PN_dt_beta = self.PN_dt_beta(alpha, l_0)

        interaction = (alpha/np.abs(alpha)) * PN_dt_alpha + (self.beta(alpha)/np.abs(self.beta(alpha)) ) * PN_dt_beta
        
        interaction = np.maximum(interaction, 0)
        
        
        dt_alpha = PN_dt_alpha - ( interaction )*alpha
        dt_beta = PN_dt_beta - ( interaction )*self.beta(alpha)
                
        
        norm = np.abs(PN_dt_alpha) + np.abs(PN_dt_beta)#, 1)

        dt_alpha /= norm
        dt_beta /= norm
        
        dt_l_0 = self.PN_dt_l_0(alpha, l_0)

        
        dt_l_0 /= np.maximum(np.abs(self.PN_dt_l_gamma(alpha)), np.abs(dt_l_0))
        
        return interaction*dt_l_0, interaction*dt_alpha
    

    # The alpha nullcine when == 0
    def alpha_nullcline(self, alpha, l_0):

        PN_dt_alpha = self.PN_dt_alpha(alpha, l_0)
        PN_dt_beta = self.PN_dt_beta(alpha, l_0)

        condition = np.sign(alpha) * PN_dt_alpha + np.sign(self.beta(alpha))*PN_dt_beta

        alpha_nullcline = PN_dt_alpha/alpha - PN_dt_beta/self.beta(alpha)
        
        #This may lead to messy plots; require to much precision :/
        #alpha_nullcline[condition < 0] = -np.max(np.abs(alpha_nullcline)) # If it doesn't satisfy the condition set it to a suuuper high value so it never returns a nullcline!
        
        
        return alpha_nullcline

    
        
    # The l_0 nullcline when == 0
    def l_0_nullcline(self, alpha, l_0):
        PN_dt_l_0 = self.PN_dt_l_0(alpha, l_0)

        l_0_nullcline = PN_dt_l_0
        
        return l_0_nullcline
