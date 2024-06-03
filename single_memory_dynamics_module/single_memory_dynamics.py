import numpy as np


class single_memory_dynamics:
    def __init__(self, A_dot_A, A_dot_B, B_dot_B, n, T):
        self.A_dot_A, self.A_dot_B, self.B_dot_B, self.n, self.T = A_dot_A, A_dot_B, B_dot_B, n, T
        self.m = n # by default. This done nothing, just to tell you this is assumed for the rest of  the code.


    def ReLU(self, x):
        return (np.abs(x) + x)/2.0
    
    def calc_nabla_A(self, alpha_A, alpha_B, ell):

        A_dot_A, A_dot_B, B_dot_B, n, T = self.A_dot_A, self.A_dot_B, self.B_dot_B, self.n, self.T
        
        M_dot_A = self.ReLU(alpha_A*A_dot_A + alpha_B*A_dot_B)
        
        l_A_o_A = np.tanh( ell * (M_dot_A/T)**n )
        l_gamma_o_A = -np.tanh( (M_dot_A/T)**n )

        nabla_A = ( (1 - l_A_o_A)**(2*n - 1)  *  (1 - l_A_o_A**2)  *  (M_dot_A/T)**(n-1) * ell

                         + 4 * (1 + l_gamma_o_A)**(2*n - 1)  *  (1 - l_gamma_o_A**2) *  (M_dot_A/T)**(n-1) 
        )

        return nabla_A

    def calc_nabla_B(self, alpha_A, alpha_B, ell):

        A_dot_A, A_dot_B, B_dot_B, n, T = self.A_dot_A, self.A_dot_B, self.B_dot_B, self.n, self.T


        M_dot_B = self.ReLU(alpha_A*A_dot_B + alpha_B*B_dot_B)
        
        l_A_o_B = np.tanh( ell * (M_dot_B/T)**n )
        l_gamma_o_B = -np.tanh( (M_dot_B/T)**n )

        
        nabla_B = ( -(1 + l_A_o_B)**(2*n - 1)  *  (1 - l_A_o_B**2)  *  (M_dot_B/T)**(n-1) * ell

                         + 4 * (1 + l_gamma_o_B)**(2*n - 1)  *  (1 - l_gamma_o_B**2) *  (M_dot_B/T)**(n-1) 
        )

        return nabla_B



    def calc_nabla_ell(self, alpha_A, alpha_B, ell):
        
        A_dot_A, A_dot_B, B_dot_B, n, T = self.A_dot_A, self.A_dot_B, self.B_dot_B, self.n, self.T
    

        M_dot_A = self.ReLU(alpha_A*A_dot_A + alpha_B*A_dot_B)
        M_dot_B = self.ReLU(alpha_A*A_dot_B + alpha_B*B_dot_B)

        l_A_o_A = np.tanh( ell * (M_dot_A/T)**n )
        l_A_o_B = np.tanh( ell * (M_dot_B/T)**n )


        nabla_ell = ( (1 - l_A_o_A)**(2*n - 1)  *  (1 - l_A_o_A**2) * (M_dot_A/T)**n
                      -  (1 + l_A_o_B)**(2*n - 1)  *  (1 - l_A_o_B**2) * (M_dot_B/T)**n
            
        )


        return nabla_ell

    def calc_nabla_gamma(self, alpha_A, alpha_B, ell):

        A_dot_A, A_dot_B, B_dot_B, n, T = self.A_dot_A, self.A_dot_B, self.B_dot_B, self.n, self.T
    
        M_dot_A = self.ReLU(alpha_A*A_dot_A + alpha_B*A_dot_B)
        M_dot_B = self.ReLU(alpha_A*A_dot_B + alpha_B*B_dot_B)
        
        l_gamma_o_A = -np.tanh( (M_dot_A/T)**n )
        l_gamma_o_B = -np.tanh( (M_dot_B/T)**n )


        nabla_gamma = ( - (1 + l_gamma_o_A)**(2*n - 1)  *  (1 - l_gamma_o_A**2)  *  (M_dot_A/T)**n

                        - (1 + l_gamma_o_B)**(2*n - 1)  *  (1 - l_gamma_o_B**2)  *  (M_dot_B/T)**n
        )


        return nabla_gamma 

       

    
    def calc_d_alphas_dt(self, alpha_A, alpha_B, ell):

        
        nabla_A = self.calc_nabla_A(alpha_A, alpha_B, ell)
        nabla_B = self.calc_nabla_B(alpha_A, alpha_B, ell)

        norm_qtty =  (np.abs(alpha_A)/alpha_A)  *  nabla_A  +  (np.abs(alpha_B)/alpha_B)  *  nabla_B

        norm_qtty = np.where( (alpha_A + alpha_B < 1) or (norm_qtty < 0), 0, norm_qtty ) # Check if normalization conditions are not held, in which case no normalization occurs


        
        # Combines both with and without normalization using ReLU
        dt_alpha_A = ( nabla_A - alpha_A*(norm_qtty) ) / ( np.abs(nabla_A) + np.abs(nabla_B) )  # We are assuming norm_qtty > 0 i.e. looking only at nullclines
        dt_alpha_B = ( nabla_B - alpha_B*(norm_qtty) ) / ( np.abs(nabla_A) + np.abs(nabla_B) )  # We are assuming norm_qtty > 0 i.e. looking only at nullclines

        return dt_alpha_A, dt_alpha_B


    
    def calc_d_ell_dt(self, alpha_A, alpha_B, ell):

        # Non normalized
        nabla_ell = self.calc_nabla_ell(alpha_A, alpha_B, ell)
        nabla_gamma = self.calc_nabla_gamma(alpha_A, alpha_B, ell)
        
        dt_ell = nabla_ell  /  np.maximum( np.abs(nabla_ell) , np.abs(nabla_gamma)  )


        dt_ell = np.where((ell >= 1) * (dt_ell > 0), 0, dt_ell) # Takes care of clipping for +ve ell
        dt_ell = np.where((ell <= -1) * (dt_ell < 0), 0, dt_ell) # Takes care of clipping for -ve ell 
        
        return dt_ell


    def simulate_and_save(self, alpha_A_init, alpha_B_init, ell_init, tmax, dt, savefile):

        alpha_As, alpha_Bs, ells = np.zeros((3, tmax))
        
        alpha_As[0], alpha_Bs[0], ells[0] = alpha_A_init, alpha_B_init, ell_init

        for t in range(1, tmax):

            d_alpha_A_dt, d_alpha_B_dt = self.calc_d_alphas_dt(alpha_As[t-1], alpha_Bs[t-1], ells[t-1])
            d_ell_dt = self.calc_d_ell_dt(alpha_As[t-1], alpha_Bs[t-1], ells[t-1])

            alpha_As[t] = alpha_As[t-1] + dt*d_alpha_A_dt
            alpha_Bs[t] = alpha_Bs[t-1] + dt*d_alpha_B_dt
            ells[t] = ells[t-1]  +  dt*d_ell_dt

        
        np.savez(savefile, alpha_As=alpha_As, alpha_Bs=alpha_Bs, ells=ells)

        return ells, alpha_As, alpha_Bs






