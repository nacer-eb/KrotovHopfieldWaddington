import numpy as np

class GatherNullClines:
    def __init__(self, A_dot_A, A_dot_B, B_dot_B, n, T, pm):


        self.A_dot_A, self.A_dot_B, self.B_dot_B, self.n, self.T, self.pm = A_dot_A, A_dot_B, B_dot_B, n, T, pm
        self.m = n # by default.

    
    # I use beta here, but this is the same as alpha_B, and alpha_A = alpha.
    def beta(self, alpha):
        if self.pm == 1:
            return 1 - np.abs(alpha)

        if self.pm == -1:
            return -1 + np.abs(alpha)

        # else
        print("Garbage in, garbage out. pm was set to", pm, "it should be either +1 or -1 and represents the sign of beta.")
        exit(-1)


    # In case you're in a weird region of the (alpha,beta)-space
    # For now not used; I assume you don't run this when both alpha and beta are negative
    def ReLU(self, x):
        return (np.abs(x) + x)/2.0


    def calc_nabla_A(self, alpha, ell):

        A_dot_A, A_dot_B, B_dot_B, n, T, pm = self.A_dot_A, self.A_dot_B, self.B_dot_B, self.n, self.T, self.pm

        beta = self.beta(alpha)

        M_dot_A = self.ReLU(alpha*A_dot_A + beta*A_dot_B)
        
        l_A_o_A = np.tanh( ell * (M_dot_A/T)**n )
        l_gamma_o_A = -np.tanh( (M_dot_A/T)**n )

        
        nabla_A = ( (1 - l_A_o_A)**(2*n - 1)  *  (1 - l_A_o_A**2)  *  (M_dot_A/T)**(n-1) * ell

                         + 4 * (1 + l_gamma_o_A)**(2*n - 1)  *  (1 - l_gamma_o_A**2) *  (M_dot_A/T)**(n-1) 
        )

        return nabla_A


    def calc_nabla_B(self, alpha, ell):

        A_dot_A, A_dot_B, B_dot_B, n, T, pm = self.A_dot_A, self.A_dot_B, self.B_dot_B, self.n, self.T, self.pm

        beta = self.beta(alpha)

        M_dot_B = self.ReLU(alpha*A_dot_B + beta*B_dot_B)
        
        l_A_o_B = np.tanh( ell * (M_dot_B/T)**n )
        l_gamma_o_B = -np.tanh( (M_dot_B/T)**n )

        
        nabla_B = ( -(1 + l_A_o_B)**(2*n - 1)  *  (1 - l_A_o_B**2)  *  (M_dot_B/T)**(n-1) * ell

                         + 4 * (1 + l_gamma_o_B)**(2*n - 1)  *  (1 - l_gamma_o_B**2) *  (M_dot_B/T)**(n-1) 
        )

        return nabla_B


    def calc_nabla_ell(self, alpha, ell):
        
        A_dot_A, A_dot_B, B_dot_B, n, T, pm = self.A_dot_A, self.A_dot_B, self.B_dot_B, self.n, self.T, self.pm
    
        beta = self.beta(alpha)

        
        M_dot_A = self.ReLU(alpha*A_dot_A + beta*A_dot_B)
        M_dot_B = self.ReLU(alpha*A_dot_B + beta*B_dot_B)

        l_A_o_A = np.tanh( ell * (M_dot_A/T)**n )        
        l_A_o_B = np.tanh( ell * (M_dot_B/T)**n )


        nabla_ell = ( (1 - l_A_o_A)**(2*n - 1)  *  (1 - l_A_o_A**2) * (M_dot_A/T)**n

                      -  (1 + l_A_o_B)**(2*n - 1)  *  (1 - l_A_o_B**2) * (M_dot_B/T)**n
            
        )


        return nabla_ell

    def calc_nabla_gamma(self, alpha, ell):

        A_dot_A, A_dot_B, B_dot_B, n, T, pm = self.A_dot_A, self.A_dot_B, self.B_dot_B, self.n, self.T, self.pm
    
        beta = self.beta(alpha)

        M_dot_A = self.ReLU(alpha*A_dot_A + beta*A_dot_B)
        M_dot_B = self.ReLU(alpha*A_dot_B + beta*B_dot_B)
        
        l_gamma_o_A = -np.tanh( (M_dot_A/T)**n )
        l_gamma_o_B = -np.tanh( (M_dot_B/T)**n )


        nabla_gamma = ( - (1 + l_gamma_o_A)**(2*n - 1)  *  (1 - l_gamma_o_A**2)  *  (M_dot_A/T)**n

                        - (1 + l_gamma_o_B)**(2*n - 1)  *  (1 - l_gamma_o_B**2)  *  (M_dot_B/T)**n
        )


        return nabla_gamma 

       
    
    def calc_d_alpha_dt(self, alpha, ell):

        beta = self.beta(alpha)
        
        nabla_A = self.calc_nabla_A(alpha, ell)
        nabla_B = self.calc_nabla_B(alpha, ell)

        norm_qtty =  (np.abs(alpha)/alpha)  *  nabla_A  +  (np.abs(beta)/beta)  *  nabla_B

        dt_alpha = ( nabla_A - alpha*(norm_qtty) ) / ( np.abs(nabla_A) + np.abs(nabla_B) )  # We are assuming norm_qtty > 0 i.e. looking only at nullclines

        return dt_alpha, norm_qtty > 0


    
    def calc_d_ell_dt(self, alpha, ell):

        # Non normalized

        nabla_ell = self.calc_nabla_ell(alpha, ell)
        nabla_gamma = self.calc_nabla_gamma(alpha, ell)
        
        dt_ell = nabla_ell  /  np.maximum( np.abs(nabla_ell) , np.abs(nabla_gamma)  )

        # deal with normalized with ell == 1 and ell = -1
        dt_ell_sat_p = np.ones_like(dt_ell)
        dt_ell_sat_m = np.ones_like(dt_ell)
        
        dt_ell_sat_p = np.where((ell == 1) * (dt_ell >= 0), 0, 1)
        dt_ell_sat_m = np.where((ell == -1) * (dt_ell <= 0), 0, 1)
        
        return dt_ell, dt_ell_sat_p, dt_ell_sat_m

