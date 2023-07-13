import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt


class KrotovLinearAnalysis:
    def __init__(self, filename):
        self.filename = filename

        # fetch data (Memories, Labels, Training images)
        self.M, self.L, self.T = self.fetch_data()
        self.coefs = self.get_coefs() # (memory, coefficients)

    def fetch_data(self):
        data = np.load(self.filename)

        return data["M"], data["L"], data["miniBatchs_images"]

    
    def reconstruction_function(self, T, coefs):
        tmp = coefs @ T

        #b=0.293 There some improvement to be made here... this is interesting.
        b = 1
        
        return np.clip(tmp/b, -1*b, 1*b)/b
    
    def reconstruct_images(self):
        return self.reconstruction_function(self.T, self.coefs)
    
    def get_coefs(self):
        T_inv = np.linalg.pinv(self.T)
        coefs = self.M @ T_inv
       
        return coefs
    

    def obtain_residuals(self):
        M_reconstructed = self.reconstruct_images()
        res = self.M - M_reconstructed
        
        return res
