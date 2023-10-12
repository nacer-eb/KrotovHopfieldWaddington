import numpy as np # <<- I'm starting to hate this library...
import matplotlib.pyplot as plt

import matplotlib
font = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 22}

matplotlib.rc('font', **font)


import sys
sys.path.append('../')

from main_module.KrotovV2_utils import *


    
class KrotovNet:
    def __init__(self, Kx=10, Ky=10, n_deg=30, m_deg=30, M=1000, nbMiniBatchs=60, momentum=0.6, rate=0.0008, temp=600, rand_init_mean=-0.03, rand_init_std=0.03,
                 initHiddenDetectors=None, initVisibleDetectors=None, selected_digits=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], useClipping=False):

        # Running the init method
        self.init_net(Kx, Ky, n_deg, m_deg, M, nbMiniBatchs, momentum, rate, temp, rand_init_mean, rand_init_std,
                      initHiddenDetectors, initVisibleDetectors, selected_digits, useClipping)
        
        
    def init_net(self, Kx=10, Ky=10, n_deg=30, m_deg=30, M=1000, nbMiniBatchs=60, momentum=0.6, rate=0.0008, temp=600, rand_init_mean=-0.03, rand_init_std=0.03,
                 initHiddenDetectors=None, initVisibleDetectors=None, selected_digits=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], useClipping=False):
        #Parameter Setup
        self.N_C = 10 #Number of classification/output neurons
        self.N = 28*28 #Number of visible inputs

        self.Kx = int( Kx )
        self.Ky = int( Ky )
        self.K = self.Kx*self.Ky #Number of memories

        self.M = int( M ) #Number of examples per trainingBatch # MAX 1000
        self.nbMiniBatchs = int( nbMiniBatchs ) #Number of minibatchs # MAX 60

        print(self.M)

        self.n_deg = n_deg # The power in the rectified polynomial
        self.m_deg = m_deg # The power in the cost function (2m is used)

        self.train_momentum = momentum # The training momentum
        self.train_rate = rate # The learning rate

        self.temperature_beta = temp

        
        self.beta = 1.0/(np.power(self.temperature_beta, self.n_deg, dtype="double")) # The temperature (normalization of the input)

        self.rand_init_mean = rand_init_mean
        self.rand_init_std = rand_init_std

        self.useClipping = useClipping

        np.random.seed()
        # Random inits the memories.
        if initHiddenDetectors is None:
            np.random.seed()
            self.hiddenDetectors = np.random.normal(self.rand_init_mean, self.rand_init_std, (self.K, self.N_C)) # You can multiply this by zero and get the same thing
        else:
            self.hiddenDetectors = initHiddenDetectors
            
        if initVisibleDetectors is None:
            np.random.seed()
            self.visibleDetectors = np.random.normal(self.rand_init_mean, self.rand_init_std, (self.K, self.N))
        else:
            self.visibleDetectors = initVisibleDetectors

        self.selected_digits = selected_digits
            
        self.dV_hiddenDetectors = np.zeros((self.K, self.N_C))
        self.dV_visibleDetectors = np.zeros((self.K, self.N))

        # Fetching the data...
        # This never changes and is a standard fetch from the database
        self.test_batch_images = get_MNIST_test_images()
        self.test_batch_keys = get_MNIST_test_labels()
        self.test_batch_size = 10000
        
        self.train_batch_images_full = get_MNIST_train_images()
        self.train_batch_keys_full = get_MNIST_train_labels()
        self.train_batch_full_size = 60000

        # This is where we store the actual training miniBatchs we use. 
        self.miniBatchs_images = np.zeros((self.nbMiniBatchs, self.M, self.N))
        self.miniBatchs_labels = np.zeros((self.nbMiniBatchs, self.M, self.N_C))+1.0
        
        # Organize it into batches.
        for mb in range(0, self.nbMiniBatchs):
            print(mb, end=' ')
            self.miniBatchs_images[mb], self.miniBatchs_labels[mb] = get_MNIST_train_partitionned(self.M, self.train_batch_images_full, self.train_batch_keys_full, self.selected_digits)
            

    """  //////////// SET METHODS \\\\\\\\\\\\\\\\\\\ """
    
    def set_nm_deg(self, n_deg):
        self.n_deg = n_deg
        self.m_deg = n_deg # This was added wayyy to late. 2023/03/19 but 1_2_2_sys now includes it
        self.beta = 1.0/(np.power(self.temperature_beta, self.n_deg, dtype="double"))
    
    def set_n_deg(self, n_deg):
        self.n_deg = n_deg
        self.beta = 1.0/(np.power(self.temperature_beta, self.n_deg, dtype="double"))

    def set_temp(self, temp):
        self.temperature_beta = temp
        self.beta = 1.0/(np.power(self.temperature_beta, self.n_deg, dtype="double"))

    """  //////////// SET METHODS - End \\\\\\\\\\\\\\\\\\\ """


    
    def load_net(self, filename, epoch=0):
        data = np.load(filename)
        
        self.init_net(*data["init_array"], selected_digits=data["selected_digits"])
                
        self.visibleDetectors = data["M"][epoch]
        self.hiddenDetectors = data["L"][epoch]

        self.miniBatchs_images = data["miniBatchs_images"]
        self.miniBatchs_labels = data["miniBatchs_labels"]

        
                      
    """  //////////// SPECIAL INIT FUNCTIONS \\\\\\\\\\\\\\\\\\\ """
    def training_diversification_2(self, tol=100):
        correlations = np.sum(np.abs(self.miniBatchs_images[0, 0] - self.miniBatchs_images[0, 1]))

        while correlations > tol:
            self.miniBatchs_images[0], self.miniBatchs_labels[0] = get_MNIST_train_partitionned(self.M, self.train_batch_images_full, self.train_batch_keys_full, self.selected_digits)
            correlations = np.sum(np.abs(self.miniBatchs_images[0, 0] - self.miniBatchs_images[0, 1]))
            print(correlations, self.miniBatchs_images[0, 0]@self.miniBatchs_images[0, 0], self.miniBatchs_images[0,0]@self.miniBatchs_images[0, 1], self.miniBatchs_images[0, 1]@self.miniBatchs_images[0, 1])

        
    def training_diversification(self, tol_min=400, tol_max=700, width_difference=10):
        # Here I make sure the two examples are different enough
        for mb in range(self.nbMiniBatchs):
            correlations = self.miniBatchs_images[mb]@self.miniBatchs_images[mb].T
            cross_correlations = correlations[~np.eye(self.M, dtype=bool)] # ~ is logical not

            widths = np.sum(self.miniBatchs_images[mb], axis=1)
                        
            while np.min(cross_correlations) < tol_min or np.max(cross_correlations) > tol_max or np.std(widths) > width_difference:
                self.miniBatchs_images[mb], self.miniBatchs_labels[mb] = get_MNIST_train_partitionned(self.M, self.train_batch_images_full, self.train_batch_keys_full, self.selected_digits)
                print("Diversification...", np.min(cross_correlations), " --- ", np.max(cross_correlations), widths, "(", np.std(widths), ")")

                correlations = self.miniBatchs_images[mb]@self.miniBatchs_images[mb].T
                cross_correlations = correlations[~np.eye(self.M, dtype=bool)] # ~ is logical not
                widths = np.sum(self.miniBatchs_images[mb], axis=1)
        
    def show_minibatchs(self):
        for mb in range(self.nbMiniBatchs):
            plt.imshow(merge_data(self.miniBatchs_images[mb], self.M, 1), cmap="bwr")
            print(self.miniBatchs_images[mb]@self.miniBatchs_images[mb].T)
            plt.title("Training data: Minibatch number " + str(mb+1) + " out of " + str(self.nbMiniBatchs))
            plt.colorbar(location='left')
            plt.show()     
           
    """  //////////// SPECIAL INIT FUNCTIONS - END \\\\\\\\\\\\\\\\\\\ """


    """  //////////// UTILS \\\\\\\\\\\\\\\\\\\ """
    # Resets the memories and labels
    def reset_ML(self):
        np.random.seed()
        self.visibleDetectors = np.random.normal(self.rand_init_mean, self.rand_init_std, (self.K, self.N))
        self.hiddenDetectors = np.random.normal(self.rand_init_mean, self.rand_init_std, (self.K, self.N_C))
        
        self.dV_hiddenDetectors = np.zeros((self.K, self.N_C))
        self.dV_visibleDetectors = np.zeros((self.K, self.N))

    # Necessary for the saving mechanism. Outputs the net's parameters in a single array.
    def get_init_array(self):
        init_array = np.zeros(11)
        init_array = np.asarray([self.Kx, self.Ky, self.n_deg, self.m_deg, self.M, self.nbMiniBatchs,
                      self.train_momentum, self.train_rate, self.temperature_beta, self.rand_init_mean,
                      self.rand_init_std])
        
        return init_array

    def save_txt_init_array(self, data_dir, sim_name):
        init_array = self.get_init_array()
        np.savetxt(data_dir+"/"+sim_name+".initarray", init_array, delimiter=",")

    def save_txt_visible_detectors(self, data_dir, sim_name):
        np.savetxt(data_dir+"/"+sim_name+".visible_detectors", self.visibleDetectors, delimiter=",")


    def save_txt_hidden_detectors(self, data_dir, sim_name):
        np.savetxt(data_dir+"/"+sim_name+".hidden_detectors", self.hiddenDetectors, delimiter=",")

    def save_txt_miniBatchs(self, data_dir, sim_name):
        for mb in range(self.nbMiniBatchs):
            np.savetxt(data_dir+"/"+sim_name+".miniBatchs_images_"+str(mb), self.miniBatchs_images[mb], delimiter=",")
            np.savetxt(data_dir+"/"+sim_name+".miniBatchs_labels_"+str(mb), self.miniBatchs_labels[mb], delimiter=",")
    
    
    # The Rectified Polynomial
    def f_n(self, x, n):
        return np.power( (x + np.abs(x))/2.0 , n, dtype="double")

    # Standard Computation function for the network
    def compute(self, v_i):
        t = self.beta*self.f_n(np.dot(self.visibleDetectors, v_i), self.n_deg)
        s = np.reshape(np.repeat(t, self.N_C), (self.K, self.N_C))
        u = np.tanh(np.sum(np.multiply(self.hiddenDetectors, s), axis=0))
        return u
    
    # Computes the activation strength. i.e. computation without applying labels or tanh
    def compute_activationStrength(self, v_i):
        t = self.beta*self.f_n(np.dot(self.visibleDetectors, v_i), self.n_deg)
        return t


    # All 2m-1 are meant to be 'odd powers'
    def odd_power(self, a, p):
        return np.sign(a) * np.power(np.abs(a), p, dtype="double")

    # All 2m are meant to be even powers
    def even_power(self, a, p):
        return np.power(np.abs(a), p, dtype="double")
        
    
    """  //////////// UTILS - END \\\\\\\\\\\\\\\\\\\ """



    """  //////////// TRAINING \\\\\\\\\\\\\\\\\\\ """
    
    # Gradient descent for the visible layer. (Fast but ugly?)
    def visibleLayer_train(self, v, t):
        v_ = v[0]
        t_ = t[0]

        c = self.compute(v_)

        
        dV_1 = ( self.odd_power(c - t_, 2*self.m_deg - 1)  )*(1-c*c)*self.hiddenDetectors
        dV_2 = np.reshape(np.repeat(self.f_n(np.dot(self.visibleDetectors, v_), self.n_deg-1), self.N_C), (self.K, self.N_C))
        dV_tmp = 2*self.n_deg*self.m_deg*self.beta*v_*np.reshape(np.repeat(np.sum(dV_1*dV_2, axis=1), self.N), (self.K, self.N))
        dV = dV_tmp

        
        E = np.sum(  self.even_power( c - t_, 2*self.m_deg ) )
        for i in range(1, self.M):

            v_ = v[i]
            t_ = t[i]

            c = self.compute(v_)

            E += np.sum( self.even_power(c-t_ , 2*self.m_deg))

            dV_1 = ( self.odd_power(c - t_, 2*self.m_deg - 1) )*(1-c*c)*self.hiddenDetectors
            dV_2 = np.reshape(np.repeat(self.f_n(np.dot(self.visibleDetectors, v_), self.n_deg-1), self.N_C), (self.K, self.N_C))


            dV_tmp = 2*self.n_deg*self.m_deg*self.beta*v_*np.reshape(np.repeat(np.sum(dV_1*dV_2, axis=1), self.N), (self.K, self.N))

            
            
            dV += dV_tmp

        #Normalization to the -1, 1 range
        #dV /= np.reshape(np.repeat(np.max(dV, axis=1), self.N), (self.K, self.N))

        

        
        return dV
    
    # Gradient descent for the hidden layer. (Fast but ugly?)
    def hiddenLayer_train(self, v, t):
        v_ = v[0]
        t_ = t[0]
        
        c = self.compute(v_)

        dV_1 = ( self.odd_power(c - t_, 2*self.m_deg - 1) )*(1-c)*(1+c)

        dV_2 = np.reshape(np.repeat(self.f_n(np.dot(self.visibleDetectors, v_), self.n_deg), self.N_C), (self.K, self.N_C))
       
        dV_tmp = 2*self.m_deg*self.beta*dV_1*dV_2
        
        dV = dV_tmp
        
        for i in range(1, self.M):
            v_ = v[i]
            t_ = t[i]

            c = self.compute(v_)

            dV_1 = ( self.odd_power(c - t_, 2*self.m_deg - 1) )*(1-c)*(1+c)
            dV_2 = np.reshape(np.repeat(self.f_n(np.dot(self.visibleDetectors, v_), self.n_deg), self.N_C), (self.K, self.N_C))

            
            
            dV_tmp = 2*self.m_deg*self.beta*dV_1*dV_2
            
            dV += dV_tmp

            #Normalization to the -1, 1 range
            #dV /= np.reshape(np.repeat(np.max(np.abs(dV), axis=1), self.N_C), (self.K, self.N_C))
        
        return dV

    # Combines gradient descent of both layers (M/L), Normalization per memory and momentum.
    def train_cycle(self, mb_v, mb_t):

        for mb in range(0, self.nbMiniBatchs):
            tmp_dV_v = self.visibleLayer_train(mb_v[mb], mb_t[mb])
            tmp_dV_v_norm = np.reshape(np.repeat(np.max(np.abs(tmp_dV_v), axis=1), self.N), (self.K, self.N))
            tmp_dV_v = np.divide(tmp_dV_v, tmp_dV_v_norm, out=np.zeros_like(tmp_dV_v), where=tmp_dV_v_norm!=0) #Fixes the divergence when the reaching a stable point (norm->0)
            
            self.dV_visibleDetectors = self.dV_visibleDetectors*self.train_momentum - tmp_dV_v
            
            # MODIFIED THIS TO FORCE QUASISTATIC
            for k in range(0, 1):
                tmp_dV_h = self.hiddenLayer_train(mb_v[mb], mb_t[mb])
                #print("t", -1*tmp_dV_h)
                tmp_dV_h_norm = np.reshape(np.repeat(np.max(np.abs(tmp_dV_h), axis=1), self.N_C), (self.K, self.N_C))
                
                
                tmp_dV_h = np.divide(tmp_dV_h, tmp_dV_h_norm, out=np.zeros_like(tmp_dV_h), where=tmp_dV_h_norm!=0) #Fixes the divergence when the reaching a stable point (norm->0)
            
                self.dV_hiddenDetectors = self.dV_hiddenDetectors*self.train_momentum - tmp_dV_h

                ##
                #Normalization to the -1, 1 range
                # Fixed for divergences... Kept the old version around just in case
                V_hidden_norm_factor = np.reshape(np.repeat(np.max(np.abs(self.dV_hiddenDetectors), axis=1), self.N_C), (self.K, self.N_C))
                
                V_hidden = np.divide(self.dV_hiddenDetectors, V_hidden_norm_factor, out=np.zeros_like(self.dV_hiddenDetectors), where=V_hidden_norm_factor!=0)
                #V_hidden = self.dV_hiddenDetectors/V_hidden_norm_factor 
                
                self.hiddenDetectors += self.train_rate*V_hidden 
                self.hiddenDetectors = np.clip(self.hiddenDetectors, -1., 1.)
                
            
            ##
            #Normalization to the -1, 1 range
            # Fixed for divergences... Kept the old version around just in case
            V_visible_norm_factor = np.reshape(np.repeat(np.max(np.abs(self.dV_visibleDetectors), axis=1), self.N), (self.K, self.N))
            V_visible = np.divide(self.dV_visibleDetectors, V_visible_norm_factor, out=np.zeros_like(self.dV_visibleDetectors), where=V_visible_norm_factor!=0)
            #V_visible = self.dV_visibleDetectors/V_visible_norm_factor
                        
            self.visibleDetectors += self.train_rate*V_visible
            
            
            # This is where linearity may arise or be corrected.
            if self.useClipping:
                # THE CLIPPING METHOD:
                self.visibleDetectors = np.clip(self.visibleDetectors, -1., 1.)
                # self.hiddenDetectors = np.clip(self.hiddenDetectors, -1., 1.)

            else:
                # THE LINEAR-CONSERVATION METHOD ONLY APPLIES TO MEMORIES:
                norm_v = np.max(np.abs(self.visibleDetectors), axis=-1)
                norm_v[norm_v < 1] = 1
                norm_v = np.expand_dims(norm_v, -1)
                norm_v = np.repeat(norm_v, np.shape(self.visibleDetectors)[-1], -1)

                """
                norm_h = np.max(np.abs(self.hiddenDetectors), axis=-1)
                norm_h[norm_h < 1] = 1
                norm_h = np.expand_dims(norm_h, -1)
                norm_h = np.repeat(norm_h, np.shape(self.hiddenDetectors)[-1], -1)
                """
                
                self.visibleDetectors = np.divide(self.visibleDetectors, norm_v)
                #self.hiddenDetectors = np.divide(self.hiddenDetectors, norm_h)
            
            
            
            #print(mb, end=', ')
        #print('Done')
        
    """  //////////// TRAINING - END \\\\\\\\\\\\\\\\\\\ """


        
    #This assumes the labels aren't arrays, but scalar     
    def evaluateNet(self, test_batch_images, test_batch_keys, test_batch_size):
        """
        Evaluates the net on a given input batch.
        
        @params 
        test_batch_images (array of 'digits' inputs) : The complete batch to test on.
        test_batch_keys (array of 'digits' inputs) : The true answers.
        test_batch_size (array of 'digits' inputs) : The effective size to use (<= max size)

        @return 
        The score: Percentage of failure (1-success_rate).
        """
        
        score = 0
        print(">................,", np.shape(test_batch_images), test_batch_size)
        for i in range(0, test_batch_size):
            score += int(np.argmax(self.compute(test_batch_images[i]), axis=0) == test_batch_keys[i])
            

        score /= test_batch_size
    
        score = 1-score
    
        return score


    def output_test_results(self, testingRegiment):
        if testingRegiment[0] > 0:
            print("Minibatchs score: ")
            for mb in range(0, self.nbMiniBatchs):
                mini_batch_scores_per_epoch = self.evaluateNet(self.miniBatchs_images[mb], np.argmax(self.miniBatchs_labels[mb], axis=1), self.M)
                print(mb, ":", mini_batch_scores_per_epoch)
            print(" -- ")

        if testingRegiment[1] > 0:
            test_batch_score = self.evaluateNet(self.test_batch_images, self.test_batch_keys, testingRegiment[1])
            print("Test batch score:", test_batch_score)
            
        if testingRegiment[2] > 0:
            train_batch_score = self.evaluateNet(self.train_batch_images_full, self.train_batch_keys_full, testingRegiment[2])
            print("Training batch score:", test_batch_score)
    

    def train_plot_update(self, epochs, isPlotting=True, isSaving=False, saving_dir=None, testFreq=100, testingRegiment=[1, 0, 0], isDecay=False, id_mem=None):
        """
        Train & Plot function.
        Trains the network and plots it. It also allows you to test the network on the various digits batchs as per the testingRegiment parameter.
   
        @params
        epochs (int) : Number of epochs of training.
        isPlotting (Boolean) : Do you want to plot. (DFLT: True)
        isSaving (Boolean) : Do you want to save. (DFLT: False)
        saving_dir (String) : Where to save the network. (DFLT: None)
        testFreq (int) : How frequently to test the network. (DFLT: 50)
        testingRegiment (int[3]) : How large the testing pool; [Test with the miniBatchs y/n? (0, 1) (BOOL), Effective test batch size to test on, Effective train batch size to test on]
        
        @return
        
        """

        # Space saver : using empty? garbagecollector could help or break ... not risking it yet.

        
        # Makes sure the mnist_data_dir exists else creates it.
        if not saving_dir is None:
            if not path.exists(os.path.dirname(saving_dir)):
                print(os.path.dirname(saving_dir), "Does not exist. It will be created ...")
                os.mkdir(os.path.dirname(saving_dir))
                print(os.path.dirname(saving_dir), "Created!")
        
        M = np.zeros((epochs, self.K, self.N))
        L = np.zeros((epochs, self.K, self.N_C))

        M[0, :, :] = self.visibleDetectors
        L[0, :, :] = self.hiddenDetectors

        data_T = self.miniBatchs_images[0]
        data_T_inv = np.linalg.pinv(data_T)
        if isPlotting:
            plt.ion()      
            fig, ax = plt.subplots(1, 1, figsize=(16, 9))        
            im = ax.imshow(merge_data(self.visibleDetectors, self.Kx, self.Ky), cmap="bwr", vmin=-1, vmax=1) #Plots

        i_end = 0
        for i in range(0, epochs):

            # This is cool - generally not implemented only in special**2 cases
            if id_mem is not None:
                for d in id_mem[1:]:
                    self.visibleDetectors[d] = self.visibleDetectors[id_mem[0]]
                    self.hiddenDetectors[d] = self.hiddenDetectors[id_mem[0]]
            
            i_end += 1
            #if i%20 == 0 and True: # Removed, too verbose
            #print("-> %d / %d" % (i, epochs) )
            #print("Rank: ", np.linalg.matrix_rank(self.visibleDetectors), np.linalg.matrix_rank(self.hiddenDetectors))
            
            if isDecay:
                decayRate = 0.998
                self.train_rate *= decayRate

            if (i % testFreq) == 0:
                print("-> %d / %d" % (i, epochs) )
                self.output_test_results(testingRegiment)

            if isPlotting:
                im.set_data(merge_data(self.visibleDetectors, self.Kx, self.Ky)) #Plots
                ax.set_title(str(i))
                plt.pause(0.01)

            if isSaving:
                M[i, :, :] = self.visibleDetectors
                L[i, :, :] = self.hiddenDetectors


            
            # IMPORTANT REMOVE WHEN DONE: MAKING LAST TWO MEM AND LABELS EQUAL
            #self.visibleDetectors[2] = self.visibleDetectors[1]
            #self.hiddenDetectors[2] = self.hiddenDetectors[1]
            ## End of temporary code - Boy is this dangerous

            self.train_cycle(self.miniBatchs_images, self.miniBatchs_labels) #Train
            
            

                
        print(i_end)   
        if isPlotting:
            plt.cla()
            plt.clf()
            plt.close()
        
        # Before saving I want to reset the train_rate to what it was before training
        if isDecay:
            self.train_rate /= np.power(decayRate, epochs, dtype="double")

        if isSaving:
            print("Saving to", saving_dir, "...")
            np.savez_compressed(saving_dir, init_array=self.get_init_array(), selected_digits=self.selected_digits, M=M[:i_end], L=L[:i_end], miniBatchs_images=self.miniBatchs_images, miniBatchs_labels=self.miniBatchs_labels)
            print("Save completed.")

            
            
