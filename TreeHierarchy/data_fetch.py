import numpy as np

data_dir = "data/[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]/trained_net_n30_T650.npz"

data = np.load(data_dir)


for s in range(1, 21):
    t_i = (s-1)*100
    t_f = s*100
    saving_dir = "data/[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]/trained_net_F"+str(s)+"_n30_T650.npz"
    
    # Downsampling size is 100 (not anymore)
    # This is now keeping only the first 2000 epochs
    np.savez_compressed(saving_dir, init_array=data['init_array'], selected_digits=data['selected_digits'],
                        M=data['M'][t_i:t_f], L=data['L'][t_i:t_f], miniBatchs_images=data['miniBatchs_images'], miniBatchs_labels=data['miniBatchs_labels'])
