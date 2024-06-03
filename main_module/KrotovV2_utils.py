import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import os
from os import path

# This is where you want the mnist (training/test) data to be saved.
# Will be created if it doesn't exist. 
mnist_data_dir = "../mnist_data"

# Makes sure the mnist_data_dir exits else creates it.
if not path.exists(mnist_data_dir):
    print(mnist_data_dir, "Does not exist. It will be created ...")
    os.mkdir(mnist_data_dir)
    print(mnist_data_dir, "Created!")


def get_MNIST_train_images():
    """
    Fetchs all the training images in MNIST
    """

    filename = mnist_data_dir + "/mnist_train_images.npy"

    if not path.exists(filename):
        import tensorflow.keras as keras
        
        MNIST_train_images = (np.reshape(keras.datasets.mnist.load_data()[0][0], (60000, 784))/255.0)*2.-1.
        np.save(filename, MNIST_train_images)
        print("First time loading the training images...")
    else:

        MNIST_train_images = np.load(filename)
        print("Found preloaded trainning images!")

    return MNIST_train_images


def get_MNIST_train_labels():
    """
    Fetchs all the training labels (actually keys) in MNIST
    """

    filename = mnist_data_dir + "/mnist_train_labels.npy"

    if not path.exists(filename):
        import tensorflow.keras as keras
        
        MNIST_train_labels = np.reshape(keras.datasets.mnist.load_data()[0][1], (60000,))
        np.save(filename, MNIST_train_labels)
        print("First time loading the training labels...")
    else:
        MNIST_train_labels = np.load(filename)
        print("Found preloaded training labels!")
    
    
    return MNIST_train_labels
    

def get_MNIST_test_images():
    """
    Fetchs all the test images in MNIST
    """

    filename = mnist_data_dir + "/mnist_test_images.npy"
    
    if not path.exists(filename):
        import tensorflow.keras as keras
        
        MNIST_test_images = (np.reshape(keras.datasets.mnist.load_data()[1][0], (10000, 784))/255.0)*2.-1.
        np.save(filename, MNIST_test_images)
        print("First time loading the test images...")
    else:
        
        MNIST_test_images = np.load(filename)
        print("Found preloaded test images!")    
    
    return MNIST_test_images


def get_MNIST_test_labels():
    """
    Fetchs all the test labels (actually keys) in MNIST
    """

    filename = mnist_data_dir + "/mnist_test_labels.npy"

    if not path.exists(filename):
        import tensorflow.keras as keras
        
        MNIST_test_labels = np.reshape(keras.datasets.mnist.load_data()[1][1], (10000,))
        np.save(filename, MNIST_test_labels)
        print("First time loading the test labels...")
    else:
        MNIST_test_labels = np.load(filename)
        print("Found preloaded test labels!")

        
    return MNIST_test_labels

def get_MNIST_train_partitionned(totalPartitionSize, train_images, train_labels, selected_digits, randomize=False, unsupervisedMode=False, N_unsupervised_classes=600):
    """
    Fetchs all the training images and labels that are randomly sampled and partitioned i.e. same number of each class.
    """

    digits_dims = len(selected_digits)
    sample_size = totalPartitionSize//digits_dims
        
    images = []
    labels = []

    indices = np.argwhere(train_labels == selected_digits[0]).T[0]
    random_subIndices = np.random.randint(0, len(indices), sample_size)

    sampled_indices = np.take(indices, random_subIndices)
    sampled_train_images = np.take(train_images, sampled_indices, axis=0)
    images = sampled_train_images
    labels = np.zeros((sample_size, 10))-1.
    labels.T[selected_digits[0]] = 1.
    
    for i in range(1, digits_dims):
        digit = selected_digits[i]
        indices = np.argwhere(train_labels == digit).T[0]
        random_subIndices = np.random.randint(0, len(indices), sample_size)
        
        sampled_indices = np.take(indices, random_subIndices)
        sampled_train_images = np.take(train_images, sampled_indices, axis=0)
        images = np.append(images, sampled_train_images, axis=0)
        label_tmp = np.zeros((sample_size, 10))-1.
        label_tmp.T[digit] = 1.
        labels = np.append(labels, label_tmp, axis=0)

    if randomize:
        randIndices = np.random.randint(0, totalPartitionSize, (totalPartitionSize))
        
        images = np.take(images, randIndices, axis=0)
        labels = np.take(labels, randIndices, axis=0)


    if unsupervisedMode:
        unsupervised_classes =  np.eye(N_unsupervised_classes)*2-1 # idealy
        label_keys =  np.arange(0, totalPartitionSize, 1) # np.random.randint(0, N_unsupervised_classes, totalPartitionSize)
        labels = unsupervised_classes[label_keys]
        #labels = images

    return (images, labels)





def merge_data(initial_data, Kx, Ky):
    """
    Plots a (Kx*Ky, 784) array of memories into a Kx, Ky images.
    """

    merged_data = np.zeros((28*Ky, 28*Kx))
    
    for k in range(0, Ky):
        for j in range(0, 28):
            for i in range(0, Kx):
                merged_data[j+k*28][28*(i):28*(i+1)] = initial_data[i+k*Kx][28*(j):28*(j+1)]

    return merged_data


def get_tab10():

    tab10_cmap = matplotlib.cm.tab10
    tab_norm = matplotlib.colors.Normalize(vmin=0, vmax=10)

    return tab10_cmap, tab_norm
