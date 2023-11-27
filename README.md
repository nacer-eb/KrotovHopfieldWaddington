# Generalized Hopfield Network

Code, figures, movies, and examples for the [Waddington differentiation and saddle bifurcation for prototype learning](https://arxiv.org/) paper. 

## Installation

## Figures

## Modules

### The main module

Contains all the code necessary to run the network. 

If you are not inside the main_module directory, you must append the main_module directory to your path. For instance, if your working directory is on the same level as the main_module directory.

    import sys
    sys.path.append('../')

Then, use
    
    from main_module.KrotovV2 import *
    
to import network class.

#### Generating the network object

To generate the network object, use,

    net = KrotovNet()
    
the KrotovNet class contains 15 optional parameters (see demo/ for a working example).

#### System parameters

`Kx, Ky` define the size of your network; `Kx*Ky` is the total number of memories, individual `Kx` and `Ky` are used to plot the grid of memories.

`n_deg` is the power on the dot product; $\langle M \vert A \langle ^n$


#### Training, plotting and saving


### The viewer module

### The nullcline module

### The simplified system module


