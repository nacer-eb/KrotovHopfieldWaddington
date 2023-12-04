# Generalized Hopfield Network + Waddington

Code, figures, movies, and examples for the [Waddington differentiation and saddle bifurcation for prototype learning](https://arxiv.org/) paper, by Nacer Eddine Boukacem, Allen Leary, Robin Thériault, Felix Gottlieb, Madhav Mani, and Paul Fran&#231;ois. 

## Installation

To install simply

    git clone https://github.com/nacer-eb/KrotovHopfieldClean.git
    
and

    pip install numpy, matplotlib, scipy, umap-learn, tensorflow
    
Note that `tensorflow` is used only to fetch the MNIST data. 

To compile the numerical solver, run
	
	make

in the C_module directory. 

## Figures

The figure directories are numbered in the same order as the main paper. The directories for main Figures are written as Figure X (e.g. Figure 1). 
Directories for Supplementary Figures are written as Figure XS.Description (e.g. Figure 5S.Nullclines is a supplementary figure based on Figure 5, which shows more nullcline examples).
Movies follow a similar naming scheme, by adding 'M' e.g. Figure 3M contains the movie(s) based on Figure 3. 

In all cases, the Figure directories contain both the Figure/Movie, the code necessary to generate the data as well as the code necessary to generate the Figures themselves. Typically, data generating code is contained in "main_gen.py". 

For convenience, the main figures and the movies referred to in the paper are also in the All_Main_Figures/ directory and the All_Movies/ directory.

### Instructions for reproducibility 

To reproduce any figure using UMAP plots, first generate the UMAP model/embedding. This can be done by going into the defaults/ folder and running:
	
	python generate_umap_model.py
	
This must be done only once. It is recommended that you run this script in the same python environment as the one which you use to generate figures. Then, you can go in the 'Figure X' directory of your choice, run the data generation script 

	python main_gen.py --poolsize NUMBER_OF_PROCESSES
	
If you benefit from a multi-core system, you may set NUMBER_OF_PROCESSES to a number proportional to your threadcount. Otherwise simply run

	python main_gen.py
	
Then run the figure generating script (generally Figure_X.py).

### Possible issue

When running any figure code for the first time, the KrotovV2_utils module will create a mnist_data directory and fetch the mnist database, by using tensorflow.keras, this generally works with no user input required. However, on some systems there is a known [issue](https://github.com/tensorflow/tensorflow/issues/33285) with keras being unable to verify the HTTPS certificates. You can follow [this](https://github.com/tensorflow/tensorflow/issues/33285#issuecomment-541417311) and [this](https://github.com/tensorflow/tensorflow/issues/33285#issuecomment-541417311) to fix it.


## Modules

### The main module

Contains all the code necessary to run the network. 

If you are not inside the main_module directory, you must append the main_module directory to your path. For instance, if your working directory is on the same level as the main_module directory. 

```python
import sys
sys.path.append('../')
```

Then, use

```python
from main_module.KrotovV2 import *
```

to import network class.

#### Generating the network object

To generate the network object, use,

```python
net = KrotovNet()
```
    
the KrotovNet class contains 15 optional parameters (see demo/ for a working example).

#### System parameters

`Kx, Ky` define the size of your network; `Kx*Ky` is the total number of memories, individual `Kx` and `Ky` are used to plot the grid of memories. (DEFAULT: 10 $\times$ 10)

`n_deg` is the power on the dot product; $\langle M \vert A \rangle ^n$. (DEFAULT: 30)

`m_deg` is the power on the cost function. Generally, `m_deg` is set to the same value as `n_deg`. (DEFAULT: 30)

`M` is the number of training samples per miniBatchs. (DEFAULT: 1000)

`nbMiniBatchs` is the number of miniBatchs. (DEFAULT: 60)

`momentum` is the standard ML momentum; reapplies part of the previous time step (this is consistent with Krotov-Hopfield [paper](https://arxiv.org/abs/1606.01164)). (DEFAULT: 0.6)

`rate` is the learning rate used in training. (DEFAULT: 0.0008)

`temp` is the factor (sometimes written $T$) used to 'renormalize' computation; $\tanh \Big( \frac{\langle M \vert A \rangle ^n}{T} \Big) $. (DEFAULT: 600)

`rand_init_mean` is the mean of the normal distribution used to initialize the memories and labels. (DEFAULT: -0.03)

`rand_init_std` is the standard deviation of the normal distribution used to initialize the memories and labels. (DEFAULT: 0.03)

`initHiddenDetectors` is the input array you want to use as initial conditions instead of Gaussian for the labels/hidden detectors - if None, 
it uses Gaussian random initial condition. (DEFAULT: None)

`initVisibleDetectors` is the input array you want to use as initial conditions instead of Gaussian for the memories/visible detectors - if None, 
it uses Gaussian random initial condition. (DEFAULT: None)

`selected_digits` is an array which specifies the digits you want to include in the training data. (DEFAULT: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

`useClipping` is a boolean which specifies whether you want to use clipping on the memory. If False, memories are normalized by dividing by the largest pixel greater than one, this is more practical analytically. (DEFAULT: False - i.e. uses normalization)

#### Training, plotting and saving

To train the network, use 

```python
net.train_plot_update(3500)
```

the only required argument is the number of epochs of training (for instance, 3500). Alternately, the following are the main optional arguments `isPlotting` plots the network at each epoch if True (DEFAULT: True), `isSaving` saves the network data (includes all epochs) if True (DEFAULT: False), `saving_dir` the directory in which to save the network (DEFAULT: None).


### The viewer module

If you choose to save the network data, the resulting .npz file can be read using the viewer module. 

```bash
python KV_window.py
```

The above (or equivalent for your python environment) starts the GUI. 

![Screenshot of view_module](viewer_module/Screenshot.png)

You can open any network training file using the "Open file..." button. The player allows for jump to the beginning of training (⏮), move back by one frame (affected by speed) (⏴), play/pause (⏯), move forward by a frame (affected by speed) (⏵), or jump to the end (⏭). In the bottom right, a slider can be used to change the speed at which training is played back.

### The nullcline module

The nullcline module calculates the $\nabla$ quantities, and the time derivative. An example of how to plot the nullclines using countour plots is included in the same directory. Note that the nullclince module is used only for the first saddles; the final states of the 1-memory system. 

### The single memory dynamics module

This is a more general version of the nullcline module. Unlike the nullcline module which is fixed to $\vert \alpha_{ \vert A \rangle } \vert + \vert \alpha_{\vert B \rangle} \vert = 1$, this module compute the entire dynamics of a 1-memory system from the initial condition, hyperparameters and training data. This module is here to verify/validate the dynamics derived in the supplemental materials. Note for simplicity this module assumes $\ell_{\gamma} = -1$ for initial conditions, but may be generalized.


