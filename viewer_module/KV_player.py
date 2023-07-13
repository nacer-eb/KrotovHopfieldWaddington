import sys
sys.path.append('../')

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import animation as anim

from main_module.KrotovV2_utils import *

class KV_player:
    def __init__(self):
        
        
        # For the memory
        self.main_fig, self.main_ax = plt.subplots(figsize=(16, 16))
        self.main_animation = 0
        self.main_im = 0
        
        self.main_ax.set_yticklabels([])
        self.main_ax.set_xticklabels([])
        self.main_fig.tight_layout()
        
        self.label_fig, self.label_ax, self.label_im = [0]*10, [0]*10, [0]*10
        for i in range(0, 10):
            self.label_fig[i], self.label_ax[i] = plt.subplots(figsize=(16, 4))
            self.label_ax[i].set_yticklabels([])
            self.label_ax[i].set_xticklabels([])

        self.playState = False
        
        
        # Now we init data & parameters
        self.M_data = 0
        self.L_data = [0]*10
        self.T_data = 0
        self.meta_data = 0
        self.selected_digits = 0
        
        self.Nx, self.Ny, self.N = 0, 0, 0
        self.N_range = 0
        
        self.epoch = 0
        self.last_frame = 99
        self.speed = 1
        
        # Init animation pointers
        self.filename = None

        self.state = "Waiting for file..."
        self.window = 0

        
    def loadData(self, filename):        
        self.filename = filename

        # Resetting axis
        self.main_ax.clear()

        if self.main_animation != 0:
            self.main_ax.clear()
            self.main_im.remove()
            self.main_animation._stop()

            for i in range(0, 10):
                self.label_ax[i].clear()
                self.label_im[i].remove()
            
        self.state = "Loading file..."
        self.update_title_state()

        data = np.load(self.filename)
        
        self.meta_data = data['init_array']
        self.selected_digits = data['selected_digits']
        
        self.Nx, self.Ny, self.N = int(self.meta_data[0]), int(self.meta_data[1]), int(self.meta_data[0]*self.meta_data[1])
        
        # Memory stuff
        self.M_data = data['M']
        self.T_data = data['miniBatchs_images']
        
        self.last_frame = np.shape(self.M_data)[0]-1
        self.main_im = self.main_ax.imshow(merge_data(self.M_data[0], self.Nx, self.Ny ), cmap="bwr", vmin=-1, vmax=+1)

        # Label stuff
        self.N_range = np.arange(0, self.N, 1)
        for i in range(0, 10):
            self.L_data[i] = data['L'][:, :, i]
            self.label_ax[i].set_xlim([-1.1, 1.1])
            self.label_ax[i].set_ylim([-1, self.N])
            self.label_im[i] = self.label_ax[i].scatter(self.L_data[i][0, :], self.N_range, s=2, cmap="tab20", c=self.N_range)
        
        self.loadAnimations()
        
        
        
    def loadAnimations(self):
        self.state = "Loading animation..."
        self.update_title_state()
        
        self.main_animation = anim.FuncAnimation(self.main_fig, self.update, interval=10, blit=True, repeat=False)

        self.state = "Paused."
        self.update_title_state()


    def set_speed(self, new_speed):
        self.speed = new_speed


    def update_title_state(self):
        self.window.title("Krotov Viewer - Current State: " + self.state + " - File: '"+ str(self.filename) +"' - Epoch: " + str(self.epoch))


    def refresh_canvases(self):
        self.main_fig.canvas.draw()
        self.main_fig.canvas.flush_events()

        for i in range(0, 10):
            self.label_fig[i].canvas.draw()
            self.label_fig[i].canvas.flush_events()
        
    def update(self, i):
        self.update_title_state()
        
        if self.playState:
            self.epoch += self.speed # Funny. Only iterate if playing.

        if self.epoch <= 0:
            self.epoch = 0
            
        if self.epoch >= self.last_frame:
            self.epoch = self.last_frame


        self.main_im.set_data(merge_data(self.M_data[self.epoch], self.Nx, self.Ny ) )
        for i in range(0, 10):
            self.label_im[i].set_offsets(np.c_[self.L_data[i][self.epoch, :], self.N_range])
        
        return self.main_im, *self.label_im
    
    def go_to_start(self):
        if self.filename is None:
            return
        
        self.epoch = 0
        
        self.playState=False
        self.main_animation.pause()
        
        self.state = "Paused."        
        self.update(self.epoch)

        self.refresh_canvases()


    def reverse(self):
        if self.filename is None:
            return
        
        if self.epoch == 0:
            return

        self.epoch += -self.speed
        self.playState=False
        self.main_animation.pause()
        
        self.state = "Paused."      
        self.update(self.epoch)
        
        self.refresh_canvases()
        

    def play_pause(self):
        if self.filename is None:
            return

        
        if not self.playState:
            self.state = "Playing..."     # Recall you toggle so if it wasn't playing now it is..
            self.main_animation.resume()
            self.update_title_state()

        if self.playState:
            self.state = "Paused."
            self.main_animation.pause()
            self.update_title_state()

        self.playState = not self.playState #Toggle

    def forward(self):
        if self.filename is None:
            return
        

        if self.epoch == self.last_frame:
            return
        
        self.epoch += self.speed
        self.playState=False
        self.main_animation.pause()

        self.state = "Paused."
        self.update(self.epoch)

        self.refresh_canvases()

    def go_to_end(self):
        if self.filename is None:
            return
        

        self.epoch = self.last_frame
        self.playState=False
        self.main_animation.pause()

        self.state = "Paused."
        self.update(self.epoch)

        self.refresh_canvases()
        

