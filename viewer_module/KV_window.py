import sys
sys.path.append('../')


import tkinter as tk

import numpy as np
import time

from tkinter import font as tkFont
from tkinter import filedialog as tkFileDialog


import matplotlib.pyplot as plt
from matplotlib import animation as anim
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)

from viewer_module.KV_player import *

class KV_window:
    def __init__(self, player):

        # Window init
        self.window = tk.Tk() # Darkmode is not worth it

        
        self.window.geometry("1080x720+0+0")

        # Reference the player
        self.player = player

        # Menu
        self.menu = tk.Menu(self.window)
        
        self.filemenu = tk.Menu(self.menu, tearoff=0)
        self.filemenu.add_command(label="Open file...", command=self.open_file)
        self.menu.add_cascade(label="File", menu=self.filemenu)

        
        self.viewmenu = tk.Menu(self.menu, tearoff=0)
        self.viewmenu.add_command(label="Init. Cond.", command=self.show_init_condition, state="disabled")
        self.viewmenu.add_command(label="Training Data", command=self.show_minibatchs, state="disabled")
        self.viewmenu.add_command(label="Labels over time", command=self.show_labels, state="disabled")
        self.menu.add_cascade(label="View", menu=self.viewmenu)

        self.menu.add_command(label="Reset", command=self.reset)
        
        self.menu.add_command(label="Exit", command=self.close)
        
        
        self.window.config(menu=self.menu)
        

        # The exit binds         
        self.window.bind('<Control-w>', self.close)
        self.window.bind('<Control-c>', self.close)

        
        # The main canvas
        self.main_canvas = FigureCanvasTkAgg(self.player.main_fig, self.window)
        self.main_canvas.draw()
        
        self.main_canvas.get_tk_widget().place(relheight=0.9, relwidth=0.8, relx=0, rely=0)

        self.secondary_canvas = [0]*10
        # The secondary canvases
        for i in range(0, 10):
            self.secondary_canvas[i] = FigureCanvasTkAgg(self.player.label_fig[i], self.window)
            self.secondary_canvas[i].draw()
            self.secondary_canvas[i].get_tk_widget().place(relheight=0.09, relwidth=0.2, relx=0.8, rely=0.09*i)
            

        # Defining Fonts for the buttons
        font_t13 = tkFont.Font(family='Times New Roman', size=13)
        font_t21 = tkFont.Font(family='Times New Roman', size=21)


        # Open file button
        self.open_file_button = tk.Button(self.window, text="Open file...", font=font_t13, command=self.open_file)
        self.open_file_button.place(relheight=0.05, relwidth=0.09, relx=0.01, rely=0.92)


        # The player buttons
        self.go_to_start_button = tk.Button(self.window, text="\u23EE", font=font_t21, borderwidth=0, command=self.player.go_to_start)
        self.go_to_start_button.place(relheight=0.05, relwidth=0.03, relx=0.45-0.03*2, rely=0.92)
        
        self.reverse_button = tk.Button(self.window, text="\u23F4", font=font_t21, borderwidth=0, command=self.player.reverse)
        self.reverse_button.place(relheight=0.05, relwidth=0.03, relx=0.45-0.03*1, rely=0.92)

        self.play_pause_button = tk.Button(self.window, text="\u23EF", font=font_t21, borderwidth=0, command=self.player.play_pause)
        self.play_pause_button.place(relheight=0.05, relwidth=0.03, relx=0.450-0.03*0, rely=0.92)

        self.forward_button = tk.Button(self.window, text="\u23F5", font=font_t21, borderwidth=0, command=self.player.forward)
        self.forward_button.place(relheight=0.05, relwidth=0.03, relx=0.450+0.03*1, rely=0.92)
        
        self.go_to_end_button = tk.Button(self.window, text="\u23ED", font=font_t21, borderwidth=0, command=self.player.go_to_end)
        self.go_to_end_button.place(relheight=0.05, relwidth=0.03, relx=0.450+0.03*2, rely=0.92)


        # Speed slider
        self.speed_slider_scale = tk.Scale(self.window, from_=1, to=100, orient=tk.HORIZONTAL, command=self.speed_slider_event)
        self.speed_slider_scale.place(relheight=0.04, relwidth=0.19, relx=0.8, rely=0.92)

        # Setting the title to reflect the state 
        self.player.window = self.window # Allowing this for the player to change the title state
        self.player.update_title_state()
        
        # The animation
        self.window.mainloop()


    
    def speed_slider_event(self, value):
        print("Slider moved!", value)
        self.player.set_speed(self.speed_slider_scale.get())
        
    
    def open_file(self):
        print("Open file...")
        filename = tkFileDialog.askopenfilename()
        
        if filename == '':
            return

        
        self.viewmenu.entryconfig("Init. Cond.", state="normal")
        self.viewmenu.entryconfig("Training Data", state="normal")
        self.viewmenu.entryconfig("Labels over time", state="normal")

        self.player.loadData(filename)
        self.reset()
        
        
    def close(self, event=None):
        self.window.destroy()

        
    # This is because animation objects lag behind and don't erase previously drawn files
    # Clean ;) Proud of this fix. 
    def reset(self, event=None):
        self.player.epoch = 0
        self.player.state = "Waiting for file..."
        
        for item in self.main_canvas.get_tk_widget().find_all():
            self.main_canvas.get_tk_widget().delete(item)

        for i in range(0, 10):    
            for item in self.secondary_canvas[i].get_tk_widget().find_all():
                self.secondary_canvas[i].get_tk_widget().delete(item)

        
        # The main canvas
        self.main_canvas = FigureCanvasTkAgg(self.player.main_fig, self.window)
        self.main_canvas.draw()
        
        self.main_canvas.get_tk_widget().place(relheight=0.9, relwidth=0.8, relx=0, rely=0)

        self.secondary_canvas = [0]*10
        # The secondary canvases
        for i in range(0, 10):
            self.secondary_canvas[i] = FigureCanvasTkAgg(self.player.label_fig[i], self.window)
            self.secondary_canvas[i].draw()
            self.secondary_canvas[i].get_tk_widget().place(relheight=0.09, relwidth=0.2, relx=0.8, rely=0.09*i)


    def show_init_condition(self):
        ic_message = "The network contains " + str(self.player.N) + " memories. \n" + \
            "The (n, m) = (" + str(self.player.meta_data[2]) + "," + str(self.player.meta_data[3]) + ").\n" + \
            "The training set size is " + str(self.player.meta_data[4]) + " ( " + str(self.player.meta_data[5]) + " minibatch(s) ).\n" + \
            "The momentum is " + str(self.player.meta_data[6]) + ".\n" + \
            "The training rate " + str(self.player.meta_data[7]) + ".\n" + \
            "The temperature is " + str(self.player.meta_data[8]) + ".\n" + \
            "The mean and std (if relevant) are: " + str(self.player.meta_data[9]) + " , " + str(self.player.meta_data[10]) + \
            "The selected digits are " + str(self.player.selected_digits)
        
        tk.messagebox.showinfo(title="Initial Conditions", message=ic_message)


    # Util
    def get_closest_to_root(self, M):
        sqrt_M = int(np.ceil(np.sqrt(M)))
        
        candidates_m_1 = np.flip(np.arange(1, sqrt_M, 1))
        
        for m_1 in candidates_m_1:
            if M%m_1 == 0:
                return np.maximum(m_1, int(M/m_1)), np.minimum(m_1, int(M/m_1))

            
    def show_minibatchs(self, event=None):        
        M = int(self.player.meta_data[4])
        nbMiniBatchs = int(self.player.meta_data[5])

        m_x, m_y = self.get_closest_to_root(M)
        for i in range(0, nbMiniBatchs):
            window_T = tk.Tk()
            window_T.wm_title("The training data")
        
            fig_T, ax_T = plt.subplots(1, figsize=(16, 9))
            ax_T.imshow(merge_data(self.player.T_data[i], m_x, m_y), cmap="bwr", vmin=-1, vmax=1)

            canvas_T = FigureCanvasTkAgg(fig_T, window_T)
            canvas_T.draw()
            canvas_T.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

            toolbar_T = NavigationToolbar2Tk(canvas_T, window_T)
            toolbar_T.update()
            canvas_T.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

            tk.mainloop()



    def show_labels(self, event=None):
        window_L = tk.Tk()
        window_L.wm_title("The selected digits label")
        
        fig_L, ax_L = plt.subplots(1, figsize=(16, 9))
        
        print(np.shape(self.player.L_data))
        for d in self.player.selected_digits:
            ax_L.plot(self.player.L_data[d][:, 0], label=d, color=plt.cm.get_cmap('tab20', 10)(d))
            for n in range(1, self.player.N):
                ax_L.plot(self.player.L_data[d][:, n], color=plt.cm.get_cmap('tab20', 10)(d))
        ax_L.legend()
        
        canvas_L = FigureCanvasTkAgg(fig_L, window_L)
        canvas_L.draw()
        canvas_L.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        
        toolbar_L = NavigationToolbar2Tk(canvas_L, window_L)
        toolbar_L.update()
        canvas_L.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        
        tk.mainloop()
        


KVP = KV_player()        
KVW = KV_window(KVP)

