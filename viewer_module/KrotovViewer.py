import os
from os import path
import time
import numpy as np
import imageio
import matplotlib.pyplot as plt

import matplotlib
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 22}

matplotlib.rc('font', **font)


import sys
sys.path.append('../')

from main_module.KrotovV2_utils import *



with open('KVTitle.txt', 'r') as f:
    print(f.read())

ratio = 2 # fps
user_input = ""
while user_input != "exit":
    print("Input the mode&file you'd like to use&view (or enter 'help' for help) ...")
    user_input = input(" > ")

    if user_input == "exit":
        continue
    
    if user_input == "ls":
        print("File list:")
        for file in os.listdir("."):
            if file.endswith(".npz"):
                print("\t - ", file)
        print("")
        continue

    # Temporary solution
    if user_input == "ratio":
        print("What do you want the ratio to be? Currently:", ratio)
        user_input = input(" > ")
        ratio = int(user_input)
        continue
    
    if user_input == "help":
        print("Help menu:")
        print("\t ls : Lists all data files")
        print("\t ratio : Change the fps ratio (more is faster)")
        print("\t stream -file: Enter stream mode and load file / does not save")
        print("\t save -file -outputfile: Enter saving mode, load file and save in outputfile / .gif or .avi")
        print("\t exit : Leave this program \n")
        continue

    split_input = user_input.split(" -")
    if len(split_input) < 2 or len(split_input) > 3:
        print("\033[1;31m ERROR :\033[0m There's a syntax issue, I expect '[stream/save] -[inputfile] -[outputfile (optional)]' \n")
        continue
        
    mode = split_input[0]
    inputfile = split_input[1]

    # work around the stream case... only used in the save case anyway... also allows for default filename
    outputfile = "output.gif"
    if len(split_input) == 3:
        outputfile = split_input[2]

        # Checking for the right extension
        outputfile_ext = outputfile.split(".")[1]
        if outputfile_ext != "gif" and outputfile_ext != "avi" and outputfile_ext != "mov":
            print("\033[1;31m ERROR :\033[0m The outputfile extension '"+ outputfile_ext + "' is not accepted. (Use gif or avi or mov) \n")
            continue


    # Check that the mode is correct
    if mode == "stream" or mode == "save":
        print("Now using mode " + mode + " !")
    else:
        print("\033[1;31m ERROR :\033[0m Not sure what kinda mode '" + mode + "' is...")
        print("\033[1;32m TIP :\033[0m I expect '[stream/save] -[inputfile] -[outputfile]' \n")
        continue

    # Check if file exits   
    if not path.exists(inputfile):
        print("\033[1;31m ERROR :\033[0m The input file '", inputfile, "' does not exist.")
        print("\033[1;32m TIP :\033[0m I expect '[stream/save] -[inputfile] -[outputfile]'")
        print("\033[1;32m TIP :\033[0m If the syntax is correct, make sure the file name is too (enter ls). \n")
        continue
    else:
        print("Loading...")
        raw_data = np.load(inputfile)
        init_array, selected_digits, M, L = raw_data['init_array'], raw_data['selected_digits'], raw_data['M'], raw_data['L']

                
        plt.ion()
        fig, ax = plt.subplots(1, 1, figsize=(16, 9))

        im = ax.imshow(merge_data(M[0], int(init_array[0]), int(init_array[1])), cmap="bwr", vmin=-1, vmax=1)

        if mode == "save":
            writer = imageio.get_writer("./"+outputfile, mode='I')

        
        for i_ in range(len(M)//ratio):
            i = ratio*i_
            im.set_data(merge_data(M[i], int(init_array[0]), int(init_array[1])))
            ax.set_title(str(i))
            plt.pause(0.01)

            print(" Labels of the selected digits. ")
            print(L[i][:, selected_digits])
            print(" ")
            print(" ")
            
            if mode == "save":
                plt.savefig("tmp.png")
                image = imageio.imread("tmp.png")
                writer.append_data(image)

        plt.clf()
        plt.cla()
        plt.close()

        if mode == "save":
            writer.close()

     
# Exit steps
with open('KVEnd.txt', 'r') as f:
    print(f.read())

time.sleep(1)

os.system('clear')

