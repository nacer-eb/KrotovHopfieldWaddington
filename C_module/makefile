C_DIR = C_interface
CUDA_DIR = GPU_library

C_TARGETS = main.o Simulation.o Solution.o SolutionSet.o vector_mtx.o



solver : simulation_lib $(C_TARGETS)
	gcc -o solver $(C_TARGETS) -L./ -l simulation_lib && make clean

simulation_lib: 
	cd $(CUDA_DIR) && make

main.o : $(C_DIR)/main.c
	gcc -c $(C_DIR)/main.c


Simulation.o : $(C_DIR)/Simulation.c $(C_DIR)/Simulation.h
	gcc -c $(C_DIR)/Simulation.c


Solution.o : $(C_DIR)/Solution.c $(C_DIR)/Solution.h
	gcc -c $(C_DIR)/Solution.c


SolutionSet.o : $(C_DIR)/SolutionSet.c $(C_DIR)/SolutionSet.h
	gcc -c $(C_DIR)/SolutionSet.c


vector_mtx.o : $(C_DIR)/vector_mtx.c $(C_DIR)/vector_mtx.h
	gcc -c $(C_DIR)/vector_mtx.c




clean :
	rm $(C_TARGETS) 
