simulation_lib: simulation_lib.cu
	nvcc -o simulation_lib.dll --shared simulation_lib.cu && mv simulation_lib.dll ../simulation_lib.dll && make clean

clean :
	rm *.lib *.exp
