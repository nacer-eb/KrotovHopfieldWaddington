#ifndef KROTOVHOPFIELD_NUMERICALSOLVER_CUDALIB_SIMULATION_LIB_H
#define KROTOVHOPFIELD_NUMERICALSOLVER_CUDALIB_SIMULATION_LIB_H

extern "C" __declspec(dllexport) void CUDA__findSolutions(int* cpu_isSolution, int size, float d_AA, float d_AB, float d_BB, float n, float T);

#endif //KROTOVHOPFIELD_NUMERICALSOLVER_CUDALIB_SIMULATION_LIB_H
