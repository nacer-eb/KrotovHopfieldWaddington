//
// Created by Nacer on 2023-01-23.
//

#ifndef KROTOVHOPFIELD_NUMERICALSOLVER_SOLUTION_H
#define KROTOVHOPFIELD_NUMERICALSOLVER_SOLUTION_H

typedef struct {
    double n, T;
    double alpha, beta, l_0;
} Solution;

// Constructor (never needed)
void Solution__new_solution(Solution* sol, double n, double T, double alpha, double beta, double l_0);

// Function
void Solution__init_solution(Solution* solution, double n, double T, double alpha, double beta, double l_0);

// Destructor
void free_solution(Solution* solution);

#endif //KROTOVHOPFIELD_NUMERICALSOLVER_SOLUTION_H
