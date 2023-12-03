//
// Created by Nacer on 2023-01-23.
//

#include "Solution.h"
#include <stdlib.h>

// Dynamical Constructor (never used i think)
void Solution__new_solution(Solution* sol, double n, double T, double alpha, double beta, double l_0) {
    sol = (Solution*) malloc(sizeof(Solution));

    sol->n = n;
    sol->T = T;

    sol->alpha = alpha;
    sol->beta = beta;
    sol->l_0 = l_0;
}

// Constructor-ish. I will use the SolutionSet anyway so no 'real' dynamical allocation.
void Solution__init_solution(Solution* solution, double n, double T, double alpha, double beta, double l_0) {
    solution->n = n;
    solution->T = T;

    solution->alpha = alpha;
    solution->beta = beta;
    solution->l_0 = l_0;
}

void free_solution(Solution* solution) {
    free(solution);
}
