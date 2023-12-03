//
// Created by Nacer on 2023-01-23.
//

#ifndef KROTOVHOPFIELD_NUMERICALSOLVER_SIMULATION_H
#define KROTOVHOPFIELD_NUMERICALSOLVER_SIMULATION_H

#include "SolutionSet.h"

typedef enum {
    POSITIVE_BETA,
    NEGATIVE_BETA
} BETA_TYPE;

typedef struct{

    int d_AA, d_AB, d_BB;

    BETA_TYPE bt;

    double res_al; //resolution for alpha,beta and l_o

    double* alpha;
    double* l_0;

    double T, n;

    SolutionSet *set;
} Simulation;

//Constructor
Simulation* new_simulation(double d_AA, double d_AB, double d_BB, double res_al, double T, double n, BETA_TYPE bt, SolutionSet* fullSet);

//Functions
double Simulation__calc_d_A(Simulation *sim, double alpha, double beta, double l_0);
double Simulation__calc_d_B(Simulation *sim, double alpha, double beta, double l_0);

double Simulation__calc_dt_tilde_alpha(Simulation *sim, double alpha, double beta, double l_0);
double Simulation__calc_dt_tilde_beta(Simulation *sim, double alpha, double beta, double l_0);
double Simulation__calc_dt_tilde_l_0(Simulation *sim, double alpha, double beta, double l_0);

void Simulation__findSolutions(Simulation *sim);
void Simulation__CUDAfindSolutions(Simulation *sim);

//Destructor
void free_simulation(Simulation* sim);


#endif //KROTOVHOPFIELD_NUMERICALSOLVER_SIMULATION_H
