//
// Created by Nacer on 2023-01-23.
//

#include "Simulation.h"
#include "vector_mtx.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "SolutionSet.h"
#include "simulation_lib.h"
//Constructor
Simulation* new_simulation(double d_AA, double d_AB, double d_BB, double res_al, double T, double n, BETA_TYPE bt, SolutionSet* fullSet) {
    Simulation* sim = malloc(sizeof(Simulation));

    sim->d_AA = d_AA;
    sim->d_AB = d_AB;
    sim->d_BB = d_BB;

    sim->res_al = res_al;
    sim->T = T;
    sim->n = n;

    if (bt != POSITIVE_BETA && bt != NEGATIVE_BETA) {
        printf("ERROR : UNKNOWN BETA_TYPE!!!");
        exit(-1);
    }

    sim->bt = bt;

    int i_max = (int) 2.0/sim->res_al;
    sim->alpha = malloc_vector_double(i_max, 0.0);
    sim->l_0 = malloc_vector_double(i_max, 0.0);

    for (int i = 0; i < i_max; ++i) {
        sim->alpha[i] = -1.0 + sim->res_al*i;
        sim->l_0[i] = -1.0 + sim->res_al*i;

    }

    sim->set = fullSet;//new_SolutionSet();

    return sim;
}


double ReLU(double x) {
    if (x > 0) {
        return x;
    }
    return 0;
}

//Functions
double Simulation__calc_d_A(Simulation *sim, double alpha, double beta, double l_0) {
    return ReLU((alpha*sim->d_AA + beta*sim->d_AB)/sim->T);
}
double Simulation__calc_d_B(Simulation *sim, double alpha, double beta, double l_0){
    return ReLU((alpha*sim->d_AB + beta*sim->d_BB)/sim->T);
}


double Simulation__calc_dt_tilde_alpha(Simulation *sim, double alpha, double beta, double l_0) {
    double d_A = Simulation__calc_d_A(sim, alpha, beta, l_0);

    double d_A_n = pow(d_A, sim->n);

    double dt_tilde_alpha = l_0*pow( 1 - tanh(l_0 * d_A_n), 2.0*sim->n - 1) * (1 - pow(tanh(l_0 * d_A_n), 2.0)) * pow(d_A, sim->n-1)
                            + 4*pow( 1 - tanh( d_A_n), 2.0*sim->n - 1) * (1 - pow(tanh(d_A_n), 2.0)) * pow(d_A, sim->n-1);

    return dt_tilde_alpha;
}

double Simulation__calc_dt_tilde_beta(Simulation *sim, double alpha, double beta, double l_0) {
    double d_B = Simulation__calc_d_B(sim, alpha, beta, l_0);

    double d_B_n = pow(d_B, sim->n);

    double dt_tilde_beta = -l_0*pow( 1 + tanh(l_0 * d_B_n), 2.0*sim->n - 1) * (1 - pow(tanh(l_0 * d_B_n), 2.0)) * pow(d_B, sim->n-1)
                           + 4*pow( 1 - tanh( d_B_n), 2.0*sim->n - 1) * (1 - pow(tanh(d_B_n), 2.0)) * pow(d_B, sim->n-1);

    return dt_tilde_beta;
}

double Simulation__calc_dt_tilde_l_0(Simulation *sim, double alpha, double beta, double l_0) {
    double d_A = Simulation__calc_d_A(sim, alpha, beta, l_0);
    double d_B = Simulation__calc_d_B(sim, alpha, beta, l_0);

    double d_A_n = pow(d_A, sim->n);
    double d_B_n = pow(d_B, sim->n);

    double dt_tilde_l_0 = pow( 1 - tanh(l_0 * d_A_n), 2.0*sim->n - 1) * (1 - pow(tanh(l_0 * d_A_n), 2.0)) * pow(d_A, sim->n)
                          - pow( 1 + tanh(l_0 * d_B_n), 2.0*sim->n - 1) * (1 - pow(tanh(l_0 * d_B_n), 2.0)) * pow(d_B, sim->n);

    return dt_tilde_l_0;
}


int sign(double x) {

    if(!isfinite(x)) { // Default is to ignore infinites and set them to zero..
        return 0;
    }

    if (x < 0) {
        return -1;
    }
    if (x > 0) {
        return 1;
    }
    return 0;
}


void Simulation__findSolutions(Simulation *sim) {


    int iter_max = (int) 2.0/sim->res_al;


    char ** alpha_boundary_condition = malloc_mtx_char(iter_max, iter_max, 0);
    char ** alpha_condition = malloc_mtx_char(iter_max, iter_max, 0);
    char ** l_0_condition = malloc_mtx_char(iter_max, iter_max, 0);

    printf("A\n");
    for (int i = 0; i < iter_max; ++i) {
        for (int j = 0; j < iter_max; ++j) {

            double alpha, beta, l_0, dt_tilde_alpha, dt_tilde_beta, dt_tilde_l_0;

            alpha = sim->alpha[i];
            if (sim->bt == POSITIVE_BETA) {
                beta = 1 - fabsf(alpha);
            }
            if (sim->bt == NEGATIVE_BETA) {
                beta = -1 + fabsf(alpha);
            }
            l_0 = sim->l_0[j];

            dt_tilde_alpha = Simulation__calc_dt_tilde_alpha(sim, alpha, beta, l_0);
            dt_tilde_beta = Simulation__calc_dt_tilde_beta(sim, alpha, beta, l_0);
            dt_tilde_l_0 = Simulation__calc_dt_tilde_l_0(sim, alpha, beta, l_0);

            double dt_alpha_over_alpha = dt_tilde_alpha/alpha;
            double dt_beta_over_beta = dt_tilde_beta/beta;

            // Trying to build a bool essentially.
            alpha_boundary_condition[i][j] = sign(sign(alpha)*dt_alpha_over_alpha + sign(beta)*dt_beta_over_beta);
            alpha_boundary_condition[i][j] = (1-alpha_boundary_condition[i][j]*sign(alpha_boundary_condition[i][j]-1)); // 0 -> 0, 1-> 0, -1->-1



            alpha_condition[i][j] = sign(dt_alpha_over_alpha - dt_beta_over_beta);
            l_0_condition[i][j] = sign(dt_tilde_l_0);



        }

        l_0_condition[i][iter_max-1] = -l_0_condition[i][iter_max-1]*sign(l_0_condition[i][iter_max-1] - 1); // 0 -> 0, 1-> 0, -1->-1
        l_0_condition[i][0] = l_0_condition[i][0]*sign(l_0_condition[i][0] + 1); // 0 -> 0, 1-> 1, -1->0

        l_0_condition[iter_max-1][i] = -l_0_condition[iter_max-1][i]*sign(l_0_condition[iter_max-1][i] - 1); // 0 -> 0, 1-> 0, -1->-1
        l_0_condition[0][i] = l_0_condition[0][i]*sign(l_0_condition[0][i] + 1); // 0 -> 0, 1-> 1, -1->0


    }

    for (int i = 1; i < iter_max-1; ++i) {
        for (int j = 1; j < iter_max-1; ++j) {

            double alpha, beta, l_0;
            int alpha_crit, l_0_crit;

            alpha = sim->alpha[i];
            if (sim->bt == POSITIVE_BETA) {
                beta = 1 - fabsf(alpha);
            }
            if (sim->bt == NEGATIVE_BETA) {
                beta = -1 + fabsf(alpha);
            }
            l_0 = sim->l_0[j];

            alpha_crit = abs(alpha_condition[i+1][j] + alpha_condition[i-1][j] + alpha_condition[i][j+1] + alpha_condition[i][j-1])<3;
            l_0_crit = abs(l_0_condition[i+1][j] + l_0_condition[i-1][j] + l_0_condition[i][j+1] + l_0_condition[i][j-1])<3;

            //*(1-l_0_condition[i][j])
            if( l_0_crit==1 && alpha_crit==1 && alpha_boundary_condition[i][j] == 1) {
                printf("A Solution was found!!\n");
                SolutionSet__addSolution(sim->set, sim->n, sim->T, alpha, beta, l_0);

                printf("%d \n", alpha_condition[i+1][j] + alpha_condition[i-1][j] + alpha_condition[i][j+1] + alpha_condition[i][j-1]);
            }
        }
    }
    printf("C\n");

}

void Simulation__CUDAfindSolutions(Simulation *sim) {


    int size =  (int) 2.0/sim->res_al;

    int* isSolution;
    isSolution = malloc_vector_int(size*size, 0);

    if(sim->bt == POSITIVE_BETA) {
        CUDA__findSolutions(isSolution, size, sim->d_AA, sim->d_AB, sim->d_BB, sim->n, sim->T);

        for (int index = 0; index < size*size; ++index) {
            if (isSolution[index] == 1) {
                int i = index%size;
                int j = (int) floor(index*1.0/size);

                float alpha, beta, l_0;

                alpha = -1.0 + i*(2.0/size);
                beta = 1 - fabsf(alpha); // Positive beta
                l_0 = -1.0 + j*(2.0/size);

                SolutionSet__addSolution(sim->set, sim->n, sim->T, alpha, beta, l_0);

            }
        }
    }

    // Cool trick, just swap d_AA and d_BB and make invert swap alpha and beta and switch sign on l_0
    if(sim->bt == NEGATIVE_BETA) {
        CUDA__findSolutions(isSolution, size, sim->d_BB, sim->d_AB, sim->d_AA, sim->n, sim->T);

        for (int index = 0; index < size*size; ++index) {
            if (isSolution[index] == 1) {
                int i = index%size;
                int j = (int) floor(index*1.0/size);

                float alpha, beta, l_0;

                alpha = -1.0 + i*(2.0/size);
                beta = 1 - fabsf(alpha); // Positive beta
                l_0 = -1.0 + j*(2.0/size);

                SolutionSet__addSolution(sim->set, sim->n, sim->T, beta, alpha, -l_0);

            }
        }
    }


    free(isSolution);
}

//Destructor
void free_simulation(Simulation* sim) {
    free(sim->alpha);
    free(sim->l_0);
    // freeSolutionSet(sim->set); Free seperately as it should be preexinsting.
    free(sim);
}