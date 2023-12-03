//
// Created by Nacer on 2023-01-23.
//

#ifndef KROTOVHOPFIELD_NUMERICALSOLVER_SOLUTIONSET_H
#define KROTOVHOPFIELD_NUMERICALSOLVER_SOLUTIONSET_H

#include "Solution.h"


#define MAX_SIZE 500000*100//500000 // // Maximum number of solutions.
typedef struct {
    int currentSize;
    char* directory_save;
    int save_number;

    Solution solution[MAX_SIZE]; // Using the struct-hack ;)
} SolutionSet;


// Constructor
SolutionSet* new_SolutionSet(char* directory_save);

// Functions
void SolutionSet__addSolution(SolutionSet* set, double n, double T, double alpha, double beta, double l_0);
void SolutionSet__listSolutions(SolutionSet* set);
void SolutionSet__saveSolutions(SolutionSet* set);


// Destructor
void freeSolutionSet(SolutionSet* set);

#endif //KROTOVHOPFIELD_NUMERICALSOLVER_SOLUTIONSET_H
