//
// Created by Nacer on 2023-01-23.
//

#include "SolutionSet.h"
#include "Solution.h"
#include <stdlib.h>
#include <stdio.h>
#include<windows.h>

// Constructor
SolutionSet* new_SolutionSet(char* directory_save) {
    SolutionSet* set = malloc(sizeof(SolutionSet));

    set->directory_save = directory_save;
    set->save_number = 0;
    set->currentSize = 0;

    return set;
}

void SolutionSet__addSolution(SolutionSet* set, double n, double T, double alpha, double beta, double l_0) {

    if(set->currentSize == MAX_SIZE) {
        printf("YOU HAVE REACHED MAXIMUM CAPACITY FOR THE SOLUTION SET!\n ");
        SolutionSet__saveSolutions(set);
        set->save_number += 1;

        printf("Temporary save complete... \n Resetting and continuing...\n");
        set->currentSize = 0;

    }

    // This is fun
    Solution__init_solution(&(set->solution[set->currentSize]), n, T, alpha, beta, l_0);

    set->currentSize += 1;

}


void SolutionSet__listSolutions(SolutionSet* set) {
    for (int i = 0; i < set->currentSize; ++i) {
        printf("Solution %d || (n=%lf, T=%lf) : alpha=%lf, beta=%lf, l_0=%lf \n",
                i + MAX_SIZE*set->save_number, set->solution[i].n, set->solution[i].T, set->solution[i].alpha, set->solution[i].beta, set->solution[i].l_0);
    }
}

void SolutionSet__saveSolutions(SolutionSet* set) {


    CreateDirectory(set->directory_save, NULL);

    int needed = snprintf(NULL, 0, "%s/save%d.dat", set->directory_save, set->save_number);
    char* filepath = (char *) malloc(needed + 1);
    snprintf(filepath, needed+1, "%s/save%d.dat", set->directory_save, set->save_number);

    FILE *output;

    output = fopen(filepath, "w");
    fprintf(output, "Solution #, n, T, alpha, beta, l_0\n");
    for (int i = 0; i < set->currentSize; ++i) {
        fprintf(output,"%d, %lf, %lf, %lf, %lf, %lf\n",
                i + MAX_SIZE*set->save_number, set->solution[i].n, set->solution[i].T, set->solution[i].alpha, set->solution[i].beta, set->solution[i].l_0);
    }
    fclose(output);
}

// Destructor
void freeSolutionSet(SolutionSet* set) {
    free(set);
}
