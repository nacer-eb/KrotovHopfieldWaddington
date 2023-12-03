//
// Created by Nacer on 2023-01-23.
//

#include "vector_mtx.h"
#include <stdlib.h>
#include <stdio.h>

// Constructors

char * malloc_vector_char(int nmax, char default_value) {
    char* vector_char = (char *) malloc(nmax*sizeof(char));

    for(int i = 0; i < nmax; i++) {
        vector_char[i] = default_value;
    }

    return vector_char;
}
int * malloc_vector_int(int nmax, int default_value) {
    int* vector_int = (int *) malloc(nmax*sizeof(int));

    for(int i = 0; i < nmax; i++) {
        vector_int[i] = default_value;
    }

    return vector_int;
}
float * malloc_vector_float(int nmax, float default_value) {
    float* vector_float = (float *) malloc(nmax*sizeof(float));

    for(int i = 0; i < nmax; i++) {
        vector_float[i] = default_value;
    }

    return vector_float;
}

double * malloc_vector_double(int nmax, double default_value) {
    double* vector_double = (double *) malloc(nmax*sizeof(double));

    for(int i = 0; i < nmax; i++) {
        vector_double[i] = default_value;
    }

    return vector_double;
}


char ** malloc_mtx_char(int nmax, int mmax, char default_value) {
    char ** mtx_char = (char **) malloc(sizeof(char*)*nmax);

    for(int i = 0; i < nmax; i++) {
        mtx_char[i] = malloc_vector_char(mmax, default_value);
    }

    return mtx_char;
}

int ** malloc_mtx_int(int nmax, int mmax, int default_value) {
    int ** mtx_int = (int **) malloc(sizeof(int*)*nmax);

    for(int i = 0; i < nmax; i++) {
        mtx_int[i] = malloc_vector_int(mmax, default_value);
    }

    return mtx_int;
}


float ** malloc_mtx_float(int nmax, int mmax, float default_value) {
    float ** mtx_float = (float **) malloc(sizeof(float*)*nmax);

    for(int i = 0; i < nmax; i++) {
        mtx_float[i] = malloc_vector_float(mmax, default_value);
    }

    return mtx_float;
}

double ** malloc_mtx_double(int nmax, int mmax, double default_value) {
    double ** mtx_double = (double **) malloc(sizeof(double*)*nmax);

    for(int i = 0; i < nmax; i++) {
        mtx_double[i] = malloc_vector_double(mmax, default_value);
    }

    return mtx_double;
}



// Destructors

void free_mtx_char(char** mtx_char, int nmax) {
    for(int i = 0; i < nmax; i++) {
        free(mtx_char[i]);
    }
    free(mtx_char);
}

void free_mtx_int(int** mtx_int, int nmax) {
    for(int i = 0; i < nmax; i++) {
        free(mtx_int[i]);
    }
    free(mtx_int);
}

void free_mtx_float(float** mtx_float, int nmax) {
    for(int i = 0; i < nmax; i++) {
        free(mtx_float[i]);
    }
    free(mtx_float);
}
void free_mtx_double(double** mtx_double, int nmax) {
    for(int i = 0; i < nmax; i++) {
        free(mtx_double[i]);
    }
    free(mtx_double);
}