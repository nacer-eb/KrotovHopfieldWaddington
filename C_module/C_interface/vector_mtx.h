//
// Created by Nacer on 2023-01-23.
//

#ifndef KROTOVHOPFIELD_NUMERICALSOLVER_VECTOR_MTX_H
#define KROTOVHOPFIELD_NUMERICALSOLVER_VECTOR_MTX_H


// Constructors
char * malloc_vector_char(int nmax, char default_value);
int * malloc_vector_int(int nmax, int default_value);
float * malloc_vector_float(int nmax, float default_value);
double * malloc_vector_double(int nmax, double default_value);

char ** malloc_mtx_char(int nmax, int mmax, char default_value);
int ** malloc_mtx_int(int nmax, int mmax, int default_value);
float ** malloc_mtx_float(int nmax, int mmax, float default_value);
double ** malloc_mtx_double(int nmax, int mmax, double default_value);


// Destructors
void free_mtx_int(int** mtx_int, int nmax);
void free_mtx_double(double** mtx_double, int nmax);


#endif //KROTOVHOPFIELD_NUMERICALSOLVER_VECTOR_MTX_H
