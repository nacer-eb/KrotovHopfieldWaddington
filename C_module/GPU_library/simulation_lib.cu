#include "simulation_lib.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>


inline void gpuAssert(cudaError_t code, int line, bool abort = true) //Error checking
{
    const char* file = "simulation_lib.cu";

    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__device__ float ReLU(float x) {
    if (x < 0) {
        return 0;
    }
    return x;
}

__device__ int sign(float x) {
    if (x > 0) {
        return 1;
    }

    if (x < 0) {
        return -1;
    }

    return 0;

}

__device__ float calc_d_A(float d_AA, float d_AB, float d_BB, float alpha, float beta, float l_0, float n, float T) {
    return ReLU( (alpha*d_AA + beta*d_AB)/T );
}

__device__ float calc_d_B(float d_AA, float d_AB, float d_BB, float alpha, float beta, float l_0, float n, float T) {
    return ReLU( (alpha*d_AB + beta*d_BB)/T );
}

__device__ float calc_dt_tilde_alpha(float d_AA, float d_AB, float d_BB, float alpha, float beta, float l_0, float n, float T) {
    float d_A = calc_d_A(d_AA, d_AB, d_BB, alpha, beta, l_0, n, T);
    float d_A_n = powf(d_A, n);

    float O_A = tanhf(l_0 * d_A_n);


    float dt_tilde_alpha = (l_0 * powf(1.0 - O_A, 2.0*n-1.0) * (1.0 - powf(O_A, 2.0) )
                                + 4.0 * powf(1.0 - tanhf(d_A_n), 2.0*n - 1.0) * (1.0 - powf( tanhf(d_A_n), 2.0) )) * powf(d_A, n-1.0);

    return dt_tilde_alpha;
}

__device__ float calc_dt_tilde_beta(float d_AA, float d_AB, float d_BB, float alpha, float beta, float l_0, float n, float T) {
    float d_B = calc_d_B(d_AA, d_AB, d_BB, alpha, beta, l_0, n, T);
    float d_B_n = powf(d_B, n);

    float O_B = tanhf(l_0 * d_B_n);


    float dt_tilde_beta = (-l_0 * powf(1.0 + O_B, 2.0*n-1) * (1.0 - powf(O_B, 2.0) )
                            + 4 * powf(1.0 - tanhf(d_B_n), 2.0*n - 1.0) * (1.0 - powf( tanhf(d_B_n), 2.0) )) * powf(d_B, n-1);

    return dt_tilde_beta;
}

__device__ float calc_dt_tilde_l_0(float d_AA, float d_AB, float d_BB, float alpha, float beta, float l_0, float n, float T) {
    float d_A = calc_d_A(d_AA, d_AB, d_BB, alpha, beta, l_0, n, T);
    float d_B = calc_d_B(d_AA, d_AB, d_BB, alpha, beta, l_0, n, T);

    float d_A_n = powf(d_A, n);
    float d_B_n = powf(d_B, n);

    float O_A = tanhf(l_0 * d_A_n);
    float O_B = tanhf(l_0 * d_B_n);


    float dt_tilde_l_0 = powf(1.0 - O_A, 2.0*n-1.0) * (1.0 - powf(O_A, 2.0) ) * powf(d_A, n) - powf(1.0 + O_B, 2.0*n-1.0) * (1.0 - powf(O_B, 2.0) ) * powf(d_B, n);


    return dt_tilde_l_0;
}

//float* alpha_boundary_condition, float* alpha_condition, float* l_0_condition, float d_AA, float d_AB, float d_BB, float alpha, float beta, float l_0, float n, float T
__global__ void calc_conditions(int* alpha_boundary_condition, int* alpha_condition, int* l_0_condition, int size, float d_AA, float d_AB, float d_BB, float n, float T) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    int i = index%size;
    int j = (int) floor(index*1.0/size);

    float alpha, beta, l_0;

    if(index < size*size) {

        alpha = -1.0 + i*(2.0/size);
        l_0 = -1.0 + j*(2.0/size);

        beta = 1.0 - fabsf(alpha); // Negative beta is dealt in other ways now.. (swapping d_AA and d_BB)

        float dt_tilde_alpha, dt_tilde_beta, dt_tilde_l_0;

        dt_tilde_alpha = calc_dt_tilde_alpha(d_AA, d_AB, d_BB, alpha, beta, l_0, n, T);
        dt_tilde_beta = calc_dt_tilde_beta(d_AA, d_AB, d_BB, alpha, beta, l_0, n, T);
        dt_tilde_l_0 = calc_dt_tilde_l_0(d_AA, d_AB, d_BB, alpha, beta, l_0, n, T);

        float dt_tilde_alpha_over_alpha = dt_tilde_alpha/alpha;
        float dt_tilde_beta_over_beta = dt_tilde_beta/beta;

        alpha_boundary_condition[index] = sign( sign(alpha)*dt_tilde_alpha_over_alpha + sign(beta)*dt_tilde_beta_over_beta );
        alpha_boundary_condition[index] = alpha_boundary_condition[index] > 0;

        if(!isfinite(dt_tilde_alpha_over_alpha - dt_tilde_beta_over_beta) ||  alpha + beta < 0) {
            alpha_boundary_condition[index] = 0;
        }


        alpha_condition[index] = sign(dt_tilde_alpha_over_alpha - dt_tilde_beta_over_beta);
        l_0_condition[index] = sign(dt_tilde_l_0); //for now only the non saturation solutions.

    }
}

__global__ void calc_solutions(int* isSolution, int* alpha_boundary_condition, int* alpha_condition, int* l_0_condition, int size) {

    int index = threadIdx.x + blockIdx.x * blockDim.x;

    int i = index%size;
    int j = (int) floor(index*1.0/size);

    if(index < size*size) {
        int index_L = (i-1) + j*size;
        int index_R = (i+1) + j*size;

        int index_U = (i) + (j-1)*size;
        int index_D = (i) + (j+1)*size;


        if(index_L < 0 || index_R < 0 || index_U < 0 || index_D < 0) {
            isSolution[index] = 0;
            return;
        }

        if(index_L >= size*size || index_R >= size*size || index_U >= size*size || index_D >= size*size) {
            isSolution[index] = 0;
            return;
        }


        int alpha_crit = ((alpha_condition[index_L] == -alpha_condition[index_R])) || ((alpha_condition[index_U] == -alpha_condition[index_D])); //<--- This thing includes a jump discontinuity at 0!

        int l_0_crit_PN_not =  ((l_0_condition[index_U] + l_0_condition[index_D])!=0) * ((l_0_condition[index_L] + l_0_condition[index_R])!=0); //( (l_0_condition[index_L] + l_0_condition[index_R])!=0)*


        // Why j==1? Because we're gonna use the the j-1 index for the alpha_crit and l_0 saturation... Remember here we want dt_l_0 to be negative
        int l_0_crit_sat_n_not =  1 - (j == 1) * (l_0_condition[index] == -1);

        // Why j==size-2? Because we're gonna use the the j-1 index for the alpha_crit and l_0 saturation...
        int l_0_crit_sat_p_not =  1 - (j == (size-2)) * l_0_condition[index];

        // Allow for PN crit OR n_sat crit or p_sat crit.
        int l_0_crit = 1 - l_0_crit_PN_not*l_0_crit_sat_n_not*l_0_crit_sat_p_not;



        isSolution[index] = l_0_crit * alpha_crit * alpha_boundary_condition[index]; // * l_0_crit;// * l_0_crit; //* alpha_boundary_condition[index] * l_0_crit; //* l_0_crit //alpha_crit * alpha_boundary_condition[index] * l_0_crit

    }

}

void checkGpuMem() {
    size_t free_t,total_t;
    cudaMemGetInfo(&free_t,&total_t);

    printf ( "  mem free %lf Bytes.\n mem total %lf Bytes.\n \n \n",(float) free_t, (float) total_t);
}

extern "C" __declspec(dllexport) void CUDA__findSolutions(int* cpu_isSolution, int size, float d_AA, float d_AB, float d_BB, float n, float T) {

    // GPU arrays
    // int -> char
    int* alpha_boundary_condition;
    int* alpha_condition;
    int* l_0_condition;
    int* isSolution;


    // GPU Memory Allocation
    int dev = 0;
    cudaSetDevice(dev);


    //gpuAssert(cudaMallocAsync((void **)&alpha_boundary_condition, sizeof(int)*size*size, cudaStreamPerThread), 150);

    gpuAssert(cudaMalloc((void **)&alpha_boundary_condition, sizeof(int)*size*size), 150);
    gpuAssert(cudaMalloc((void **)&alpha_condition, sizeof(int)*size*size), 151);
    gpuAssert(cudaMalloc((void **)&l_0_condition, sizeof(int)*size*size), 152);
    gpuAssert(cudaMalloc((void **)&isSolution, sizeof(int)*size*size), 153);


    gpuAssert(cudaMemset(alpha_boundary_condition, 0, sizeof(int)*size*size), 154);
    gpuAssert(cudaMemset(alpha_condition, 0, sizeof(int)*size*size), 155);
    gpuAssert(cudaMemset(l_0_condition, 0, sizeof(int)*size*size), 156);
    gpuAssert(cudaMemset(isSolution, 0, sizeof(int)*size*size), 156);


    int NB_THREADS_PER_BLOCK = 1024;
    int NB_BLOCKS = (int) ceil(size*size*1.0/NB_THREADS_PER_BLOCK);

    // Calculate Conditions
    calc_conditions <<<NB_BLOCKS, NB_THREADS_PER_BLOCK>>> (alpha_boundary_condition, alpha_condition, l_0_condition, size, d_AA, d_AB, d_BB, n, T);
    gpuAssert(cudaDeviceSynchronize(), 165);

    // Calculate Solutions
    calc_solutions  <<<NB_BLOCKS, NB_THREADS_PER_BLOCK>>> (isSolution, alpha_boundary_condition, alpha_condition, l_0_condition, size);
    gpuAssert(cudaDeviceSynchronize(), 166);

    // Copy GPU solution array to CPU
    gpuAssert(cudaMemcpy(cpu_isSolution, isSolution, sizeof(int)*size*size, cudaMemcpyDeviceToHost), 169);

    // Free GPU arrays
    cudaFree(alpha_boundary_condition);
    cudaFree(alpha_condition);
    cudaFree(l_0_condition);
    cudaFree(isSolution);

}
