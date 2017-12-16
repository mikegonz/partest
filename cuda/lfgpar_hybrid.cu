#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

/*
 * The Lagged Fibonacci Generator has a state that is an array of lag2
 * elements. The state begins as initvals. To produce a new random number,
 * the function returns
 *     (state[lag2-lag1] + state[0]) mod modval
 * It then shift all the states left by 1 to add the new
 * values to the state values.
 * NOTE: lag1 should be less than lag2
 *       modval should be a power of 2 minus 1
 *       state should be an array with lag2 elements
 */

//int iters = 100000;
int nthreads = 40;
int iters = 100000;
int nprocs;
int rank;
int TAG_J = 111;
int TAG_K = 222;
int TAG_RESULT = 555;
//uint32_t modval = 4294967295;
#define MODVAL 0xFFFFFFFF
#define LAG1 9739
#define LAG2 23209
//uint32_t lag1 = 9739; //5;
//uint32_t lag2 = 23209; //17;
uint32_t * statearray;
uint32_t * d_arr;
uint32_t * d_newarr;

__global__ void lfgkernel(uint32_t * arr, uint32_t * newarr){
    int offset = blockIdx.x * blockDim.x + threadIdx.x;
    newarr[offset] = (arr[(LAG2 - LAG1) + offset] + arr[offset]) & MODVAL;
}

int main(int argc, char * argv[]){
    srand(0);
    statearray = (uint32_t *)malloc(LAG2 * sizeof(uint32_t));
    for(int i = 0; i < LAG2; i++) statearray[i] = rand(); //this should suck maybe do something about this
    cudaMalloc((void**)&d_arr, LAG2*sizeof(uint32_t));
    cudaMalloc((void**)&d_newarr, LAG1*sizeof(uint32_t));
    cudaMemcpy(d_arr, statearray, LAG2*sizeof(uint32_t), cudaMemcpyHostToDevice);
    dim3 numBlocks(ceil(LAG1 / 1024.0));
    dim3 threadsPerBlock(ceil(LAG1 / ((float) numBlocks.x)));
    while(iters-- > 0){
      uint32_t * temparray = (uint32_t *)malloc(LAG2 * sizeof(uint32_t));
      lfgkernel<<<numBlocks, threadsPerBlock>>>(d_arr, d_newarr);
#pragma omp parallel for shared(statearray)
      for(int i = 0; i < (LAG2 - LAG1); i++)
	temparray[i] = statearray[i + LAG1];
      cudaDeviceSynchronize();
      cudaMemcpy(temparray + (LAG2 - LAG1), d_newarr, LAG1*sizeof(uint32_t), cudaMemcpyDeviceToHost);
      uint32_t * trash = statearray;
      statearray = temparray;
      free(trash);
      //for(int i = (LAG2 - LAG1); i < LAG2; i++) printf("%u\n", statearray[i]);
    }
    cudaFree(d_arr);
    cudaFree(d_newarr);
    return 0;
}
