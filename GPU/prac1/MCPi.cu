#include <cuda_runtime.h>
#include "helper_cuda.h"
#include "device_launch_parameters.h"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <string.h>
#include <curand_kernel.h>
#define SEED 1234 
#define ITER 10000
#define EPBLOCK 16

using namespace std;

__global__ void setup_rng_states(curandState *state, const int n)
{
  unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int stride = gridDim.x * blockDim.x;
  for(; id < n; id += stride) curand_init(SEED, id * 7, 0, &state[id]);
}

__global__ void generate_coordinates(curandState *state, float *x,
                                                          float *y, const int n)
{
  unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int stride = gridDim.x * blockDim.x;
  curandState localState = state[id];
  for(; id < n; id += stride){
    x[id] = curand_uniform(&localState);
    y[id] = curand_uniform(&localState);
  }
}

__global__ void monte_carlo(float *x, 
                                                          float *y, const int n)
{
  unsigned int id = threadIdx.x + blockIdx.x * blockDim.x,
               stride = gridDim.x * blockDim.x,
               count = 0;

  float xl, yl;

  printf("hi");
  for(; id < n; id += stride){
    xl = x[id];
    yl = y[id];
    printf("(%.f, %.f)", xl, yl);
    count += (xl * xl + yl * yl) <=1;
  }
}

void checkerror(cudaError err)
{ if (err != cudaSuccess) cout << cudaGetErrorString(err) << endl;
	return;
}

int main()//int argc, char* argv)
{
  unsigned int threadsPerBlock = 256, BlocksPerGrid;
  float *x, *y;

  BlocksPerGrid = (ITER + threadsPerBlock - 1) / threadsPerBlock / EPBLOCK;
  curandState *devStates;

  //space for state rng states
  cudaMalloc((void **) &devStates, threadsPerBlock * BlocksPerGrid 
                                                        * sizeof(curandState));

  cudaMalloc((void **) &x, ITER * sizeof(float));

  cudaMalloc((void **) &y, ITER * sizeof(float));


  cout << BlocksPerGrid << endl;
  cout << threadsPerBlock << endl;
  setup_rng_states<<<BlocksPerGrid, threadsPerBlock>>>(devStates, ITER);
  checkerror(cudaGetLastError());


  generate_coordinates<<<BlocksPerGrid, threadsPerBlock>>>(devStates, 
                                                                    x, y ,ITER);
  checkerror(cudaGetLastError());

  monte_carlo<<<BlocksPerGrid, threadsPerBlock>>>(x, y ,ITER);
  checkerror(cudaGetLastError());
  /*
   int niter=0;
   double x,y;
   int i,count=0; // # of points in the 1st quadrant of unit circle 
   double z;
   double pi;

   printf("Enter the number of iterations used to estimate pi: ");
   scanf("%d",&niter);

   // initialize random numbers
   srand(SEED);
   count=0;
   for ( i=0; i<niter; i++) {
      x = (double)rand()/RAND_MAX;
      y = (double)rand()/RAND_MAX;
      z = x*x+y*y;
      if (z<=1) count++;
      }
   pi=(double)count/niter*4;
   printf("# of trials= %d , estimate of pi is %g \n",niter,pi);
   */
   cudaFree(devStates);
   cudaFree(x);
   cudaFree(y);
   return 0;
}
