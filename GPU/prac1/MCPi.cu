#include <cuda_runtime.h>
#include "helper_cuda.h"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <curand_kernel.h>
#define SEED 35791246
#define ITER 10000000

__global__ void monte_carlo(const int iterThread)
{
    curandState state;
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x, count = 0;
    float x, y;

    curand_init(SEED, 2 * id, 0, &state);
    for(int i = 0; i < iterThread; i++){
      x = curand_uniform(&state);
      y = curand_uniform(&state);
      count += (x * x + y * y) <=1;
    }
}

int main(int argc, char* argv)
{
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
   return 0;
}
