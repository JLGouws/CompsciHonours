/*
 * GPU implementation of montecarlo estimation of PI
 *
 * J L Gouws -- 19G4436
 * π ≈ 3.141438
 * Total execution time ≈ 100 ms
 * Time to set up RNG ≈ 31.5 ms
 * Coordinate generation time ≈ 11.8 ms
 * Main kernel time ≈ 10.5 ms
 * Reduction time ≈ 0.0010 ms
 *
 * This was my first attempt at writing this program.
 * I cannot think of many optimizations to do.
 * I could use multiple streams, but I am not transferring much data between host and device.
 * Using mulitple streams would also make timing difficult
 * Most of the execution time is caused by setting up the RNG states
 * Thus doing more coordinates per thread reduces the time.
 *
 * For 128 coordinates per thread:
 * Total execution time ≈ 225 ms
 *
 * Doing more elements per block also results in fewer blocks being launched.
 * This can reduce some overhead
 *
 * The kernels calculate averages on the GPU to decrease copy time from Device to host.
 * The kernels use shared memory to reduce transactions with global memory while reducing.
 * Last few items copied to CPU to average in serial, since CPU is faster
 * at doing this
 */

#include <cuda_runtime.h>

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <string.h>
#include <curand_kernel.h>
#include <sys/time.h>
#define SEED 1234 
#define ITER 100000000
#define EPTHREAD 600             //number of coordinates per thread.
#define REDEPTHREAD 16
#define TPBLOCK 128
#define WARPSIZE 32

using namespace std;

__global__ void setup_rng_states(curandState *const state)
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  curand_init(SEED, 7 * id, 0, &state[id]);
}

__global__ void generate_coordinates(curandState *state, float *x,
                                                          float *y, const int n)
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = gridDim.x * blockDim.x;
  curandState localState = state[id];
  for(; id < n; id += stride){//get numbers from uniform distribution
    x[id] = curand_uniform(&localState);
    y[id] = curand_uniform(&localState);
  }
}

__global__ void monte_carlo(float *x, float *y, float *bAverages, const int N, const int n)
{
  __shared__ float averages[TPBLOCK];
  unsigned int id = threadIdx.x + blockIdx.x * blockDim.x,
               stride = gridDim.x * blockDim.x,
               count = 0;

  float xl, yl, total = 0;

  for(; id < N; id += stride, total++){ //stride over coordinate arrays
                                        //memory coalescence
    xl = x[id];
    yl = y[id];
    count += (xl * xl + yl * yl) <=1;
  }


  averages[threadIdx.x] = (float) count / total;
  
  __syncthreads();

  if(threadIdx.x < WARPSIZE) //last warp reduces elements in shared memory.
  { 
    float smallAverage = 0;
    for(int i = threadIdx.x; i < TPBLOCK; i+= WARPSIZE)
      smallAverage += averages[i];
    smallAverage /= TPBLOCK / WARPSIZE;
    int idx = blockIdx.x * WARPSIZE + threadIdx.x;
    if (idx < n)
      bAverages[idx] = smallAverage; //reducing block down to 32 values
                                     //I don't reduce it down to one to improve
                                     //memory efficiency
  }
}

/*
 * Averages more on GPU -> does not do much good here, but helps for larger
 * number of iterations.
 * 200000000 iterations takes about 175 ms in total
 */
__global__ void reduce(float* values, const int n, float *bAverages)
{
  __shared__ float averages[TPBLOCK];
  unsigned int id = threadIdx.x + blockIdx.x * blockDim.x,
               stride = gridDim.x * blockDim.x;
  float tAve = 0, num = 0;

  for(; id < n; id += stride, num++){
    tAve += values[id];
  }

  averages[threadIdx.x] = tAve / num;

  __syncthreads();

  if(threadIdx.x < WARPSIZE) //last warp reduces elements in shared memory.
  { 
    float smallAverage = 0;
    for(int i = threadIdx.x; i < TPBLOCK; i+= WARPSIZE)
      smallAverage += averages[i];
    smallAverage /= TPBLOCK / WARPSIZE;
    bAverages[blockIdx.x * WARPSIZE + threadIdx.x] = smallAverage; //reducing block down to 32 values
                                                                   //I don't reduce it down to one to improve
                                                                   //memory efficiency

  }
}

void checkerror(cudaError err)
{ if (err != cudaSuccess) fputs(cudaGetErrorString(err), stdout);
	return;
}

int main()//int argc, char* argv)
{
  unsigned int blocksPerGrid, redBlocksPerGrid;
  float *x, *y, *averages, *redAverages, *hostPi;

#ifdef TIMEWHOLE
  struct timeval start_time_whole, stop_time_whole, elapsed_time_whole;
  gettimeofday(&start_time_whole,NULL); // Unix timer
#endif

  blocksPerGrid = ((ITER + EPTHREAD - 1) / EPTHREAD + TPBLOCK - 1) / TPBLOCK;
  curandState *devStates = 0;

  //space for state rng states
  checkerror(cudaMalloc((void **) &devStates, blocksPerGrid * TPBLOCK * sizeof(curandState)));

  checkerror(cudaMalloc((void **) &x, ITER * sizeof(float)));

  checkerror(cudaMalloc((void **) &y, ITER * sizeof(float)));

#ifdef TIMESETUP
  float setup_time;
  cudaEvent_t setup_start, setup_stop;
	cudaEventCreate(&setup_start);
	cudaEventCreate(&setup_stop);
	cudaEventRecord(setup_start);  // start timing
#endif
  setup_rng_states<<<blocksPerGrid, TPBLOCK>>>(devStates);//set up states
  checkerror(cudaGetLastError());
#ifdef TIMESETUP
  cudaEventRecord(setup_stop);
	checkerror(cudaEventSynchronize(setup_stop));
	cudaEventElapsedTime(&setup_time, setup_start, setup_stop);  // time random generation

	printf("Time to setup states: %f ms\n", setup_time);
#endif


#ifdef TIMEGENCOORDS
  float coord_time;
  cudaEvent_t coord_start, coord_stop;
	cudaEventCreate(&coord_start);
	cudaEventCreate(&coord_stop);
	cudaEventRecord(coord_start);  // start timing
#endif
  generate_coordinates<<<blocksPerGrid, TPBLOCK>>>(devStates, x, y ,ITER); //generate coordinates
  checkerror(cudaGetLastError());
#ifdef TIMEGENCOORDS
  cudaEventRecord(coord_stop);
	checkerror(cudaEventSynchronize(coord_stop));
	cudaEventElapsedTime(&coord_time, coord_start, coord_stop);  // time random generation

	printf("Time to generate coordinates: %f ms\n", coord_time);
#endif

  checkerror(cudaMalloc((void **) &averages, 
                                     WARPSIZE * blocksPerGrid * sizeof(float)));
#ifdef TIMEBULK
  float bulk_time;
  cudaEvent_t bulk_start, bulk_stop;
	cudaEventCreate(&bulk_start);
	cudaEventCreate(&bulk_stop);
	cudaEventRecord(bulk_start);  // start timing
#endif
  monte_carlo<<<blocksPerGrid, TPBLOCK>>>(x, y, averages, ITER, WARPSIZE * blocksPerGrid);
  checkerror(cudaGetLastError());
#ifdef TIMEBULK  
  cudaEventRecord(bulk_stop);
	checkerror(cudaEventSynchronize(bulk_stop));
	cudaEventElapsedTime(&bulk_time, bulk_start, bulk_stop);  // time random generation

	printf("Time to run main computation: %f ms\n", bulk_time);
#endif


  redBlocksPerGrid = ((WARPSIZE * blocksPerGrid + REDEPTHREAD - 1) / REDEPTHREAD + 
                                                  TPBLOCK - 1) / TPBLOCK;

  checkerror(cudaMalloc((void **) &redAverages, WARPSIZE * 
                                             redBlocksPerGrid * sizeof(float)));
  
#ifdef TIMEREDUC
  float reduc_time;
  cudaEvent_t reduc_start, reduc_stop;
	cudaEventCreate(&reduc_start);
	cudaEventCreate(&reduc_stop);
	cudaEventRecord(reduc_start);  // start timing
#endif
  //average calculations on GPU
  reduce<<<redBlocksPerGrid, TPBLOCK>>>(averages, 
                                      WARPSIZE * blocksPerGrid,
                                      redAverages);
  checkerror(cudaGetLastError());
#ifdef TIMEREDUC 
  cudaEventRecord(reduc_stop);
	checkerror(cudaEventSynchronize(reduc_stop));
	cudaEventElapsedTime(&reduc_time, reduc_start, reduc_stop);  // time random generation

	printf("Time to reduce on GPU: %f ms\n", reduc_time);
#endif


  hostPi = new float[WARPSIZE * redBlocksPerGrid];
  checkerror(cudaMemcpy(hostPi, redAverages,
                                    WARPSIZE * redBlocksPerGrid * sizeof(float)
                                                     , cudaMemcpyDeviceToHost));
  float sum = 0;
  for(int i = 0; i < WARPSIZE * redBlocksPerGrid; i++ )
    sum += hostPi[i];
  printf("The program estimates to π be: %f\n", 4 * sum / (WARPSIZE * redBlocksPerGrid));

#ifdef TIMEWHOLE
  gettimeofday(&stop_time_whole,NULL);
  timersub(&stop_time_whole, &start_time_whole, &elapsed_time_whole); // Unix time subtract routine

  printf("Total time was %f milliseconds.\n", 1000.0 * (
               elapsed_time_whole.tv_sec+elapsed_time_whole.tv_usec/1000000.0));
#endif

  free(hostPi);
  cudaFree(devStates);
  cudaFree(x);
  cudaFree(y);
  cudaFree(averages);
  cudaFree(redAverages);

  return 0;
}
