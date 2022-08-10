/*************************************************
 * Laplace Serial C Version
 *
 * Temperature is initially 0.0
 * Boundaries are as follows:
 *
 *      0         T         0
 *   0  +-------------------+  0
 *      |                   |
 *      |                   |
 *      |                   |
 *   T  |                   |  T
 *      |                   |
 *      |                   |
 *      |                   |
 *   0  +-------------------+ 100
 *      0         T        100
 *
 *  Copyright John Urbanic, PSC 2017
 *
 ************************************************/
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include "device_launch_parameters.h"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>

#define BLOCKX     32
#define BLOCKY     16

// size of plate
#define COLUMNS    5
#define ROWS       33

#ifndef MAX_ITER
#define MAX_ITER 100
#endif

// largest permitted change in temp (This value takes about 3400 steps)
#define MAX_TEMP_ERROR 0.01

__global__ void init_grid(double **grid, unsigned int Nx, unsigned int Ny){
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x,
               idy = blockIdx.y * blockDim.y + threadIdx.y;
  if(idx < Nx + 2 && idy < Ny + 2)
    grid[idy][idx] = 0.0 + (idx == Nx + 1) * 100.0 * idy / Ny + 
                                           (idy == Ny + 1) * 100.0 * idx / Nx;
}

__global__ void evolve_grid(double **grid_in, double **grid_out, 
                                            unsigned int Nx, unsigned int Ny) {

  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x + 1, //offsets of boundaries
               idy = blockIdx.y * blockDim.y + threadIdx.y + 1;

  __shared__ double tile[BLOCKY + 2][BLOCKX + 2];

  if (threadIdx.y < 2)
    tile[threadIdx.y][threadIdx.x + 1] = 
                                            grid_in[idy - 1][idx];
  if (threadIdx.x == 0 && idy <= Ny)
    tile[threadIdx.y + 1][threadIdx.x] = grid_in[idy][idx - 1];

  if (threadIdx.x == BLOCKX - 1 && idy <= Ny)
    tile[threadIdx.y + 1][threadIdx.x + 2] = grid_in[idy][idx + 1];

  if(idx <= Nx + 1 && idy <= Ny) {
      tile[threadIdx.y + 2][threadIdx.x + 1] 
                                              = grid_in[idy + 1][idx];
  }

  __syncthreads();

  if (idx <= Nx && idy <= Ny)
    grid_out[idy][idx] 
                    = 0.25 * (tile[threadIdx.y + 1][threadIdx.x + 2]
                              +  tile[threadIdx.y + 1][threadIdx.x] 
                              +  tile[threadIdx.y][threadIdx.x + 1]
                              +  tile[threadIdx.y + 2][threadIdx.x + 1]);
}

__global__ void evaluate_grid(double **grid_last, double **grid_new, int *error, 
                              unsigned int Nx, unsigned int Ny) {

  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x + 1, //offsets of boundaries
               idy = blockIdx.y * blockDim.y + threadIdx.y + 1;

  if (idx <= Nx && idy <= Ny && fabsf(grid_last[idy][idx] - grid_new[idy][idx]) > MAX_TEMP_ERROR){
    *error = 1;
  }
}

double Temperature[ROWS+2][COLUMNS+2];      // temperature grid
double Temperature_last[ROWS+2][COLUMNS+2]; // temperature grid from last iteration

//   helper routines
void initialize();
void print_grid_last();
void track_progress(int iter);
void checkerror(cudaError err);


int main(int argc, char *argv[]) {

    int max_iterations;                                  // number of iterations
    int iteration=1;                                     // current iteration
    int cont = 1, *d_cont, cont_tmp = 0;
    //double dt=100;                                       // largest change in t
    double **d_grid_last, **d_grid_new, 
            *h_grid_last[ROWS + 2], *h_grid_new[ROWS + 2], **d_grid_temp;
    //struct timeval start_time, stop_time, elapsed_time;  // timers

    max_iterations = MAX_ITER;

    checkerror(cudaMalloc((void **) &d_grid_last, (ROWS + 2) * sizeof(double *)));   
    checkerror(cudaMalloc((void **) &d_grid_new, (ROWS + 2) * sizeof(double *)));   
    checkerror(cudaMalloc((void **) &d_cont, sizeof(int)));   

    for(int i = 0; i < ROWS + 2; i++) {
      checkerror(cudaMalloc((void **) &(i[h_grid_last]), (COLUMNS + 2) * sizeof(double)));   
      checkerror(cudaMalloc((void **) &(i[h_grid_new]), (COLUMNS + 2) * sizeof(double)));   
    }

    cudaMemcpy(d_grid_last, h_grid_last, (ROWS + 2) * sizeof(double *), 
                                                        cudaMemcpyHostToDevice);
    cudaMemcpy(d_grid_new, h_grid_new, (ROWS + 2) * sizeof(double *), 
                                                        cudaMemcpyHostToDevice);
    cudaMemcpy(d_cont, &cont_tmp, sizeof(int), cudaMemcpyHostToDevice);

    dim3 l_block(BLOCKX, BLOCKY);
    dim3 l_grid((COLUMNS + l_block.x + 1) / l_block.x, 
                                     (ROWS + l_block.y + 1) / l_block.y);
    init_grid<<<l_grid, l_block>>>(d_grid_last, COLUMNS, ROWS);
    init_grid<<<l_grid, l_block>>>(d_grid_new, COLUMNS, ROWS);
    checkerror(cudaGetLastError());

    l_grid = dim3((COLUMNS + l_block.x - 1) / l_block.x, 
                                     (ROWS + l_block.y - 1) / l_block.y);

    while (iteration <= max_iterations && cont == 1) {
      evolve_grid<<<l_grid, l_block>>>(d_grid_last, d_grid_new, COLUMNS, ROWS);
      checkerror(cudaGetLastError());

      evaluate_grid<<<l_grid, l_block>>>(d_grid_last, d_grid_new, d_cont, COLUMNS, ROWS);
      checkerror(cudaGetLastError());

      checkerror(cudaMemcpy(&cont, d_cont, sizeof(int), cudaMemcpyDeviceToHost));
      checkerror(cudaMemcpy(d_cont, &cont_tmp, sizeof(int), cudaMemcpyHostToDevice));

      d_grid_temp = d_grid_last; //switch grids
      d_grid_last = d_grid_new;
      d_grid_new = d_grid_temp;

      iteration++;
    }

    for(int i = 0; i < ROWS + 2; i++) {
      cudaMemcpy(Temperature_last[i], h_grid_new[i], 
                        (COLUMNS + 2) * sizeof(double), cudaMemcpyDeviceToHost);
    }

    print_grid_last();

    for(int i = 0; i < ROWS + 2; i++) {
      cudaFree(i[h_grid_last]);
      cudaFree(i[h_grid_new]);
    }
    cudaFree(d_grid_last);
    cudaFree(d_grid_new);
    cudaFree(d_cont);
}


// initialize plate and boundary conditions
// Temp_last is used to to start first iteration
void initialize(){

    int i,j;

    for(i = 0; i <= ROWS+1; i++){
        for (j = 0; j <= COLUMNS+1; j++){
            Temperature_last[i][j] = 0.0;
        }
    }

    // these boundary conditions never change throughout run

    // set left side to 0 and right to a linear increase
    for(i = 0; i <= ROWS+1; i++) {
        Temperature_last[i][0] = 0.0;
        Temperature_last[i][COLUMNS+1] = (100.0/ROWS)*i;
    }
    
    // set top to 0 and bottom to linear increase
    for(j = 0; j <= COLUMNS+1; j++) {
        Temperature_last[0][j] = 0.0;
        Temperature_last[ROWS+1][j] = (100.0/COLUMNS)*j;
    }

}


// print diagonal in bottom right corner where most action is
void track_progress(int iteration) {

    int i;

    printf("---------- Iteration number: %d ------------\n", iteration);
    for(i = ROWS-5; i <= ROWS; i++) {
        printf("[%d,%d]: %5.2f  ", i, i, Temperature[i][i]);
    }
    printf("\n");
}

void print_grid_last() {
    for(int i = 0; i <= ROWS+1; i++){
        printf("%03d ", i);
        for (int j = 0; j <= COLUMNS+1; j++){
            printf("%5f ", Temperature_last[i][j]);
        }
        fputs("\n", stdout);
    }
}

void checkerror(cudaError err)
{ if (err != cudaSuccess) fputs(cudaGetErrorString(err), stdout);
	return;
}
