// C++ version of sumarray from GPU course slides
// Compile: nvcc -I "C:\ProgramData\NVIDIA Corporation\CUDA Samples\v10.1\common\inc" sumArraycpp.cu -o sumarraycpp
// Run: sumarraycpp

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <helper_cuda.h>

using namespace std;

/*
 * This example implements Array addition on the host and GPU.
 * sumArrayOnHost iterates over the elements, adding
 * elements from A and B together and storing the results in C. 
 * sumArrayOnGPU implements the same logic, but using CUDA threads to process each element.
 */


void sumArrayOnHost(float *A, float *B, float *C, const int n)
{
    float *ia = A;
    float *ib = B;
    float *ic = C;

     for (int ix = 0; ix < n; ix++)
        {
            ic[ix] = ia[ix] + ib[ix];
        }
    return;
}


__global__ void sumArrayOnGPU(float *A, float *B, float *C, int N)
{
    unsigned int idx = threadIdx.x;
 
    if (idx < N)
    {
        C[idx] = A[idx] + B[idx];
    }
}

#define nelem 5000

void checkerror(cudaError err)
{ if (err != cudaSuccess) cout << cudaGetErrorString(err) << endl;
	return; 
}


int main(int argc, char **argv)
{
    int nBytes = nelem * sizeof(float);

    // malloc host memory
    float *h_A, *h_B, *hostC, *gpuC;
    h_A = new float[nelem];
    h_B = new float[nelem];
    hostC = new float[nelem];
    gpuC = new float[nelem];

    // initialise A and B
     for (int i=0; i < nelem; i++)
      { h_A[i] = i;
        h_B[i] = i; }

    memset(hostC, 0, nBytes);
    memset(gpuC, 0, nBytes);

    // add Array at host side for result checks
    sumArrayOnHost (h_A, h_B, hostC, nelem);
   cout << "Host sum is: " << hostC[9] << endl;

	// malloc device global memory
    float *d_A , *d_B, *d_C;
    checkerror(cudaMalloc((void **)&d_A , nBytes)); 
    checkerror(cudaMalloc((void **)&d_B, nBytes)); 
    checkerror(cudaMalloc((void **)&d_C, nBytes)); 

     // transfer data from host to device
    checkerror(cudaMemcpy(d_A , h_A, nBytes, cudaMemcpyHostToDevice));
    checkerror(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));

	// execute the kernel
    checkerror(cudaDeviceSynchronize());
    sumArrayOnGPU<<<10,nelem/10>>>(d_A, d_B, d_C, nelem);
    checkerror(cudaGetLastError());

 
    // copy kernel result back to host side
    checkerror(cudaMemcpy(gpuC, d_C, nBytes, cudaMemcpyDeviceToHost));
	cout << "GPU sum is: " << gpuC[9] << endl;
	
     // free device global memory
    checkerror(cudaFree(d_A ));
    checkerror(cudaFree(d_B));
    checkerror(cudaFree(d_C));

    // free host memory
    free(h_A);
    free(h_B);
    free(hostC);
    free(gpuC);

    // reset device
    checkerror(cudaDeviceReset());

    return 0;
}
