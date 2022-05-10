/**
 * J L Gouws -- 19G4436 Sumarray modified
 */
// C++ version of sumarray from GPU course slides
// Compile: nvcc sumArraycpp.cu -o sumarraycpp
// Run: sumarraycpp

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
// #include <helper_cuda.h>

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

void compareArrayOnHost(float *A, float *B, char* error, const int n)
{
    float *ia = A;
    float *ib = B;

    for (int ix = 0; ix < n; ix++)
      *error |= ia[ix] != ib[ix];
    return;
}


__global__ void sumArrayOnGPU(float *A, float *B, float *C, int N)
{

//******************************************************************************
    //get index of element
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x; 
//______________________________________________________________________________
 
    if (idx < N)
    {
        C[idx] = A[idx] + B[idx];
//        C[idx] += 10 * (idx == 10);
    }
}

__global__ void compareArrayOnGPU(float *A, float *B, char* error, int N)
{
//******************************************************************************
    //get index of element
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x; 
//______________________________________________________________________________
    if (idx < N && A[idx] != B[idx])
      *error = 1;
}

//******************************************************************************
#define nelem 1000 //changed this to 1000 elements
//______________________________________________________________________________

void checkerror(cudaError err)
{ if (err != cudaSuccess) cout << cudaGetErrorString(err) << endl;
	return; 
}


int main(int argc, char **argv)
{
    int nBytes = nelem * sizeof(float);

//******************************************************************************
    char herror = 0;
    int threadsPerBlock = 256, //more robust thread and grid setup.
        blocksPerGrid = (nelem + threadsPerBlock - 1) / threadsPerBlock;
//______________________________________________________________________________

    // malloc host memory
    float *h_A, *h_B, *hostC, *gpuC;
    h_A = new float[nelem];
    h_B = new float[nelem];
    hostC = new float[nelem];
    gpuC = new float[nelem];

    // initialise A and B
     for (int i=0; i < nelem; i++)
      { 
        h_A[i] = i;
        h_B[i] = i; 
      }

     // add Array at host side for result checks
    sumArrayOnHost (h_A, h_B, hostC, nelem);
    cout << "Host sum is: " << hostC[9] << endl;

	// malloc device global memory
    float *d_A , *d_B, *d_C, *dCPU_C;
    char *derror;
    checkerror(cudaMalloc((void **)&d_A , nBytes)); 
    checkerror(cudaMalloc((void **)&d_B, nBytes)); 
    checkerror(cudaMalloc((void **)&d_C, nBytes)); 
//******************************************************************************
    checkerror(cudaMalloc((void **)&dCPU_C, nBytes)); 
    checkerror(cudaMalloc((void **)&derror, sizeof(char))); 
//______________________________________________________________________________

     // transfer data from host to device
    checkerror(cudaMemcpy(d_A , h_A, nBytes, cudaMemcpyHostToDevice));
    checkerror(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));
//******************************************************************************
    checkerror(cudaMemcpy(&derror, &herror, sizeof(char), cudaMemcpyHostToDevice));
//______________________________________________________________________________

	// execute the kernel
    checkerror(cudaDeviceSynchronize());
    sumArrayOnGPU<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, nelem);
    checkerror(cudaGetLastError());

 
    // copy kernel result back to host side
    checkerror(cudaMemcpy(gpuC, d_C, nBytes, cudaMemcpyDeviceToHost));
  	cout << "GPU sum is: " << gpuC[9] << endl;
    
//******************************************************************************
    compareArrayOnHost (gpuC, hostC, &herror, nelem);
    cout << (herror ? "CPU found at least one error" : 
                      "CPU did not find any errors") << endl;

    //copy the host solution to device
    checkerror(cudaMemcpy(dCPU_C, hostC, nBytes, cudaMemcpyHostToDevice));

    compareArrayOnGPU<<<blocksPerGrid, threadsPerBlock>>>(d_C, dCPU_C, derror, nelem);
    checkerror(cudaMemcpy(&herror, derror, sizeof(char), cudaMemcpyDeviceToHost));
    cout << ( herror ? "GPU found at least one error" : 
                       "GPU did not find any errors" )<< endl;
//______________________________________________________________________________
	
     // free device global memory
    checkerror(cudaFree(d_A ));
    checkerror(cudaFree(d_B));
    checkerror(cudaFree(d_C));
//******************************************************************************
    checkerror(cudaFree(dCPU_C));
    checkerror(cudaFree(derror));
//______________________________________________________________________________

    // free host memory
    free(h_A);
    free(h_B);
    free(hostC);
    free(gpuC);

    // reset device
    checkerror(cudaDeviceReset());

    return 0;
}
