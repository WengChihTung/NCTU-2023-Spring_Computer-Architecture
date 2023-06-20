#include "parameters.h"


__global__ void cuda_kernel(int *B,int *A,IndexSave *dInd)
{	
	// complete cuda kernel function	
};


float GPU_kernel(int *B,int *A,IndexSave* indsave){

	int *dA,*dB;
	IndexSave* dInd;

	// Creat Timing Event
  	cudaEvent_t start, stop;
	cudaEventCreate (&start);
	cudaEventCreate (&stop); 	

	// Allocate Memory Space on Device

	// Allocate Memory Space on Device (for observation)
	cudaMalloc((void**)&dInd,sizeof(IndexSave)*SIZE);

	// Copy Data to be Calculated

	// Copy Data (indsave array) to device
	cudaMemcpy(dInd, indsave, sizeof(IndexSave)*SIZE, cudaMemcpyHostToDevice);
	
	// Start Timer
	cudaEventRecord(start, 0);

	// Lunch Kernel
	dim3 dimGrid(2);
	dim3 dimBlock(4);
	cuda_kernel<<<dimGrid,dimBlock>>>(dB,dA,dInd);

	// Stop Timer
	cudaEventRecord(stop, 0);
  	cudaEventSynchronize(stop); 

	// Copy Output back

	// Release Memory Space on Device
	cudaFree(dA);
	cudaFree(dB);
	cudaFree(dInd);

	// Calculate Elapsed Time
  	float elapsedTime; 
  	cudaEventElapsedTime(&elapsedTime, start, stop);  

	return elapsedTime;
}
