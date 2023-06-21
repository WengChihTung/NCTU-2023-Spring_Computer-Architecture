#include "parameters.h"


__global__ void cuda_kernel(int dB[][SIZE],int dA[][SIZE],IndexSave dInd[][SIZE])
{	
	int i=0;
	int TotalThread = blockDim.x*gridDim.x;
	int stripe = SIZE * SIZE / TotalThread;
	int head   = (blockIdx.x*blockDim.x + threadIdx.x)*stripe;
	int LoopLim = head+stripe;
	
	for(i=head ; i<LoopLim ; i++ ){
		for(int j=0; j<SIZE; j++){
			dInd[i][j].blockInd_x = blockIdx.x;
			dInd[i][j].threadInd_x = threadIdx.x;
			dInd[i][j].head = head;
			dInd[i][j].stripe = stripe;
			dB[i][j] += dA[i][j];
			dB[i][j] += dA[i][j];
		}
	}
};


float GPU_kernel(int B[][SIZE],int A[][SIZE],IndexSave indsave[][SIZE]){

	//int dA[SIZE][SIZE],dB[SIZE][SIZE];
	//IndexSave dInd[SIZE][SIZE];

	// Creat Timing Event
  	cudaEvent_t start, stop;
	cudaEventCreate (&start);
	cudaEventCreate (&stop); 	

	// Allocate Memory Space on Device

	// Allocate Memory Space on Device (for observation)
	//cudaMalloc((void**)&dInd,sizeof(IndexSave)*SIZE);

	// Copy Data to be Calculated
	//cudaMemcpy(dA, A, sizeof(int)*SIZE*SIZE, cudaMemcpyHostToDevice);
	//cudaMemcpy(dB, B, sizeof(int)*SIZE*SIZE, cudaMemcpyHostToDevice);

	// Copy Data (indsave array) to device
	//cudaMemcpy(dInd, indsave, sizeof(IndexSave)*SIZE*SIZE, cudaMemcpyHostToDevice);
	
	// Start Timer
	cudaEventRecord(start, 0);

	// Lunch Kernel
	dim3 dimGrid(2);
	dim3 dimBlock(4);
	cuda_kernel<<<dimGrid,dimBlock>>>(B,A,indsave);

	// Stop Timer
	cudaEventRecord(stop, 0);
  	cudaEventSynchronize(stop); 

	// Copy Output back
	//cudaMemcpy(A, dA, sizeof(int)*SIZE*SIZE, cudaMemcpyDeviceToHost);
	//cudaMemcpy(B, dB, sizeof(int)*SIZE*SIZE, cudaMemcpyDeviceToHost);
	
	//cudaMemcpy(indsave, dInd, sizeof(IndexSave)*SIZE*SIZE, cudaMemcpyDeviceToHost);

	// Release Memory Space on Device
	//cudaFree(dA);
	//cudaFree(dB);
	//cudaFree(dInd);

	// Calculate Elapsed Time
  	float elapsedTime; 
  	cudaEventElapsedTime(&elapsedTime, start, stop);  

	return elapsedTime;
}
