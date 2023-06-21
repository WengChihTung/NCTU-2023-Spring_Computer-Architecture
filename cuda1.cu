#include "parameters.h"


__global__ void cuda_kernel(int **dB, int **dA, IndexSave **dInd)
{	
	// int idx = blockIdx.x * blockDim.x + threadIdx.x;
	// if(idx < SIZE * SIZE){
	// 	int m = idx / SIZE;
	// 	int n = idx % SIZE;
	//  	dInd[m][n].blockInd_x = blockIdx.x;
	//  	dInd[m][n].threadInd_x = threadIdx.x;
	//  	dInd[m][n].head = idx;
	//  	dInd[m][n].stripe = 1;
	//  	dB[m][n] += dA[m][n];
	//  	dB[m][n] += dA[m][n];

	// }
	int i = 0;
	int TotalThread = blockDim.x * gridDim.x;
	int      stripe = SIZE * SIZE / TotalThread;
	int        head = (blockIdx.x * blockDim.x + threadIdx.x) * stripe;
	int     LoopLim = head + stripe;
	
	for(i = head; i < LoopLim; i++){
		int m = i / SIZE;
		int n = i % SIZE;
		dInd[m][n].blockInd_x = blockIdx.x;
		dInd[m][n].threadInd_x = threadIdx.x;
		dInd[m][n].head = head;
		dInd[m][n].stripe = stripe;
		dB[m][n] += dA[m][n];
		dB[m][n] += dA[m][n];
	}
};


float GPU_kernel(int **B, int **A, IndexSave **indsave){

	int** dA; 
	int** dB;
	IndexSave** dInd;

	// Creat Timing Event
  	cudaEvent_t start, stop;
	cudaEventCreate (&start);
	cudaEventCreate (&stop); 	

	// Allocate Memory Space on Device
	cudaMalloc((void**)&dA, sizeof(int*) * SIZE);
	cudaMalloc((void**)&dB, sizeof(int*) * SIZE);

	// Allocate Memory Space on Device (for observation)
	cudaMalloc((void**)&dInd, sizeof(IndexSave*) * SIZE);
	
	for(int i = 0; i < SIZE; i++){
		cudaMalloc((void**)&dA[i], sizeof(int) * SIZE);
		cudaMalloc((void**)&dB[i], sizeof(int) * SIZE);
		cudaMalloc((void**)&dInd[i], sizeof(IndexSave) * SIZE);
	}

	// Copy Data to be Calculated
	for(int i = 0; i < SIZE; i++){
		cudaMemcpy(dA[i], A[i], sizeof(int)*SIZE, cudaMemcpyHostToDevice);
		cudaMemcpy(dB[i], B[i], sizeof(int)*SIZE, cudaMemcpyHostToDevice);
		// Copy Data (indsave array) to device
		cudaMemcpy(dInd[i], indsave[i], sizeof(IndexSave)*SIZE, cudaMemcpyHostToDevice);
	}

	
	// Start Timer
	cudaEventRecord(start, 0);

	// Launch Kernel
	dim3 dimGrid(4);
	dim3 dimBlock(4);
	cuda_kernel<<<dimGrid,dimBlock>>>(dB,dA,dInd);
	//cudaDeviceSynchronize();

	// Stop Timer
	cudaEventRecord(stop, 0);
  	cudaEventSynchronize(stop); 

	// Copy Output back
	for(int i = 0; i < SIZE; i++){
		cudaMemcpy(A[i], dA[i], sizeof(int)*SIZE, cudaMemcpyDeviceToHost);
		cudaMemcpy(B[i], dB[i], sizeof(int)*SIZE, cudaMemcpyDeviceToHost);
		
		cudaMemcpy(indsave[i], dInd[i], sizeof(IndexSave)*SIZE, cudaMemcpyDeviceToHost);
	}

	// Release Memory Space on Device
	for(int i = 0; i < SIZE; i++){
		cudaFree(dA[i]);
		cudaFree(dB[i]);
		cudaFree(dInd[i]);
	}
	cudaFree(dA);
	cudaFree(dB);
	cudaFree(dInd);

	// Calculate Elapsed Time
  	float elapsedTime; 
  	cudaEventElapsedTime(&elapsedTime, start, stop);  

	return elapsedTime;
}
