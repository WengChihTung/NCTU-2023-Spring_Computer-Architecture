#include <stdio.h>

#include "parameters.h"

extern float GPU_kernel(int **B, int **A, IndexSave **indsave);

void genNumbers(int **A, int size){
	for(int i = 0; i < size; i++){
		for(int j = 0; j < size; j++){
			A[i][j] = rand()%256;
		}
	}
}

void function_1(int **C, int **A){
	for(int i = 0; i < SIZE; i++){
		for(int j = 0; j < SIZE; j++){
			C[i][j] = A[i][j];
			C[i][j] += A[i][j];
		}
	}
}

bool verify(int **A, int **B){

	for(int i = 0; i < SIZE; i++){
		for(int j = 0; j < SIZE; j++){
			if(A[i][j]!=B[i][j]) return true;
		}
	}
	return false;
}

void printIndex(IndexSave **indsave, int **B, int **C)
{
	for(int i = 0; i < SIZE; i++){
		for(int j = 0; j < SIZE; j++){
			printf("%d, %d : blockInd_x=%d, threadInd_x=%d, head=%d, stripe=%d", i, j, (indsave[i][j]).blockInd_x, (indsave[i][j]).threadInd_x, (indsave[i][j]).head, (indsave[i][j]).stripe);
			printf(" || GPU result=%d, CPU result=%d\n",B[i][j],C[i][j]);
		}
	}
}

int main()
{
	// random seed
	int **A = new int* [SIZE];
	// random number sequence computed by GPU
	int **B = new int* [SIZE];
	// random number sequence computed by CPU
	int **C = new int* [SIZE];
	// Indices saver (for checking correctness)
	IndexSave **indsave = new IndexSave* [SIZE];

	for(int i = 0; i < SIZE; i++){
		A[i] = new int[SIZE];
		B[i] = new int[SIZE];
		C[i] = new int[SIZE];
		indsave[i] = new IndexSave[SIZE];
	}
	
	genNumbers(A,SIZE);
	for(int i = 0; i < SIZE; i++){
		memset(B[i], 0, sizeof(int) * SIZE);
	}

	/* CPU side*/
	function_1(C,A);

	/* GPU side*/
	float elapsedTime = GPU_kernel(B, A, indsave);

	/*Show threads execution info*/
	printIndex(indsave, B, C);

	printf("==============================================\n");
	/* verify the result*/
	if(verify(B, C)){printf("wrong answer\n");}
	printf("GPU time = %5.2f ms\n", elapsedTime);

	/*Please press any key to exit the program*/
	getchar();

}
