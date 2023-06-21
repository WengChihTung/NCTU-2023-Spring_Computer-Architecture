#include <stdio.h>

#include "parameters.h"

extern float GPU_kernel(int B[][SIZE],int A[][SIZE],IndexSave indsave[][SIZE]);

void genNumbers(int A[][SIZE], int size){
	for(int i = 0; i < size; i++){
		for(int j = 0; j < size; j++){
			A[i][j] = rand()%256;
		}
	}
}

void function_1(int C[][SIZE],int A[][SIZE]){
	for(int i=0;i<SIZE;i++){
		for(int j=0;j<SIZE;j++){
			C[i][j]=A[i][j];
			C[i][j]+=A[i][j];
		}
	}
}

bool verify(int A[][SIZE],int B[][SIZE]){

	for(int i=0;i<SIZE;i++){
		for(int j = 0; j < SIZE; j++){
			if(A[i][j]!=B[i][j]) return true;
		}
	}
	return false;
}

void printIndex(IndexSave indsave[][SIZE],int B[][SIZE],int C[][SIZE])
{
	for(int i=0;i<SIZE;i++){
		for(int j = 0; j < SIZE; j++){
			printf("%d, %d : blockInd_x=%d,threadInd_x=%d,head=%d,stripe=%d",i,j,(indsave[i][j]).blockInd_x,(indsave[i][j]).threadInd_x,(indsave[i][j]).head,(indsave[i][j]).stripe);
			printf(" || GPU result=%d,CPU result=%d\n",B[i][j],C[i][j]);
		}
	}
}

int main()
{
	// random seed
	int A[SIZE][SIZE];
	// random number sequence computed by GPU
	int B[SIZE][SIZE];
	// random number sequence computed by CPU
	int C[SIZE][SIZE];
	// Indices saver (for checking correctness)
	IndexSave indsave[SIZE][SIZE];
	
	genNumbers(A,SIZE);
	memset( B, 0, sizeof(int)*SIZE*SIZE );

	/* CPU side*/
	function_1(C,A);

	/* GPU side*/
	float elapsedTime = GPU_kernel(B,A,indsave);

	/*Show threads execution info*/
	printIndex(indsave,B,C);

	printf("==============================================\n");
	/* verify the result*/
	if(verify(B,C)){printf("wrong answer\n");}
	printf("GPU time = %5.2f ms\n", elapsedTime);

	/*Please press any key to exit the program*/
	getchar();

}
