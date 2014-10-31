#ifndef PROJ_HELPER_FUNS
#define PROJ_HELPER_FUNS

//#include <vector>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include "Constants.h"

#include <cuda_runtime.h>

using namespace std;

void reportMemoryUsage();
void reportMemoryUsageInit();

struct PrivGlobs {
	
	int numX;
	int numY;
	int numT;
	int outer;
	int myXindex;
	int myYindex;
	
	// grid
	REAL *myX; // [numX]
	REAL *myY; // [numY]
	REAL *myTimeline; // [numT]
	
	// variable
	REAL *myResult; // [outer][numX][numY]
	REAL *myResult_trans; // [outer][numY][numX]
	
	// coeffs
	REAL *myVarX; // [outer][numX][numY]
	REAL *myVarY; // [outer][numX][numY]
	
	// operators
	REAL *myDxx; // [numX][4]
	REAL *myDyy; // [numY][4]
	
	// vectors for rollback()
	REAL *u; // [outer][y][max(numX,numY)]
	REAL *v; // [outer][x][max(numX,numY)]
	REAL *a; // [outer][y][max(numX,numY)]
	REAL *b; // [outer][y][max(numX,numY)]
	REAL *c; // [outer][y][max(numX,numY)]
	REAL *y; // [outer][y][max(numX,numY)]
	REAL *yy; // [outer][y][max(numX,numY)]
	REAL *y_trans; // [outer][y][max(numX,numY)]
	
	// host pointer to struct containing device pointers
	struct PrivGlobs *device;
	// device pointer to struct containing device pointers
	struct PrivGlobs *d_globs;
	
	void init(
			const unsigned int &numX,
			const unsigned int &numY,
			const unsigned int &numT,
			const unsigned int &outer) {
		this->numX = numX;
		this->numY = numY;
		this->numT = numT;
		this->outer = outer;

		this->device = (struct PrivGlobs *) malloc(sizeof(struct PrivGlobs));
		this->device->numX = numX;
		this->device->numY = numY;
		this->device->numT = numT;
		this->device->outer = outer;
		
		size_t size_estimate = sizeof(REAL) * ((size_t) numX + numY + numX*4 + numY*4 + numT + numX*numY + numX*numY + numX*numY + outer*numY*numX + outer*numX*numY + outer*numY*numY + outer*numY*numY + outer*numY*numY + outer*numY*numY + outer*numY*numY) + sizeof(PrivGlobs);
		printf("Trying to malloc ~%dMiB on device\n", size_estimate/1024/1024);
		
		// hack: force init so memory use report works
		float *dummy;
		cudaMalloc(&dummy, 0);
		reportMemoryUsageInit();
		
		cudaMalloc(&this->device->myX, sizeof(REAL) * numX);
		cudaMalloc(&this->device->myY, sizeof(REAL) * numY);
		cudaMalloc(&this->device->myDxx, sizeof(REAL) * numX * 4);
		cudaMalloc(&this->device->myDyy, sizeof(REAL) * numY * 4);
		cudaMalloc(&this->device->myTimeline, sizeof(REAL) * numT);
		cudaMalloc(&this->device->myResult, sizeof(REAL) * outer * numX * numY);
		cudaMalloc(&this->device->myResult_trans, sizeof(REAL) * outer * numX * numY);
		cudaMalloc(&this->device->myVarX, sizeof(REAL) * outer * numX * numY);
		cudaMalloc(&this->device->myVarY, sizeof(REAL) * outer * numX * numY);
		cudaMalloc(&this->device->u, sizeof(REAL) * outer * numY * numY);
		cudaMalloc(&this->device->v, sizeof(REAL) * outer * numX * numY);
		cudaMalloc(&this->device->a, sizeof(REAL) * outer * numY * numY);
		cudaMalloc(&this->device->b, sizeof(REAL) * outer * numY * numY);
		cudaMalloc(&this->device->c, sizeof(REAL) * outer * numY * numY);
		cudaMalloc(&this->device->y, sizeof(REAL) * outer * numY * numY);
		cudaMalloc(&this->device->yy, sizeof(REAL) * outer * numY * numY);
		cudaMalloc(&this->device->y_trans, sizeof(REAL) * outer * numY * numY);
		cudaMalloc(&this->d_globs, sizeof(PrivGlobs));
		cudaMemcpy(this->d_globs, this->device, sizeof(struct PrivGlobs), cudaMemcpyHostToDevice);
	}
	
	void free() {
		cudaFree(this->device->myX);
		cudaFree(this->device->myY);
		cudaFree(this->device->myDxx);
		cudaFree(this->device->myDyy);
		cudaFree(this->device->myTimeline);
		cudaFree(this->device->myResult);
		cudaFree(this->device->myResult_trans);
		cudaFree(this->device->myVarX);
		cudaFree(this->device->myVarY);
		cudaFree(this->device->u);
		cudaFree(this->device->v);
		cudaFree(this->device->a);
		cudaFree(this->device->b);
		cudaFree(this->device->c);
		cudaFree(this->device->y);
		cudaFree(this->device->yy);
		cudaFree(this->device->y_trans);

		cudaError err = cudaGetLastError();
		if (err != cudaSuccess) {
			printf("CUDA error when freeing: %s\n", cudaGetErrorString(err));
			exit(1);
		}

	}
};

void run(  
		const unsigned int &outer,
		const unsigned int &numX,
		const unsigned int &numY,
		const unsigned int &numT,
		const REAL &s0,
		const REAL &t, 
		const REAL &alpha, 
		const REAL &nu, 
		const REAL &beta,
		REAL *res // [outer] RESULT
);

#endif // PROJ_HELPER_FUNS
