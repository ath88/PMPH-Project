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
	REAL *myVarX; // [numX][numY]
	REAL *myVarY; // [numX][numY]
	
	// operators
	//vector<vector<REAL> > myDxx;  // [numX][4]
	REAL *myDxx; // [numX][4]
	REAL *myDyy; // [numY][4]
	//vector<vector<REAL> > myDyy;  // [numY][4]
	
	// vectors for rollback()
	REAL *u;
	REAL *v;
	REAL *u_trans;
	REAL *a; // [outer][y][max(numX,numY)]
	REAL *b; // [outer][y][max(numX,numY)]
	REAL *c; // [outer][y][max(numX,numY)]
	REAL *y; // [outer][y][max(numX,numY)]
	REAL *yy; // [outer][y][max(numX,numY)]
	REAL *a_trans; // [outer][y][max(numX,numY)]
	REAL *b_trans; // [outer][y][max(numX,numY)]
	REAL *c_trans; // [outer][y][max(numX,numY)]
	REAL *y_trans; // [outer][y][max(numX,numY)]
	REAL *yy_trans; // [outer][y][max(numX,numY)]
	
	// host pointer to struct containing device pointers
	struct PrivGlobs *device;
	// device pointer to struct containing device pointers
	struct PrivGlobs *d_globs;
	
	PrivGlobs() {
	}
	
	void init(
			const unsigned int &numX,
			const unsigned int &numY,
			const unsigned int &numT,
			const unsigned int &outer) {
		this->numX = numX;
		this->numY = numY;
		this->numT = numT;
		this->outer = outer;
		
		this->myX = (REAL *) malloc(sizeof(REAL) * numX);
		this->myY = (REAL *) malloc(sizeof(REAL) * numY);
		
		this->myDxx = (REAL *) malloc(sizeof(REAL) * numX * 4);
		this->myDyy = (REAL *) malloc(sizeof(REAL) * numY * 4);
				
		this->myTimeline = (REAL *) malloc(sizeof(REAL) * numT);
		
		this->myResult = (REAL *) malloc(sizeof(REAL) * outer * numX * numY);
		this->myResult_trans = (REAL *) malloc(sizeof(REAL) * outer * numX * numY);
		this->myVarX = (REAL *) malloc(sizeof(REAL) * outer * numX * numY);
		this->myVarY = (REAL *) malloc(sizeof(REAL) * outer * numX * numY);
		
		
		this->u = (REAL *) malloc(sizeof(REAL) * outer * numY * numY);
		this->v = (REAL *) malloc(sizeof(REAL) * outer * numX * numY);
		this->u_trans = (REAL *) malloc(sizeof(REAL) * outer * numY * numY);
		
		// note: assuming that: numY == max(numX, numY)
		this->a = (REAL *) malloc(sizeof(REAL) * outer * numY * numY);
		this->b = (REAL *) malloc(sizeof(REAL) * outer * numY * numY);
		this->c = (REAL *) malloc(sizeof(REAL) * outer * numY * numY);
		this->y = (REAL *) malloc(sizeof(REAL) * outer * numY * numY);
		this->yy = (REAL *) malloc(sizeof(REAL) * outer * numY * numY);
		this->a_trans = (REAL *) malloc(sizeof(REAL) * outer * numY * numY);
		this->b_trans = (REAL *) malloc(sizeof(REAL) * outer * numY * numY);
		this->c_trans = (REAL *) malloc(sizeof(REAL) * outer * numY * numY);
		this->y_trans = (REAL *) malloc(sizeof(REAL) * outer * numY * numY);
		this->yy_trans = (REAL *) malloc(sizeof(REAL) * outer * numY * numY);
	}

	void cuda_init() {		
		this->device = (struct PrivGlobs *) malloc(sizeof(struct PrivGlobs));
		this->device->numX = numX;
		this->device->numY = numY;
		this->device->numT = numT;
		this->device->outer = outer;
		this->device->myXindex = myXindex;
		this->device->myYindex = myYindex;
		
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
		cudaMalloc(&this->device->u_trans, sizeof(REAL) * outer * numY * numY);
		cudaMalloc(&this->device->a, sizeof(REAL) * outer * numY * numY);
		cudaMalloc(&this->device->b, sizeof(REAL) * outer * numY * numY);
		cudaMalloc(&this->device->c, sizeof(REAL) * outer * numY * numY);
		cudaMalloc(&this->device->y, sizeof(REAL) * outer * numY * numY);
		cudaMalloc(&this->device->yy, sizeof(REAL) * outer * numY * numY);
		cudaMalloc(&this->device->a_trans, sizeof(REAL) * outer * numY * numY);
		cudaMalloc(&this->device->b_trans, sizeof(REAL) * outer * numY * numY);
		cudaMalloc(&this->device->c_trans, sizeof(REAL) * outer * numY * numY);
		cudaMalloc(&this->device->y_trans, sizeof(REAL) * outer * numY * numY);
		cudaMalloc(&this->device->yy_trans, sizeof(REAL) * outer * numY * numY);
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
		cudaFree(this->device->u_trans);
		cudaFree(this->device->a);
		cudaFree(this->device->b);
		cudaFree(this->device->c);
		cudaFree(this->device->y);
		cudaFree(this->device->yy);
		cudaFree(this->device->a_trans);
		cudaFree(this->device->b_trans);
		cudaFree(this->device->c_trans);
		cudaFree(this->device->y_trans);
		cudaFree(this->device->yy_trans);
	}
	
	void copyToDevice() {
		cudaMemcpy(this->device->myX, this->myX, sizeof(REAL) * numX, cudaMemcpyHostToDevice);
		cudaMemcpy(this->device->myY, this->myY, sizeof(REAL) * numY, cudaMemcpyHostToDevice);
		cudaMemcpy(this->device->myDxx, this->myDxx, sizeof(REAL) * numX * 4, cudaMemcpyHostToDevice);
		cudaMemcpy(this->device->myDyy, this->myDyy, sizeof(REAL) * numY * 4, cudaMemcpyHostToDevice);
		cudaMemcpy(this->device->myTimeline, this->myTimeline, sizeof(REAL) * numT, cudaMemcpyHostToDevice);
		cudaMemcpy(this->device->myResult, this->myResult, sizeof(REAL) * outer * numX * numY, cudaMemcpyHostToDevice);
		cudaMemcpy(this->device->myResult_trans, this->myResult_trans, sizeof(REAL) * outer * numX * numY, cudaMemcpyHostToDevice);
		cudaMemcpy(this->device->myVarX, this->myVarX, sizeof(REAL) * numX * numY, cudaMemcpyHostToDevice);
		cudaMemcpy(this->device->myVarY, this->myVarY, sizeof(REAL) * numX * numY, cudaMemcpyHostToDevice);
		cudaMemcpy(this->device->u, this->u, sizeof(REAL) * outer * numY * numX, cudaMemcpyHostToDevice);
		cudaMemcpy(this->device->v, this->v, sizeof(REAL) * outer * numX * numY, cudaMemcpyHostToDevice);
		cudaMemcpy(this->device->a, this->a, sizeof(REAL) * outer * numY * numY, cudaMemcpyHostToDevice);
		cudaMemcpy(this->device->b, this->b, sizeof(REAL) * outer * numY * numY, cudaMemcpyHostToDevice);
		cudaMemcpy(this->device->c, this->c, sizeof(REAL) * outer * numY * numY, cudaMemcpyHostToDevice);
		cudaMemcpy(this->device->y, this->y, sizeof(REAL) * outer * numY * numY, cudaMemcpyHostToDevice);
		cudaMemcpy(this->device->yy, this->yy, sizeof(REAL) * outer * numY * numY, cudaMemcpyHostToDevice);
	}
	
	void copyFromDevice() {
		cudaMemcpy(this->myX, this->device->myX, sizeof(REAL) * numX, cudaMemcpyDeviceToHost);
		cudaMemcpy(this->myY, this->device->myY, sizeof(REAL) * numY, cudaMemcpyDeviceToHost);
		cudaMemcpy(this->myDxx, this->device->myDxx, sizeof(REAL) * numX * 4, cudaMemcpyDeviceToHost);
		cudaMemcpy(this->myDyy, this->device->myDyy, sizeof(REAL) * numY * 4, cudaMemcpyDeviceToHost);
		cudaMemcpy(this->myTimeline, this->device->myTimeline, sizeof(REAL) * numT, cudaMemcpyDeviceToHost);
		cudaMemcpy(this->myResult, this->device->myResult, sizeof(REAL) * outer * numX * numY, cudaMemcpyDeviceToHost);
		cudaMemcpy(this->myResult_trans, this->device->myResult_trans, sizeof(REAL) * outer * numX * numY, cudaMemcpyDeviceToHost);
		cudaMemcpy(this->myVarX, this->device->myVarX, sizeof(REAL) * outer * numX * numY, cudaMemcpyDeviceToHost);
		cudaMemcpy(this->myVarY, this->device->myVarY, sizeof(REAL) * outer * numX * numY, cudaMemcpyDeviceToHost);
		cudaMemcpy(this->u, this->device->u, sizeof(REAL) * outer * numY * numY * numX, cudaMemcpyDeviceToHost);
		cudaMemcpy(this->v, this->device->v, sizeof(REAL) * outer * numY * numX * numY, cudaMemcpyDeviceToHost);
		cudaMemcpy(this->a, this->device->a, sizeof(REAL) * outer * numY * numY, cudaMemcpyDeviceToHost);
		cudaMemcpy(this->b, this->device->b, sizeof(REAL) * outer * numY * numY, cudaMemcpyDeviceToHost);
		cudaMemcpy(this->c, this->device->c, sizeof(REAL) * outer * numY * numY, cudaMemcpyDeviceToHost);
		cudaMemcpy(this->y, this->device->y, sizeof(REAL) * outer * numY * numY, cudaMemcpyDeviceToHost);
		cudaMemcpy(this->yy, this->device->yy, sizeof(REAL) * outer * numY * numY, cudaMemcpyDeviceToHost);
	}
};


void initGrid(const REAL s0, const REAL alpha, const REAL nu,const REAL t,
		const unsigned numX, const unsigned numY, const unsigned numT, PrivGlobs& globs);

void initOperator(const REAL *x, REAL *Dxx, const int n);

void updateParams(const unsigned g, const REAL alpha, const REAL beta, const REAL nu, PrivGlobs &globs);

void setPayoff(const REAL strike, PrivGlobs &globs);

//void tridag(
//	const vector<REAL> &a, // size [n]
//	const vector<REAL> &b, // size [n]
//	const vector<REAL> &c, // size [n]
//	const vector<REAL> &r, // size [n]
//	const int n,
//	vector<REAL> &u, // size [n]
//	vector<REAL> &uu // size [n] temporary
//);

void rollback(const unsigned g, PrivGlobs &globs);

void value(
		PrivGlobs globs,
		const REAL s0,
		const REAL strike, 
		const REAL t, 
		const REAL alpha, 
		const REAL nu, 
		const REAL beta,
		const unsigned int numX,
		const unsigned int numY,
		const unsigned int numT
);

void run_OrigCPU(  
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
