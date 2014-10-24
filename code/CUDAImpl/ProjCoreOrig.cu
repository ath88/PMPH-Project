#include "ProjHelperFun.h"
#include "Constants.h"
#include "timers.h"

#include <cuda_runtime.h>

void report_cuda_error(char*);
void rollback0(unsigned int, PrivGlobs&);
void rollback0_host(unsigned int, PrivGlobs&);

__global__ void updateParams_kernel(const unsigned g, const REAL alpha, const REAL beta, const REAL nu, PrivGlobs &globs) {

	unsigned int j = threadIdx.y + blockDim.y * blockIdx.y;

	if (j >= globs.numY) return;
	//printf("j: %d, numY: %d\n", j, globs.numY);

	for(unsigned i=0;i<globs.numX;++i) {
		globs.myVarX[i*globs.numY + j] = exp(2.0 * (
				beta*log(globs.myX[i])
				+ globs.myY[j]
				- 0.5*nu*nu*globs.myTimeline[g]));
		globs.myVarY[i*globs.numY + j] = exp(2.0 * (
				alpha*log(globs.myX[i])
				+ globs.myY[j]
				- 0.5*nu*nu*globs.myTimeline[g])); // nu*nu
	} 
//	printf("Kernel is runnings! Yay this is fucking awesome!\n");
}

__global__ void rollback0_kernel(unsigned int g, PrivGlobs *globs) {
	unsigned numX = globs->numX;
	unsigned numY = globs->numY;
	unsigned numZ = numX > numY ? numX : numY;
	
	unsigned i, j;
	
	REAL dtInv = 1.0 / (globs->myTimeline[g+1] - globs->myTimeline[g]);
	
	int outerId = 0;
	int yId = 0;
	
	REAL *u = globs->u
			+ outerId * numY * numY * numX // [outer][y][numY][numX]
			+ yId * numY * numX; 

	for(i=0; i<numX; i++) {
		for(j=0; j<numY; j++) {
			u[j*numX + i] = dtInv*globs->myResult[i*globs->numY + j];
			
			if(i > 0) { 
				u[j*numX + i] += 0.5 * (0.5
						* globs->myVarX[i*globs->numY + j] * globs->myDxx[i*4 + 0])
						* globs->myResult[(i-1)*globs->numY + j];
			}
			u[j*numX + i] += 0.5 * (0.5
					* globs->myVarX[i*globs->numY + j] * globs->myDxx[i*4 + 1])
					* globs->myResult[i*globs->numY + j];
			if(i < numX - 1) {
				u[j*numX + i] += 0.5 * (0.5
						* globs->myVarX[i*globs->numY + j] * globs->myDxx[i*4 + 2])
						* globs->myResult[(i+1)*globs->numY + j];
			}
		}
	}
}


TIMER_DEFINE(run_OrigCPU);
	TIMER_DEFINE(updateParams);
	TIMER_DEFINE(rollback);
		TIMER_DEFINE(rollback_0);
		TIMER_DEFINE(rollback_1);
		TIMER_DEFINE(rollback_2);
		TIMER_DEFINE(rollback_3);

void updateParams(const unsigned g, const REAL alpha, const REAL beta, const REAL nu, PrivGlobs& globs) {
	TIMER_START(updateParams);
	
	for(unsigned i=0;i<globs.numX;++i) {
		for(unsigned j=0;j<globs.numY;++j) {
			globs.myVarX[i*globs.numY + j] = exp(2.0 * (
					beta*log(globs.myX[i])
					+ globs.myY[j]
					- 0.5*nu*nu*globs.myTimeline[g]));
			globs.myVarY[i*globs.numY + j] = exp(2.0 * (
					alpha*log(globs.myX[i])
					+ globs.myY[j]
					- 0.5*nu*nu*globs.myTimeline[g])); // nu*nu
		}
	}
	
	TIMER_STOP(updateParams);
}

void updateParams_host(const unsigned g, const REAL alpha, const REAL beta, const REAL nu, PrivGlobs& globs) {
	TIMER_START(updateParams);

	int outer = globs.outer;
	int numX = globs.numX;
	int numY = globs.numY;
	int numT = globs.numT;
	
	PrivGlobs h_globs;
	h_globs = globs;

	cudaMalloc(&h_globs.myX, sizeof(REAL) * numX);
	cudaMalloc(&h_globs.myY, sizeof(REAL) * numY);
	cudaMalloc(&h_globs.myDxx, sizeof(REAL) * numX * 4);
	cudaMalloc(&h_globs.myDyy, sizeof(REAL) * numY * 4);
	cudaMalloc(&h_globs.myTimeline, sizeof(REAL) * numT);
	cudaMalloc(&h_globs.myResult, sizeof(REAL) * numX * numY);
	cudaMalloc(&h_globs.myVarX, sizeof(REAL) * numX * numY);
	cudaMalloc(&h_globs.myVarY, sizeof(REAL) * numX * numY);
	cudaMalloc(&h_globs.u, sizeof(REAL) * outer * numY * numY * numX);
	cudaMalloc(&h_globs.v, sizeof(REAL) * outer * numY * numX * numY);
	cudaMalloc(&h_globs.a, sizeof(REAL) * outer * numY * numY);
	cudaMalloc(&h_globs.b, sizeof(REAL) * outer * numY * numY);
	cudaMalloc(&h_globs.c, sizeof(REAL) * outer * numY * numY);
	cudaMalloc(&h_globs.y, sizeof(REAL) * outer * numY * numY);
	cudaMalloc(&h_globs.yy, sizeof(REAL) * outer * numY * numY);
	
	report_cuda_error("One\n");

 	cudaMemcpy(h_globs.myX, globs.myX, sizeof(REAL) * numX, cudaMemcpyHostToDevice);
	report_cuda_error("Hej\n");
	cudaMemcpy(h_globs.myY, globs.myY, sizeof(REAL) * numY, cudaMemcpyHostToDevice);
	cudaMemcpy(h_globs.myDxx, globs.myDxx, sizeof(REAL) * numX * 4, cudaMemcpyHostToDevice);
	cudaMemcpy(h_globs.myDyy, globs.myDyy, sizeof(REAL) * numY * 4, cudaMemcpyHostToDevice);
	cudaMemcpy(h_globs.myTimeline, globs.myTimeline, sizeof(REAL) * numT, cudaMemcpyHostToDevice);
	cudaMemcpy(h_globs.myResult, globs.myResult, sizeof(REAL) * numX * numY, cudaMemcpyHostToDevice);
	cudaMemcpy(h_globs.myVarX, globs.myVarX, sizeof(REAL) * numX * numY, cudaMemcpyHostToDevice);
	cudaMemcpy(h_globs.myVarY, globs.myVarY, sizeof(REAL) * numX * numY, cudaMemcpyHostToDevice);
	cudaMemcpy(h_globs.u, globs.u, sizeof(REAL) * outer * numY * numY * numX, cudaMemcpyHostToDevice);
	cudaMemcpy(h_globs.v, globs.v, sizeof(REAL) * outer * numY * numX * numY, cudaMemcpyHostToDevice);
	cudaMemcpy(h_globs.a, globs.a, sizeof(REAL) * outer * numY * numY, cudaMemcpyHostToDevice);
	cudaMemcpy(h_globs.b, globs.b, sizeof(REAL) * outer * numY * numY, cudaMemcpyHostToDevice);
	cudaMemcpy(h_globs.c, globs.c, sizeof(REAL) * outer * numY * numY, cudaMemcpyHostToDevice);
	cudaMemcpy(h_globs.y, globs.y, sizeof(REAL) * outer * numY * numY, cudaMemcpyHostToDevice);
	cudaMemcpy(h_globs.yy, globs.yy, sizeof(REAL) * outer * numY * numY, cudaMemcpyHostToDevice);
	
	report_cuda_error("Two\n");

	PrivGlobs *d_globs;
	cudaMalloc((void **) &d_globs, sizeof(PrivGlobs));
	cudaMemcpy(d_globs, &h_globs, sizeof(PrivGlobs), cudaMemcpyHostToDevice);

	report_cuda_error("Three\n");
	
	updateParams_kernel <<< dim3(1, globs.numY), dim3(32,32) >>> (g, alpha, beta, nu, *d_globs);

	report_cuda_error("Four\n");

	cudaMemcpy(globs.myX, h_globs.myX, sizeof(REAL) * numX, cudaMemcpyDeviceToHost);
	cudaMemcpy(globs.myY, h_globs.myY, sizeof(REAL) * numY, cudaMemcpyDeviceToHost);
	cudaMemcpy(globs.myDxx, h_globs.myDxx, sizeof(REAL) * numX * 4, cudaMemcpyDeviceToHost);
	cudaMemcpy(globs.myDyy, h_globs.myDyy, sizeof(REAL) * numY * 4, cudaMemcpyDeviceToHost);
	cudaMemcpy(globs.myTimeline, h_globs.myTimeline, sizeof(REAL) * numT, cudaMemcpyDeviceToHost);
	cudaMemcpy(globs.myResult, h_globs.myResult, sizeof(REAL) * numX * numY, cudaMemcpyDeviceToHost);
	cudaMemcpy(globs.myVarX, h_globs.myVarX, sizeof(REAL) * numX * numY, cudaMemcpyDeviceToHost);
	cudaMemcpy(globs.myVarY, h_globs.myVarY, sizeof(REAL) * numX * numY, cudaMemcpyDeviceToHost);
	cudaMemcpy(globs.u, h_globs.u, sizeof(REAL) * outer * numY * numY * numX, cudaMemcpyDeviceToHost);
	cudaMemcpy(globs.v, h_globs.v, sizeof(REAL) * outer * numY * numX * numY, cudaMemcpyDeviceToHost);
	cudaMemcpy(globs.a, h_globs.a, sizeof(REAL) * outer * numY * numY, cudaMemcpyDeviceToHost);
	cudaMemcpy(globs.b, h_globs.b, sizeof(REAL) * outer * numY * numY, cudaMemcpyDeviceToHost);
	cudaMemcpy(globs.c, h_globs.c, sizeof(REAL) * outer * numY * numY, cudaMemcpyDeviceToHost);
	cudaMemcpy(globs.y, h_globs.y, sizeof(REAL) * outer * numY * numY, cudaMemcpyDeviceToHost);
	cudaMemcpy(globs.yy, h_globs.yy, sizeof(REAL) * outer * numY * numY, cudaMemcpyDeviceToHost);
	report_cuda_error("Five\n");
	
	cudaFree(h_globs.myX);
	cudaFree(h_globs.myY);
	cudaFree(h_globs.myDxx);
	cudaFree(h_globs.myDyy);
	cudaFree(h_globs.myTimeline);
	cudaFree(h_globs.myResult);
	cudaFree(h_globs.myVarX);
	cudaFree(h_globs.myVarY);
	cudaFree(h_globs.u);
	cudaFree(h_globs.v);
	cudaFree(h_globs.a);
	cudaFree(h_globs.b);
	cudaFree(h_globs.c);
	cudaFree(h_globs.y);
	cudaFree(h_globs.yy);
	
	TIMER_STOP(updateParams);
}

void setPayoff(const REAL strike, PrivGlobs& globs) {
	for(unsigned i = 0; i < globs.numX; ++i) {
		//REAL payoff = max(globs.myX[i] - strike, (REAL)0.0);
		REAL payoff = globs.myX[i] - strike > (REAL)0.0
				? globs.myX[i] - strike : (REAL)0.0;
		for(unsigned j = 0; j < globs.numY; ++j) {
			globs.myResult[i*globs.numY + j] = payoff;
		}
	}
}

inline void tridag(
	const REAL *a,   // size [n]
	const REAL *b,   // size [n]
	const REAL *c,   // size [n]
	const REAL *r,   // size [n]
	const int n,
	REAL *u,   // size [n]
	REAL *uu   // size [n] temporary
) {
	int i, offset;
	REAL beta;
	
	u[0] = r[0];
	uu[0] = b[0];
	
	for(i=1; i<n; i++) {
		beta  = a[i] / uu[i-1];
		
		uu[i] = b[i] - beta*c[i-1];
		u[i]  = r[i] - beta*u[i-1];
	}

#if 1
	// X) this is a backward recurrence
	u[n-1] = u[n-1] / uu[n-1];
	for(i=n-2; i>=0; i--) {
		u[i] = (u[i] - c[i]*u[i+1]) / uu[i];
	}
#else
	// Hint: X) can be written smth like (once you make a non-constant)
	for(i=0; i<n; i++) a[i] =  u[n-1-i];
	a[0] = a[0] / uu[n-1];
	for(i=1; i<n; i++) a[i] = (a[i] - c[n-1-i]*a[i-1]) / uu[n-1-i];
	for(i=0; i<n; i++) u[i] = a[n-1-i];
#endif
}

void rollback(const unsigned g, PrivGlobs &globs) {
	TIMER_START(rollback);
	
	unsigned numX = globs.numX;
	unsigned numY = globs.numY;
	
	//unsigned numZ = max(numX,numY);
	unsigned numZ = numX > numY ? numX : numY;
	
	unsigned i, j;
	
	REAL dtInv = 1.0 / (globs.myTimeline[g+1] - globs.myTimeline[g]);
	
	int outerId = 0;
	int yId = 0;
	
	REAL *u = globs.u
			+ outerId * yId * numX * numY
			+ yId * numY * numX; // [outer][y][numY][numX]
	REAL *v = globs.v
			+ outerId * yId * numX * numY
			+ yId * numY * numX; // [outer][y][numY][numX]
	REAL *a = globs.a
			+ outerId * yId * numY
			+ yId * numY; // [outer][y][max(numX,numY)]
	REAL *b = globs.b
			+ outerId * yId * numY
			+ yId * numY; // [outer][y][max(numX,numY)]
	REAL *c = globs.c
			+ outerId * yId * numY
			+ yId * numY; // [outer][y][max(numX,numY)]
	REAL *y = globs.y
			+ outerId * yId * numY
			+ yId * numY; // [outer][y][max(numX,numY)]
	REAL *yy = globs.yy
			+ outerId * yId * numY
			+ yId * numY; // [outer][y][max(numX,numY)]
	
	// explicit x
	TIMER_START(rollback_0);
	rollback0(g, globs);
	TIMER_STOP(rollback_0);
	
	// explicit y
	TIMER_START(rollback_1);
	for(j=0; j<numY; j++) {
		for(i=0; i<numX; i++) {
			v[i*numY + j] = 0.0;
			
			if(j > 0) {
				v[i*numY + j] += (0.5
						* globs.myVarY[i*globs.numY + j] * globs.myDyy[j*4 + 0])
						* globs.myResult[i*globs.numY + j-1];
			}
			v[i*numY + j] += (0.5
					* globs.myVarY[i*globs.numY + j] * globs.myDyy[j*4 + 1])
					* globs.myResult[i*globs.numY + j];
			if(j < numY - 1) {
				v[i*numY + j] += (0.5
						* globs.myVarY[i*globs.numY + j] * globs.myDyy[j*4 + 2])
						* globs.myResult[i*globs.numY + j+1];
			}
			u[j*numX + i] += v[i*numY + j];
		}
	}
	TIMER_STOP(rollback_1);
	
	// implicit x
	TIMER_START(rollback_2);
	for(j=0; j<numY; j++) {
		for(i=0; i<numX; i++) { // here a, b,c should have size [numX]
			a[i] = -0.5 * (0.5 * globs.myVarX[i*globs.numY + j]
					* globs.myDxx[i*4 + 0]);
			b[i] = dtInv - 0.5 * (0.5 * globs.myVarX[i*globs.numY + j]
					* globs.myDxx[i*4 + 1]);
			c[i] = -0.5 * (0.5 * globs.myVarX[i*globs.numY + j]
					* globs.myDxx[i*4 + 2]);
		}
		// here yy should have size [numX]
		tridag(a,b,c,u + numX*j,numX,u + numX*j,yy);
	}
	TIMER_STOP(rollback_2);
	
	// implicit y
	TIMER_START(rollback_3);
	for(i=0; i<numX; i++) { 
		for(j=0; j<numY; j++) { // here a, b, c should have size [numY]
			a[j] = -0.5 * (0.5 * globs.myVarY[i*globs.numY + j]
					* globs.myDyy[j*4 + 0]);
			b[j] = dtInv - 0.5 * (0.5 * globs.myVarY[i*globs.numY + j]
					* globs.myDyy[j*4 + 1]);
			c[j] = -0.5 * (0.5 * globs.myVarY[i*globs.numY + j]
					* globs.myDyy[j*4 + 2]);
		}
		
		for(j=0; j<numY; j++) {
			y[j] = dtInv*u[j*numX + i] - 0.5*v[i*numY + j];
		}
		
		// here yy should have size [numY]
		tridag(a,b,c,y,numY,globs.myResult + i*globs.numY,yy);
	}
	TIMER_STOP(rollback_3);
	
	TIMER_STOP(rollback);
}

void report_cuda_error(char* id) {
	//printf(id);
	cudaError err;
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Error: %s\n", cudaGetErrorString(err));
		exit(1);
	}
}

void rollback0_host (unsigned int g, PrivGlobs &globs) {

	int outer = globs.outer;
	int numX = globs.numX;
	int numY = globs.numY;
	int numT = globs.numT;
	
	PrivGlobs h_globs;

	cudaMalloc(&h_globs.myX, sizeof(REAL) * numX);
	cudaMalloc(&h_globs.myY, sizeof(REAL) * numY);
	cudaMalloc(&h_globs.myDxx, sizeof(REAL) * numX * 4);
	cudaMalloc(&h_globs.myDyy, sizeof(REAL) * numY * 4);
	cudaMalloc(&h_globs.myTimeline, sizeof(REAL) * numT);
	cudaMalloc(&h_globs.myResult, sizeof(REAL) * numX * numY);
	cudaMalloc(&h_globs.myVarX, sizeof(REAL) * numX * numY);
	cudaMalloc(&h_globs.myVarY, sizeof(REAL) * numX * numY);
	cudaMalloc(&h_globs.u, sizeof(REAL) * outer * numY * numY * numX);
	cudaMalloc(&h_globs.v, sizeof(REAL) * outer * numY * numX * numY);
	cudaMalloc(&h_globs.a, sizeof(REAL) * outer * numY * numY);
	cudaMalloc(&h_globs.b, sizeof(REAL) * outer * numY * numY);
	cudaMalloc(&h_globs.c, sizeof(REAL) * outer * numY * numY);
	cudaMalloc(&h_globs.y, sizeof(REAL) * outer * numY * numY);
	cudaMalloc(&h_globs.yy, sizeof(REAL) * outer * numY * numY);
	
	report_cuda_error("One\n");

 	cudaMemcpy(h_globs.myX, globs.myX, sizeof(REAL) * numX, cudaMemcpyHostToDevice);
	report_cuda_error("Hej\n");
	cudaMemcpy(h_globs.myY, globs.myY, sizeof(REAL) * numY, cudaMemcpyHostToDevice);
	cudaMemcpy(h_globs.myDxx, globs.myDxx, sizeof(REAL) * numX * 4, cudaMemcpyHostToDevice);
	cudaMemcpy(h_globs.myDyy, globs.myDyy, sizeof(REAL) * numY * 4, cudaMemcpyHostToDevice);
	cudaMemcpy(h_globs.myTimeline, globs.myTimeline, sizeof(REAL) * numT, cudaMemcpyHostToDevice);
	cudaMemcpy(h_globs.myResult, globs.myResult, sizeof(REAL) * numX * numY, cudaMemcpyHostToDevice);
	cudaMemcpy(h_globs.myVarX, globs.myVarX, sizeof(REAL) * numX * numY, cudaMemcpyHostToDevice);
	cudaMemcpy(h_globs.myVarY, globs.myVarY, sizeof(REAL) * numX * numY, cudaMemcpyHostToDevice);
	cudaMemcpy(h_globs.u, globs.u, sizeof(REAL) * outer * numY * numY * numX, cudaMemcpyHostToDevice);
	cudaMemcpy(h_globs.v, globs.v, sizeof(REAL) * outer * numY * numX * numY, cudaMemcpyHostToDevice);
	cudaMemcpy(h_globs.a, globs.a, sizeof(REAL) * outer * numY * numY, cudaMemcpyHostToDevice);
	cudaMemcpy(h_globs.b, globs.b, sizeof(REAL) * outer * numY * numY, cudaMemcpyHostToDevice);
	cudaMemcpy(h_globs.c, globs.c, sizeof(REAL) * outer * numY * numY, cudaMemcpyHostToDevice);
	cudaMemcpy(h_globs.y, globs.y, sizeof(REAL) * outer * numY * numY, cudaMemcpyHostToDevice);
	cudaMemcpy(h_globs.yy, globs.yy, sizeof(REAL) * outer * numY * numY, cudaMemcpyHostToDevice);
	
	report_cuda_error("Two\n");

	PrivGlobs *d_globs;
	cudaMalloc((void **) &d_globs, sizeof(PrivGlobs));
	cudaMemcpy(d_globs, &h_globs, sizeof(PrivGlobs), cudaMemcpyHostToDevice);

	report_cuda_error("Three\n");
	
	rollback0_kernel <<< 1,1 >>> (g, d_globs);

	report_cuda_error("Four\n");

	cudaMemcpy(globs.myX, h_globs.myX, sizeof(REAL) * numX, cudaMemcpyDeviceToHost);
	cudaMemcpy(globs.myY, h_globs.myY, sizeof(REAL) * numY, cudaMemcpyDeviceToHost);
	cudaMemcpy(globs.myDxx, h_globs.myDxx, sizeof(REAL) * numX * 4, cudaMemcpyDeviceToHost);
	cudaMemcpy(globs.myDyy, h_globs.myDyy, sizeof(REAL) * numY * 4, cudaMemcpyDeviceToHost);
	cudaMemcpy(globs.myTimeline, h_globs.myTimeline, sizeof(REAL) * numT, cudaMemcpyDeviceToHost);
	cudaMemcpy(globs.myResult, h_globs.myResult, sizeof(REAL) * numX * numY, cudaMemcpyDeviceToHost);
	cudaMemcpy(globs.myVarX, h_globs.myVarX, sizeof(REAL) * numX * numY, cudaMemcpyDeviceToHost);
	cudaMemcpy(globs.myVarY, h_globs.myVarY, sizeof(REAL) * numX * numY, cudaMemcpyDeviceToHost);
	cudaMemcpy(globs.u, h_globs.u, sizeof(REAL) * outer * numY * numY * numX, cudaMemcpyDeviceToHost);
	cudaMemcpy(globs.v, h_globs.v, sizeof(REAL) * outer * numY * numX * numY, cudaMemcpyDeviceToHost);
	cudaMemcpy(globs.a, h_globs.a, sizeof(REAL) * outer * numY * numY, cudaMemcpyDeviceToHost);
	cudaMemcpy(globs.b, h_globs.b, sizeof(REAL) * outer * numY * numY, cudaMemcpyDeviceToHost);
	cudaMemcpy(globs.c, h_globs.c, sizeof(REAL) * outer * numY * numY, cudaMemcpyDeviceToHost);
	cudaMemcpy(globs.y, h_globs.y, sizeof(REAL) * outer * numY * numY, cudaMemcpyDeviceToHost);
	cudaMemcpy(globs.yy, h_globs.yy, sizeof(REAL) * outer * numY * numY, cudaMemcpyDeviceToHost);
	report_cuda_error("Five\n");
	
	cudaFree(h_globs.myX);
	cudaFree(h_globs.myY);
	cudaFree(h_globs.myDxx);
	cudaFree(h_globs.myDyy);
	cudaFree(h_globs.myTimeline);
	cudaFree(h_globs.myResult);
	cudaFree(h_globs.myVarX);
	cudaFree(h_globs.myVarY);
	cudaFree(h_globs.u);
	cudaFree(h_globs.v);
	cudaFree(h_globs.a);
	cudaFree(h_globs.b);
	cudaFree(h_globs.c);
	cudaFree(h_globs.y);
	cudaFree(h_globs.yy);
}

void rollback0 (unsigned int g, PrivGlobs &globs) {
	unsigned numX = globs.numX;
	unsigned numY = globs.numY;
	unsigned numZ = numX > numY ? numX : numY;
	
	unsigned i, j;
	
	REAL dtInv = 1.0 / (globs.myTimeline[g+1] - globs.myTimeline[g]);
	
	int outerId = 0;
	int yId = 0;
	
	REAL *u = globs.u
			+ outerId * numY * numY * numX // [outer][y][numY][numX]
			+ yId * numY * numX; 

	for(i=0; i<numX; i++) {
		for(j=0; j<numY; j++) {
			u[j*numX + i] = dtInv*globs.myResult[i*globs.numY + j];
			
			if(i > 0) { 
				u[j*numX + i] += 0.5 * (0.5
						* globs.myVarX[i*globs.numY + j] * globs.myDxx[i*4 + 0])
						* globs.myResult[(i-1)*globs.numY + j];
			}
			u[j*numX + i] += 0.5 * (0.5
					* globs.myVarX[i*globs.numY + j] * globs.myDxx[i*4 + 1])
					* globs.myResult[i*globs.numY + j];
			if(i < numX - 1) {
				u[j*numX + i] += 0.5 * (0.5
						* globs.myVarX[i*globs.numY + j] * globs.myDxx[i*4 + 2])
						* globs.myResult[(i+1)*globs.numY + j];
			}
		}
	}
}

REAL value(
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
) {
	initGrid(s0,alpha,nu,t, numX, numY, numT, globs);
	initOperator(globs.myX, globs.myDxx, globs.numX);
	initOperator(globs.myY, globs.myDyy, globs.numY);

	setPayoff(strike, globs);
	
	unsigned int count = 0;
	for(int i = globs.numT-2; i>=0; --i) {
		printf("count: %d\n", count++);
		updateParams_host(i,alpha,beta,nu,globs);
		
		rollback(i, globs);
	}
	
	
	return globs.myResult[globs.myXindex*globs.numY + globs.myYindex];
}

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
) {
	TIMER_INIT(run_OrigCPU);
	TIMER_START(run_OrigCPU);
	
	TIMER_INIT(rollback_0);
	TIMER_INIT(rollback_1);
	TIMER_INIT(rollback_2);
	TIMER_INIT(rollback_3);
	
	REAL strike;
	PrivGlobs globs;
	globs.init(numX, numY, numT, outer);
	
	TIMER_INIT(updateParams);
	TIMER_INIT(rollback);
	
	for(unsigned i = 0; i < outer; ++i) {
		strike = 0.001*i;
		res[i] = value(
				globs, s0, strike, t,
				alpha, nu, beta,
				numX, numY, numT );
	}
	
	TIMER_STOP(run_OrigCPU);
	
	TIMER_REPORT(run_OrigCPU);
	TIMER_GROUP();
	TIMER_REPORT(updateParams);
	TIMER_REPORT(rollback);
	TIMER_GROUP();
	TIMER_REPORT(rollback_0);
	TIMER_REPORT(rollback_1);
	TIMER_REPORT(rollback_2);
	TIMER_REPORT(rollback_3);
	TIMER_GROUP_END();
	TIMER_GROUP_END();
}

//#endif // PROJ_CORE_ORIG
