#include "ProjHelperFun.h"
#include "Constants.h"
#include "timers.h"

#include <cuda_runtime.h>
#include <cuda.h>

void report_cuda_error(char*);

__device__ inline void d_tridag(const REAL*, const REAL*, const REAL*, const REAL*, const int, REAL*, REAL*);


// can only run up to 1024 threads, since it doesnt use blockIdx and blockDim
__global__ void copyResult_kernel(REAL *d_res, PrivGlobs &globs) {
	unsigned o = threadIdx.x;
	REAL *myResult = globs.myResult + o * globs.numY * globs.numX;
	d_res[o] = myResult[globs.myXindex * globs.numY + globs.myYindex];
}

// can only run up to 1024 threads, since it doesnt use blockIdx and blockDim
__global__ void setPayoff_kernel(PrivGlobs& globs) {
	unsigned o = threadIdx.x;
	REAL *myResult = globs.myResult + o * globs.numY * globs.numX;
	const REAL strike = 0.001 * o;

	for(unsigned i = 0; i < globs.numX; ++i) {
		//REAL payoff = max(globs.myX[i] - strike, (REAL)0.0);
		REAL payoff = globs.myX[i] - strike > (REAL)0.0
				? globs.myX[i] - strike : (REAL)0.0;
		for(unsigned j = 0; j < globs.numY; ++j) {
			myResult[i*globs.numY + j ] = payoff;
		}
	}
}

__global__ void updateParams_kernel(const unsigned g, const REAL alpha, const REAL beta, const REAL nu, PrivGlobs &globs) {
	unsigned int o = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int j = threadIdx.y + blockDim.y * blockIdx.y;
	unsigned numX = globs.numX;
	unsigned numY = globs.numY;

	if (o >= globs.outer) return;
	if (j >= globs.numY) return;

	REAL *myVarX = globs.myVarX + o * numX * numY;
	REAL *myVarY = globs.myVarY + o * numX * numY;

	for(unsigned i=0;i<globs.numX;++i) {
		myVarX[i*globs.numY + j] = exp(2.0 * (
				beta*log(globs.myX[i])
				+ globs.myY[j]
				- 0.5*nu*nu*globs.myTimeline[g]));
		myVarY[i*globs.numY + j] = exp(2.0 * (
				alpha*log(globs.myX[i])
				+ globs.myY[j]
				- 0.5*nu*nu*globs.myTimeline[g])); // nu*nu
	} 
}

__global__ void rollback0_kernel(unsigned int g, PrivGlobs &globs) {
	unsigned numX = globs.numX;
	unsigned numY = globs.numY;
	
	unsigned int j = threadIdx.y + blockDim.y * blockIdx.y;
	unsigned int o = threadIdx.x + blockDim.x * blockIdx.x;
	if (j >= globs.numY) return;
	if (o >= globs.outer) return;
	
	REAL dtInv = 1.0 / (globs.myTimeline[g+1] - globs.myTimeline[g]);
	
	REAL *myResult = globs.myResult + o * numY * numX;
	REAL *u = globs.u + o * numY * numX; // [outer][numY][numX]

	for(int i=0; i<numX; i++) {
		u[j*numX + i] = dtInv*myResult[i*globs.numY + j];
			
		if(i > 0) { 
			u[j*numX + i] += 0.5 * (0.5
					* globs.myVarX[i*globs.numY + j] * globs.myDxx[i*4 + 0])
					* myResult[(i-1)*globs.numY + j];
		}
		u[j*numX + i] += 0.5 * (0.5
				* globs.myVarX[i*globs.numY + j] * globs.myDxx[i*4 + 1])
				* myResult[i*globs.numY + j];
		if(i < numX - 1) {
			u[j*numX + i] += 0.5 * (0.5
					* globs.myVarX[i*globs.numY + j] * globs.myDxx[i*4 + 2])
					* myResult[(i+1)*globs.numY + j];
		}
	}
}

__global__ void rollback1_kernel(unsigned int g, PrivGlobs &globs) {
	unsigned numX = globs.numX;
	unsigned numY = globs.numY;
	
	unsigned int j = threadIdx.y + blockDim.y * blockIdx.y;
	unsigned int o = threadIdx.x + blockDim.x * blockIdx.x;
	if (j >= globs.numY) return;
	if (o >= globs.outer) return;
	
	REAL *myResult = globs.myResult + o * numY * numX;
	REAL *u = globs.u + o * numY * numX; // [outer][numY][numX]
	REAL *v = globs.v + o * numY * numX; // [outer][numY][numX]

	for(int i=0; i<numX; i++) {
		v[i*numY + j] = 0.0;
		
		if(j > 0) {
			v[i*numY + j] += (0.5
					* globs.myVarY[i*globs.numY + j] * globs.myDyy[j*4 + 0])
					* myResult[i*globs.numY + j-1];
		}
		v[i*numY + j] += (0.5
				* globs.myVarY[i*globs.numY + j] * globs.myDyy[j*4 + 1])
				* myResult[i*globs.numY + j];
		if(j < numY - 1) {
			v[i*numY + j] += (0.5
					* globs.myVarY[i*globs.numY + j] * globs.myDyy[j*4 + 2])
					* myResult[i*globs.numY + j+1];
		}
		u[j*numX + i] += v[i*numY + j];
	}
}

__global__ void rollback2_kernel(unsigned int g, PrivGlobs &globs) {
	unsigned numX = globs.numX;
	unsigned numY = globs.numY;
	
	unsigned int j = threadIdx.y + blockDim.y * blockIdx.y;
	unsigned int o = threadIdx.x + blockDim.x * blockIdx.x;
	if (j >= globs.numY) return;
	if (o >= globs.outer) return;
	
	REAL dtInv = 1.0 / (globs.myTimeline[g+1] - globs.myTimeline[g]);
	
	REAL *u = globs.u + o * numY * numX; // [outer][numY][numX]
	REAL *a = globs.a
			+ o * numY * numY
			+ j * numY; // [outer][y][max(numX,numY)]
	REAL *b = globs.b
			+ o * numY * numY
			+ j  * numY; // [outer][y][max(numX,numY)]
	REAL *c = globs.c
			+ o * numY * numY
			+ j * numY; // [outer][y][max(numX,numY)]
	REAL *yy = globs.yy
			+ o * numY * numY
			+ j * numY; // [outer][y][max(numX,numY)]
	
	for(int i=0; i<numX; i++) { // here a, b,c should have size [numX]
		a[i] = -0.5 * (0.5 * globs.myVarX[i*globs.numY + j]
				* globs.myDxx[i*4 + 0]);
		b[i] = dtInv - 0.5 * (0.5 * globs.myVarX[i*globs.numY + j]
				* globs.myDxx[i*4 + 1]);
		c[i] = -0.5 * (0.5 * globs.myVarX[i*globs.numY + j]
				* globs.myDxx[i*4 + 2]);
	}
	// here yy should have size [numX]
	d_tridag(a,b,c,u + numX*j,numX,u + numX*j,yy);
}

__global__ void rollback3_kernel(unsigned int g, PrivGlobs &globs) {
	unsigned numX = globs.numX;
	unsigned numY = globs.numY;
	
	unsigned int i = threadIdx.y + blockDim.y * blockIdx.y;
	unsigned int o = threadIdx.x + blockDim.x * blockIdx.x;
	if (i >= globs.numX) return;
	if (o >= globs.outer) return;
	
	REAL dtInv = 1.0 / (globs.myTimeline[g+1] - globs.myTimeline[g]);
	
	REAL *myResult = globs.myResult + o * numY * numX;
	REAL *u = globs.u + o * numY * numX; // [outer][numY][numX]
	REAL *v = globs.v + o * numY * numX; // [outer][numY][numX]
	REAL *a = globs.a
			+ o * numY * numY
			+ i * numY; // [outer][y][max(numX,numY)]
	REAL *b = globs.b
			+ o * numY * numY
			+ i * numY; // [outer][y][max(numX,numY)]
	REAL *c = globs.c
			+ o * numY * numY
			+ i * numY; // [outer][y][max(numX,numY)]
	REAL *y = globs.y
			+ o * numY * numY
			+ i * numY; // [outer][y][max(numX,numY)]
	REAL *yy = globs.yy
			+ o * numY * numY
			+ i * numY; // [outer][y][max(numX,numY)]
	
	for(int j=0; j<numY; j++) { // here a, b, c should have size [numY]
		a[j] = -0.5 * (0.5 * globs.myVarY[i*globs.numY + j]
				* globs.myDyy[j*4 + 0]);
		b[j] = dtInv - 0.5 * (0.5 * globs.myVarY[i*globs.numY + j]
				* globs.myDyy[j*4 + 1]);
		c[j] = -0.5 * (0.5 * globs.myVarY[i*globs.numY + j]
				* globs.myDyy[j*4 + 2]);
	}
	
	for(int j=0; j<numY; j++) {
		y[j] = dtInv*u[j*numX + i] - 0.5*v[i*numY + j];
	}
	
	// here yy should have size [numY]
	d_tridag(a,b,c,y,numY,myResult + i*globs.numY,yy);
}

TIMER_DEFINE(run_OrigCPU);
	TIMER_DEFINE(updateParams);
	TIMER_DEFINE(rollback);
		TIMER_DEFINE(rollback_0);
		TIMER_DEFINE(rollback_1);
		TIMER_DEFINE(rollback_2);
		TIMER_DEFINE(rollback_3);

void updateParams_host(const unsigned g, const REAL alpha, const REAL beta, const REAL nu, PrivGlobs& globs) {
	TIMER_START(updateParams);
	dim3 blocks = dim3(ceil((globs.outer+0.f)/32),ceil((globs.numY+0.f)/32));
	dim3 threads = dim3(32,32);
	updateParams_kernel <<< blocks, threads >>> (g, alpha, beta, nu, *globs.d_globs);
	//cudaDeviceSynchronize();
	report_cuda_error("Two\n");
	TIMER_STOP(updateParams);
}

void setPayoff_host(PrivGlobs& globs) {
	setPayoff_kernel <<< 1, globs.outer >>> (*globs.d_globs);
	report_cuda_error("Two\n");
}

__device__ inline void d_tridag(
	const REAL *a,   // size [n]
	const REAL *b,   // size [n]
	const REAL *c,   // size [n]
	const REAL *r,   // size [n]
	const int n,
	REAL *u,   // size [n]
	REAL *uu   // size [n] temporary
) {
	int i;
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

void report_cuda_error(char* id) {
	cudaError err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("CUDA error at id \"%s\": %s\n", id, cudaGetErrorString(err));
		exit(1);
	}
}

void rollback0_host (unsigned int g, PrivGlobs &globs) {
	dim3 blocks = dim3(ceil((globs.outer+0.f)/32),ceil((globs.numY+0.f)/32));
	dim3 threads = dim3(32,32);
	rollback0_kernel <<< blocks, threads >>> (g, *globs.d_globs);
	//cudaDeviceSynchronize();
	report_cuda_error("Two\n");
}

void rollback1_host (unsigned int g, PrivGlobs &globs) {
	dim3 blocks = dim3(ceil((globs.outer+0.f)/32),ceil((globs.numY+0.f)/32));
	dim3 threads = dim3(32,32);
	rollback1_kernel <<< blocks, threads >>> (g, *globs.d_globs);
	//cudaDeviceSynchronize();
	report_cuda_error("Two\n");
}

void rollback2_host (unsigned int g, PrivGlobs &globs) {
	dim3 blocks = dim3(ceil((globs.outer+0.f)/32),ceil((globs.numY+0.f)/32));
	dim3 threads = dim3(32,32);
	rollback2_kernel <<< blocks, threads >>> (g, *globs.d_globs);
	//cudaDeviceSynchronize();
	report_cuda_error("Two\n");
}

void rollback3_host (unsigned int g, PrivGlobs &globs) {
	dim3 blocks = dim3(ceil((globs.outer+0.f)/32),ceil((globs.numX+0.f)/32));
	dim3 threads = dim3(32,32);
	rollback3_kernel <<< blocks, threads >>> (g, *globs.d_globs);
	//cudaDeviceSynchronize();
	report_cuda_error("Two\n");
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
	TIMER_INIT(updateParams);
	TIMER_INIT(rollback);
	TIMER_INIT(rollback_0);
	TIMER_INIT(rollback_1);
	TIMER_INIT(rollback_2);
	TIMER_INIT(rollback_3);
	
	PrivGlobs globs;
	globs.init(numX, numY, numT, outer);
	
	initGrid(s0,alpha,nu,t, numX, numY, numT, globs);
	initOperator(globs.myX, globs.myDxx, globs.numX);
	initOperator(globs.myY, globs.myDyy, globs.numY);
	
	globs.cuda_init();
	report_cuda_error("Init\n");
	reportMemoryUsage();
	
	globs.copyToDevice();
	report_cuda_error("CopyToDevice\n");
	setPayoff_host(globs);
	
	TIMER_START(run_OrigCPU);
	int count = 0;
	for(int t = globs.numT-2; t>=0; --t) {
		//printf("%d / %d\n", count++, globs.numT-2);
		updateParams_host(t,alpha,beta,nu,globs);

		TIMER_START(rollback);

		// explicit x
		TIMER_START(rollback_0);
		rollback0_host(t, globs);
		TIMER_STOP(rollback_0);

		// explicit y
		TIMER_START(rollback_1);
		rollback1_host(t, globs);
		TIMER_STOP(rollback_1);

		// implicit x
		TIMER_START(rollback_2);
		rollback2_host(t, globs);
		TIMER_STOP(rollback_2);

		// implicit y
		TIMER_START(rollback_3);
		rollback3_host(t, globs);
		TIMER_STOP(rollback_3);

		TIMER_STOP(rollback);
		//cudaDeviceSynchronize();
	}
	TIMER_STOP(run_OrigCPU);

	// arrange and copy back data
	REAL *d_res;
	cudaMalloc(&d_res, sizeof(REAL) * outer);
	copyResult_kernel <<< 1, globs.outer >>> (d_res, *globs.d_globs);
	cudaMemcpy(res, d_res, sizeof(REAL) * outer, cudaMemcpyDeviceToHost);

	globs.free();
	
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
