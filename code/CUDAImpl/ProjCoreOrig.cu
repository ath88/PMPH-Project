#include "ProjHelperFun.h"
#include "Constants.h"
#include "timers.h"

#include <cuda_runtime.h>
#include <cuda.h>

#define TILE_DIMENSION 32

TIMER_DEFINE(run_OrigCPU);
	TIMER_DEFINE(updateParams);
	TIMER_DEFINE(rollback);
		TIMER_DEFINE(rollback_0);
		TIMER_DEFINE(rollback_1);
		TIMER_DEFINE(rollback_2);
		TIMER_DEFINE(transpose);
		TIMER_DEFINE(rollback_2_tridag);
		TIMER_DEFINE(rollback_3);
		TIMER_DEFINE(rollback_3_tridag);

void report_cuda_error(char*);

__device__ inline void d_tridag(const REAL*, const REAL*, const REAL*, const REAL*, const int, REAL*, REAL*);
__device__ inline void d_tridag_trans(const REAL*, const REAL*, const REAL*, const REAL*, const int, REAL*, REAL*, int, int);
__device__ inline void d_tridag_2_trans(const REAL*, const REAL*, const REAL*, const REAL*, const int, REAL*, REAL*, int, int, int);
__device__ inline void d_tridag_trans_u(const REAL*, const REAL*, const REAL*, const REAL*, const int, REAL*, REAL*, int, int, int);


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
	unsigned int o = threadIdx.y + blockDim.y * blockIdx.y;
	unsigned int j = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned numX = globs.numX;
	unsigned numY = globs.numY;

	if (o >= globs.outer) return;
	if (j >= globs.numY) return;

	REAL *myVarX = globs.myVarX + o * numX * numY;
	REAL *myVarY = globs.myVarY + o * numX * numY;

	REAL constant_addition = 2.0
			* (globs.myY[j] - 0.5*nu*nu*globs.myTimeline[g]);
	for(unsigned i=0;i<globs.numX;++i) {
		REAL constant_multiplication = 2.0 * log(globs.myX[i]);
		myVarX[i*globs.numY + j] = exp(
				beta * constant_multiplication
				+ constant_addition);
		myVarY[i*globs.numY + j] = exp(
				alpha * constant_multiplication
				+ constant_addition); // nu*nu
	} 
}

__global__ void rollback0_kernel(unsigned int g, PrivGlobs &globs) {
	unsigned numX = globs.numX;
	unsigned numY = globs.numY;
	
	unsigned int j = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int o = threadIdx.y + blockDim.y * blockIdx.y;
	
	extern __shared__ REAL shared_myDxx[];//[numX * 4];
	int idx = threadIdx.x + blockDim.x * threadIdx.y;
	if(idx < numX) {
		shared_myDxx[idx*4] = globs.myDxx[idx*4];
		shared_myDxx[idx*4 + 1] = globs.myDxx[idx*4 + 1];
		shared_myDxx[idx*4 + 2] = globs.myDxx[idx*4 + 2];
		shared_myDxx[idx*4 + 3] = globs.myDxx[idx*4 + 3];
	}
	
	if (j >= globs.numY) return;
	if (o >= globs.outer) return;
	
	REAL dtInv = 1.0 / (globs.myTimeline[g+1] - globs.myTimeline[g]);
	
	REAL *myResult = globs.myResult + o * numY * numX;
	REAL *u = globs.u + o * numY * numY; // [outer][numY][numX]
	//REAL *u = globs.u_trans + o * numY * numY; // [outer][numY][numX]
	__syncthreads();
	
	int this_index = j;
	for(int i=0; i<numX; i++) {
		
		REAL this_u;
		REAL this_myVarX;
		
		this_u = dtInv*myResult[this_index];
		
		this_myVarX = globs.myVarX[this_index];
		
		if(i > 0) {
			this_u += 0.5 * (0.5
					* this_myVarX * shared_myDxx[i*4 + 0])
					* myResult[this_index - globs.numY];
		}
		this_u += 0.5 * (0.5
				* this_myVarX * shared_myDxx[i*4 + 1])
				* myResult[this_index];
		if(i < numX - 1) {
			this_u += 0.5 * (0.5
					* this_myVarX * shared_myDxx[i*4 + 2])
					* myResult[this_index + globs.numY];
		}
		u[i*numY + j] = this_u;
		
		this_index += globs.numY;
	}
}

__global__ void rollback1_kernel(unsigned int g, PrivGlobs &globs) {
	unsigned numX = globs.numX;
	unsigned numY = globs.numY;
	
	unsigned int j = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int o = threadIdx.y + blockDim.y * blockIdx.y;
	if (j >= globs.numY) return;
	if (o >= globs.outer) return;
	
	REAL *myResult = globs.myResult + o * numY * numX;
	REAL *u = globs.u + o * numY * numY; // [outer][numY][numX]
	REAL *v = globs.v + o * numY * numX; // [outer][numY][numX]
	
	REAL this_myDyy[3];
	this_myDyy[0] = globs.myDyy[j*4 + 0];
	this_myDyy[1] = globs.myDyy[j*4 + 1];
	this_myDyy[2] = globs.myDyy[j*4 + 2];
	
	int this_index = j;
	for(int i=0; i<numX; i++) {
		v[i*numY + j] = 0.0;
		REAL myVarY = globs.myVarY[this_index];
		REAL myResultSub[3] = {
				myResult[this_index - 1],
				myResult[this_index],
				myResult[this_index + 1],
		};
		
		if(j > 0) {
			v[i*numY + j] += (0.5
					* myVarY * this_myDyy[0])
					* myResultSub[0];
		}
		v[i*numY + j] += (0.5
				* myVarY * this_myDyy[1])
				* myResultSub[1];
		if(j < numY - 1) {
			v[i*numY + j] += (0.5
					* myVarY * this_myDyy[2])
					* myResultSub[2];
		}
		u[i*numY + j] += v[i*numY + j];
		
		this_index += numY;
	}
}

__global__ void rollback2_kernel(unsigned int g, PrivGlobs &globs) {
	unsigned numX = globs.numX;
	unsigned numY = globs.numY;
	
	unsigned int j = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int o = threadIdx.y + blockDim.y * blockIdx.y;
	
	extern __shared__ REAL shared_myDxx[];//[numX * 4];
	int idx = threadIdx.x + blockDim.x * threadIdx.y;
	if(idx < numX) {
		shared_myDxx[idx*4] = globs.myDxx[idx*4];
		shared_myDxx[idx*4 + 1] = globs.myDxx[idx*4 + 1];
		shared_myDxx[idx*4 + 2] = globs.myDxx[idx*4 + 2];
		//shared_myDxx[idx*4 + 3] = globs.myDxx[idx*4 + 3];
	}
	
	if (j >= globs.numY) return;
	if (o >= globs.outer) return;
	
	REAL dtInv = 1.0 / (globs.myTimeline[g+1] - globs.myTimeline[g]);
	
	//REAL *u = globs.u + o * numY * numY; // [outer][numY][numX]
	REAL *a = globs.a
			+ o * numY * numY;
			//+ j * numY; // [outer][y][max(numX,numY)]
	REAL *b = globs.b
			+ o * numY * numY;
			//+ j  * numY; // [outer][y][max(numX,numY)]
	REAL *c = globs.c
			+ o * numY * numY;
			//+ j * numY; // [outer][y][max(numX,numY)]
	
	//int this_index = j;
	for(int i=0; i<numX; i++) { // here a, b,c should have size [numX]
		REAL myVarX = globs.myVarX[i*globs.numY + j];
		a[i*numY + j] = -0.5 * (0.5 * myVarX
				* shared_myDxx[i*4 + 0]);
		b[i*numY + j] = dtInv - 0.5 * (0.5 * myVarX
				* shared_myDxx[i*4 + 1]);
		c[i*numY + j] = -0.5 * (0.5 * myVarX
				* shared_myDxx[i*4 + 2]);
		//this_index += numY;
	}
}
__global__ void rollback2_tridag_kernel(unsigned int g, PrivGlobs &globs) {
	unsigned numX = globs.numX;
	unsigned numY = globs.numY;
	
	unsigned int j = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int o = threadIdx.y + blockDim.y * blockIdx.y;
	if (j >= globs.numY) return;
	if (o >= globs.outer) return;
	
	REAL *u = globs.u + o * numY * numY; // [outer][numY][numX]
	REAL *a = globs.a
			+ o * numY * numY;
			//+ j * numY; // [outer][y][max(numX,numY)]
	REAL *b = globs.b
			+ o * numY * numY;
			//+ j  * numY; // [outer][y][max(numX,numY)]
	REAL *c = globs.c
			+ o * numY * numY;
			//+ j * numY; // [outer][y][max(numX,numY)]
	REAL *yy = globs.yy
			+ o * numY * numY
			+ j * numY; // [outer][y][max(numX,numY)]
	
	d_tridag_2_trans(a,b,c,u + j,numX,u + j,yy,j,numX,numY);
}

__global__ void rollback3_kernel(unsigned int g, PrivGlobs &globs) {
	unsigned numX = globs.numX;
	unsigned numY = globs.numY;
	
	unsigned int j = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int o = threadIdx.y + blockDim.y * blockIdx.y;
	if (j >= globs.numY) return;
	if (o >= globs.outer) return;
	
	REAL dtInv = 1.0 / (globs.myTimeline[g+1] - globs.myTimeline[g]);
	
	REAL *u = globs.u + o * numY * numY; // [outer][numY][numX]
	REAL *v = globs.v + o * numY * numX; // [outer][numY][numX]
	REAL *a = globs.a
			+ o * numY * numY;
	REAL *b = globs.b
			+ o * numY * numY;
	REAL *c = globs.c
			+ o * numY * numY;
	REAL *y = globs.y
			+ o * numY * numY;
	//REAL *y_trans = globs.y_trans
	//		+ o * numY * numY;
	
	REAL this_myDyy[3];
	this_myDyy[0] = globs.myDyy[j*4 + 0];
	this_myDyy[1] = globs.myDyy[j*4 + 1];
	this_myDyy[2] = globs.myDyy[j*4 + 2];
	
	for(int i=0; i<numX; i++) { // here a, b, c should have size [numY]
	
		REAL myVarY = globs.myVarY[i*globs.numY + j];
		a[i*numY + j] = -0.5 * (0.5 * myVarY
				* this_myDyy[0]);
		b[i*numY + j] = dtInv - 0.5 * (0.5 * myVarY
				* this_myDyy[1]);
		c[i*numY + j] = -0.5 * (0.5 * myVarY
				* this_myDyy[2]);
	}
	
	//for(int j=0; j<numY; j++) {
	for(int i=0; i<numX; i++) {
		//y_trans[j*numY + i] = dtInv*u[j*numX + i] - 0.5*v[i*numY + j];
		//y[j] = dtInv*u[j*numX + i] - 0.5*v[i*numY + j];
		y[i*numY + j] = dtInv*u[i*numY + j] - 0.5*v[i*numY + j];
	}
}
__global__ void rollback3_tridag_kernel(unsigned int g, PrivGlobs &globs) {
	unsigned numX = globs.numX;
	unsigned numY = globs.numY;
	
	//unsigned int rollback_i = threadIdx.y + blockDim.y * blockIdx.y;
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
	//unsigned int j = threadIdx.x + blockDim.x * blockIdx.x;
	//unsigned int tridag_i = i % numY;
	//unsigned int rollback_i = i;// / numY;
	//unsigned int rollback_i = i;
	unsigned int o = threadIdx.y + blockDim.y * blockIdx.y;
	if (i >= globs.numX) return;
	if (o >= globs.outer) return;
	
	REAL *myResult = globs.myResult + o * numY * numX;
	REAL *a = globs.a
			+ o * numY * numY
			+ i * numY;
	REAL *b = globs.b
			+ o * numY * numY
			+ i * numY;
	REAL *c = globs.c
			+ o * numY * numY
			+ i * numY;
	REAL *y = globs.y
			+ o * numY * numY
			+ i * numY;
	//REAL *y_trans = globs.y_trans
	//		+ o * numY * numY;
	REAL *yy = globs.yy
			+ o * numY * numY
			+ i * numY;
	//REAL *yy_trans = globs.yy_trans
	//		+ o * numY * numY;
	
	// here yy should have size [numY]
	d_tridag(a,b,c,y,numY,myResult + i*globs.numY,yy);
	//d_tridag_2_trans(a,b,c,y,numY,myResult + i*numY,yy,i,numX,numY);
	//d_tridag_trans(a_trans,b_trans,c_trans,y,numY,myResult + i*globs.numY,yy,i,numY);
}

__global__ void optimized_transpose_kernel(
		REAL *result, REAL *input, int width, int height) {
	__shared__ REAL shared_tile[TILE_DIMENSION * TILE_DIMENSION];
	
	const unsigned int tile_i =
			blockIdx.x;
	const unsigned int tile_j =
			blockIdx.y;
	const unsigned int tile_offset =
			tile_j * width * TILE_DIMENSION + tile_i * TILE_DIMENSION;
	const unsigned int tile_offset_transposed =
			tile_i * height * TILE_DIMENSION + tile_j * TILE_DIMENSION;
	
	const unsigned int i_in_tile =
			threadIdx.y;
	const unsigned int j_in_tile =
			threadIdx.x;
	const unsigned int id_in_tile =
			j_in_tile * TILE_DIMENSION + i_in_tile;
	const unsigned int id_in_tile_transposed =
			i_in_tile * TILE_DIMENSION + j_in_tile;
	const unsigned int offset_in_tile =
			i_in_tile * width + j_in_tile;
	const unsigned int offset_in_tile_transposed =
			i_in_tile * height + j_in_tile;
	
	const unsigned int i = tile_i * TILE_DIMENSION + i_in_tile;
	const unsigned int j = tile_j * TILE_DIMENSION + j_in_tile;
	
	const unsigned int i_tileflipped = tile_j * TILE_DIMENSION + i_in_tile;
	const unsigned int j_tileflipped = tile_i * TILE_DIMENSION + j_in_tile;
	
	if(i_tileflipped < height && j_tileflipped < width) {
		shared_tile[id_in_tile_transposed] =
				input[tile_offset + offset_in_tile];
	}
	__syncthreads();
	if(j < height && i < width) {
		result[tile_offset_transposed + offset_in_tile_transposed] =
				shared_tile[id_in_tile];
	}
}
__global__ void transpose_kernel(unsigned int g, PrivGlobs &globs) {
	
}

void updateParams_host(const unsigned g, const REAL alpha, const REAL beta, const REAL nu, PrivGlobs& globs) {
	TIMER_START(updateParams);
	dim3 blocks = dim3(ceil((globs.numY+0.f)/32),ceil((globs.outer+0.f)/32));
	dim3 threads = dim3(32,32);
	updateParams_kernel <<< blocks, threads >>> (g, alpha, beta, nu, *globs.d_globs);
	cudaDeviceSynchronize();
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
	for(i=0; i<n; i++) {
		a[i] = u[n-1-i];
	}
	a[0] = a[0] / uu[n-1];
	for(i=1; i<n; i++) {
		a[i] = (a[i] - c[n-1-i]*a[i-1]) / uu[n-1-i];
	}
	for(i=0; i<n; i++) {
		u[i] = a[n-1-i];
	}
#endif
}
__device__ inline void d_tridag_trans(
	const REAL *a,   // size [n]
	const REAL *b,   // size [n]
	const REAL *c,   // size [n]
	const REAL *r,   // size [n]
	const int n,
	REAL *u,   // size [n]
	REAL *uu,   // size [n] temporary
	int j,
	int numY
) {
	int i;
	REAL beta;
	
	u[0] = r[0];
	uu[0] = b[0];
	
	for(i=1; i<n; i++) {
		//beta  = a[i*numY + j] / uu[(i-1)*numY + j];
		beta  = a[i*numY + j] / uu[i-1];
		
		//uu[i*numY + j] = b[i*numY + j] - beta*c[(i-1)*numY + j];
		uu[i] = b[i*numY + j] - beta*c[(i-1)*numY + j];
		u[i]  = r[i] - beta*u[i-1];
	}

	//u[n-1] = u[n-1] / uu[(i-1)*numY + j];
	u[n-1] = u[n-1] / uu[i-1];
	for(i=n-2; i>=0; i--) {
		//u[i] = (u[i] - c[i*numY + j]*u[i+1]) / uu[i*numY + j];
		u[i] = (u[i] - c[i*numY + j]*u[i+1]) / uu[i];
	}
}
__device__ inline void d_tridag_2_trans(
	const REAL *a,   // size [n]
	const REAL *b,   // size [n]
	const REAL *c,   // size [n]
	const REAL *r,   // size [n] //TRANS
	const int n,
	REAL *u,   // size [n] //TRANS
	REAL *uu,   // size [n] temporary
	int j,
	int numX,
	int numY
) {
	int i;
	REAL beta;
	
	u[0] = r[0];
	uu[0] = b[0];
	
	for(i=1; i<n; i++) {
		//beta  = a[i*numY + j] / uu[(i-1)*numY + j];
		beta  = a[i*numY + j] / uu[i-1];
		
		//uu[i*numY + j] = b[i*numY + j] - beta*c[(i-1)*numY + j];
		uu[i] = b[i*numY + j] - beta*c[(i-1)*numY + j];
		//u[i]  = r[i] - beta*u[i-1];
		u[i*numY]  = r[i*numY] - beta*u[(i-1)*numY];
	}

	//u[n-1] = u[n-1] / uu[(i-1)*numY + j];
	//u[n-1] = u[n-1] / uu[i-1];
	u[(n-1)*numY] = u[(n-1)*numY] / uu[i-1];
	for(i=n-2; i>=0; i--) {
		//u[i] = (u[i] - c[i*numY + j]*u[i+1]) / uu[i*numY + j];
		u[i*numY] = (u[i*numY] - c[i*numY + j]*u[(i+1)*numY]) / uu[i];
	}
}
__device__ inline void d_tridag_trans_u(
	const REAL *a,   // size [n]
	const REAL *b,   // size [n]
	const REAL *c,   // size [n]
	const REAL *r,   // size [n]
	const int n,
	REAL *u,   // size [n]
	REAL *uu,   // size [n] temporary
	int j,
	int numX,
	int numY
) {
	int i;
	REAL beta;
	
	u[0] = r[0];
	uu[0] = b[0];
	
	int u_span = numY;
	for(i=1; i<n; i++) {
		//beta  = a[i*numY + j] / uu[(i-1)*numY + j];
		beta  = a[i*numY + j] / uu[i-1];
		
		//uu[i*numY + j] = b[i*numY + j] - beta*c[(i-1)*numY + j];
		uu[i] = b[i*numY + j] - beta*c[(i-1)*numY + j];
		u[i*u_span + j]  = r[i*u_span] - beta*u[(i-1)*u_span + j];
	}

	//u[n-1] = u[n-1] / uu[(i-1)*numY + j];
	u[(n-1)*u_span + j] = u[(n-1)*u_span + j] / uu[i-1];
	for(i=n-2; i>=0; i--) {
		//u[i] = (u[i] - c[i*numY + j]*u[i+1]) / uu[i*numY + j];
		u[i*u_span + j] = (u[i*u_span + j] - c[i*numY + j]*u[(i+1)*u_span + j]) / uu[i];
	}
}

void report_cuda_error(char* id) {
	cudaError err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("CUDA error at id \"%s\": %s\n", id, cudaGetErrorString(err));
		exit(1);
	}
}

#define ROLLBACK_BLOCK_SIZE 32
void rollback0_host (unsigned int g, PrivGlobs &globs) {
	//dim3 blocks = dim3(ceil((globs.outer+0.f)/32),ceil((globs.numY+0.f)/32));
	dim3 blocks = dim3(
			ceil((float) globs.numY / ROLLBACK_BLOCK_SIZE),
			ceil((float) globs.outer / ROLLBACK_BLOCK_SIZE));
	dim3 threads = dim3(ROLLBACK_BLOCK_SIZE, ROLLBACK_BLOCK_SIZE);
	rollback0_kernel <<< blocks, threads, sizeof(REAL)*globs.numY*4 >>>
			(g, *globs.d_globs);
	cudaDeviceSynchronize();
	report_cuda_error("rollback0");
}

#define BLOCK_SIZE 32
void rollback1_host (unsigned int g, PrivGlobs &globs) {
	dim3 blocks = dim3(
			ceil((float) globs.numY / ROLLBACK_BLOCK_SIZE),
			ceil((float) globs.outer / ROLLBACK_BLOCK_SIZE));
	dim3 threads = dim3(ROLLBACK_BLOCK_SIZE, ROLLBACK_BLOCK_SIZE);
	rollback1_kernel <<< blocks, threads, sizeof(REAL)*globs.numY*4 >>>
			(g, *globs.d_globs);
	cudaDeviceSynchronize();
	report_cuda_error("rollback1");
}

void rollback2_host (unsigned int g, PrivGlobs &globs) {
	dim3 blocks = dim3(ceil((globs.numY+0.f)/32),ceil((globs.outer+0.f)/32));
	dim3 threads = dim3(32,32);
	rollback2_kernel <<< blocks, threads, sizeof(REAL)*globs.numY*4 >>> (g, *globs.d_globs);
	cudaDeviceSynchronize();
	report_cuda_error("rollback2");
}
void rollback2_tridag_host (unsigned int g, PrivGlobs &globs) {
	dim3 blocks = dim3(ceil((globs.numY+0.f)/32),ceil((globs.outer+0.f)/32));
	dim3 threads = dim3(32,32);
	rollback2_tridag_kernel <<< blocks, threads >>> (g, *globs.d_globs);
	cudaDeviceSynchronize();
	report_cuda_error("rollback2_tridag");
}

void rollback3_host (unsigned int g, PrivGlobs &globs) {
	dim3 blocks = dim3(ceil((globs.numY+0.f)/32),ceil((globs.outer+0.f)/32));
	dim3 threads = dim3(32,32);
	rollback3_kernel <<< blocks, threads >>> (g, *globs.d_globs);
	cudaDeviceSynchronize();
	report_cuda_error("rollback3");
}
void rollback3_tridag_host (unsigned int g, PrivGlobs &globs) {
	dim3 blocks = dim3(
			ceil((float) globs.numX / 32),
			ceil((float) globs.outer / 32)
			//ceil((float) globs.numY*globs.numY / 32)
			);
	dim3 threads = dim3(32, 32);
	rollback3_tridag_kernel <<< blocks, threads >>> (g, *globs.d_globs);
	cudaDeviceSynchronize();
	report_cuda_error("rollback3_tridag");
}
void transpose_host (unsigned int g, PrivGlobs &globs) {
	dim3 blocks = dim3(
			ceil((float) globs.numY / TILE_DIMENSION),
			ceil((float) globs.numY / TILE_DIMENSION)
			);
	dim3 threads = dim3(TILE_DIMENSION, TILE_DIMENSION);
	for(int o = 0; o < globs.outer; o += 1) {
		int offset = o * globs.numY * globs.numX;
		optimized_transpose_kernel <<< blocks, threads >>>
				(globs.device->u_trans + offset, globs.device->u + offset,
				globs.numY, globs.numX);
		
		offset = o * globs.numY * globs.numY;
		optimized_transpose_kernel <<< blocks, threads >>>
				(globs.device->a_trans + offset, globs.device->a + offset,
				globs.numY, globs.numY);
		optimized_transpose_kernel <<< blocks, threads >>>
				(globs.device->b_trans + offset, globs.device->b + offset,
				globs.numY, globs.numY);
		optimized_transpose_kernel <<< blocks, threads >>>
				(globs.device->c_trans + offset, globs.device->c + offset,
				globs.numY, globs.numY);
		optimized_transpose_kernel <<< blocks, threads >>>
				(globs.device->y_trans + offset, globs.device->y + offset,
				globs.numY, globs.numY);
		optimized_transpose_kernel <<< blocks, threads >>>
				(globs.device->yy_trans + offset, globs.device->yy + offset,
				globs.numY, globs.numY);
	}
	cudaDeviceSynchronize();
	report_cuda_error("transpose");
}

void transpose_u_host (unsigned int g, PrivGlobs &globs) {
	dim3 blocks = dim3(
			ceil((float) globs.numY / TILE_DIMENSION),
			ceil((float) globs.numY / TILE_DIMENSION)
			);
	dim3 threads = dim3(TILE_DIMENSION, TILE_DIMENSION);
	for(int o = 0; o < globs.outer; o += 1) {
		int offset = o * globs.numY * globs.numY;
		optimized_transpose_kernel <<< blocks, threads >>>
				(globs.device->u_trans + offset, globs.device->u + offset,
				globs.numY, globs.numY);
	}
}
void transpose_u_back_host (unsigned int g, PrivGlobs &globs) {
	dim3 blocks = dim3(
			ceil((float) globs.numY / TILE_DIMENSION),
			ceil((float) globs.numY / TILE_DIMENSION)
			);
	dim3 threads = dim3(TILE_DIMENSION, TILE_DIMENSION);
	for(int o = 0; o < globs.outer; o += 1) {
		int offset = o * globs.numY * globs.numY;
		optimized_transpose_kernel <<< blocks, threads >>>
				(globs.device->u + offset, globs.device->u_trans + offset,
				globs.numY, globs.numY);
	}
	cudaDeviceSynchronize();
	report_cuda_error("transpose");
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
	TIMER_INIT(transpose);
	TIMER_INIT(rollback_2_tridag);
	TIMER_INIT(rollback_3);
	TIMER_INIT(rollback_3_tridag);
	
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
	for(int t = globs.numT-2; t>=0; --t) {
		//printf("%d / %d\n", count++, globs.numT-2);
		updateParams_host(t,alpha,beta,nu,globs);
		
		TIMER_START(rollback);
		
		
		//transpose_u_host(t, globs);
		// explicit x
		TIMER_START(rollback_0);
		rollback0_host(t, globs);
		TIMER_STOP(rollback_0);
		//transpose_u_back_host(t, globs);
		
		//transpose_u_host(t, globs);
		//transpose_u_back_host(t, globs);
		
		// explicit y
		TIMER_START(rollback_1);
		rollback1_host(t, globs);
		TIMER_STOP(rollback_1);

		// implicit x
		TIMER_START(rollback_2);
		rollback2_host(t, globs);
		TIMER_STOP(rollback_2);
		
		//TIMER_START(transpose);
		//transpose_host(t, globs);
		//TIMER_STOP(transpose);
		
		TIMER_START(rollback_2_tridag);
		rollback2_tridag_host(t, globs);
		TIMER_STOP(rollback_2_tridag);
		
		//transpose_u_host(t, globs);
		//transpose_u_back_host(t, globs);
		
		// implicit y
		TIMER_START(rollback_3);
		rollback3_host(t, globs);
		TIMER_STOP(rollback_3);
		TIMER_START(rollback_3_tridag);
		rollback3_tridag_host(t, globs);
		TIMER_STOP(rollback_3_tridag);

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
	TIMER_REPORT(transpose);
	TIMER_REPORT(rollback_2_tridag);
	TIMER_REPORT(rollback_3);
	TIMER_REPORT(rollback_3_tridag);
	TIMER_GROUP_END();
	TIMER_GROUP_END();
	
	int serial_time = 0;
	if(outer == 16) {
		serial_time = 2000000;
	} else if(outer == 32) {
		serial_time = 4200000;
	} else if(outer == 128) {
		serial_time = 183374932;
	}
	if(serial_time != 0) {
		printf("Speedup vs serial: %f\n",
				(float) serial_time / TIMER_MU_S(run_OrigCPU));
	}
}

//#endif // PROJ_CORE_ORIG
