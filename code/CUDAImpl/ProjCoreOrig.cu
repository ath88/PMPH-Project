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

__device__ inline void d_tridag_2(const REAL*, const REAL*, const REAL*, const REAL*, const int, REAL*, REAL*, int, int, int);
__device__ inline void d_tridag_3(const REAL*, const REAL*, const REAL*, const REAL*, const int, REAL*, REAL*, int, int, int);

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
	REAL *a = globs.a + o * numY * numY;
	REAL *b = globs.b + o * numY * numY;
	REAL *c = globs.c + o * numY * numY;
	
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
	REAL *b = globs.b
			+ o * numY * numY;
	REAL *c = globs.c
			+ o * numY * numY;
	REAL *yy = globs.yy
			+ o * numY * numY;
	
	d_tridag_2(a,b,c,u,numX,u,yy,j,numX,numY);
}

__global__ void rollback3_0_kernel(unsigned int g, PrivGlobs &globs) {
	unsigned numX = globs.numX;
	unsigned numY = globs.numY;
	
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int o = threadIdx.y + blockDim.y * blockIdx.y;
	
	extern __shared__ REAL shared_myDyy[];//[numX * 4];
	int idx = threadIdx.x + blockDim.x * threadIdx.y;
	if(idx < numY) {
		shared_myDyy[idx*4] = globs.myDyy[idx*4];
		shared_myDyy[idx*4 + 1] = globs.myDyy[idx*4 + 1];
		shared_myDyy[idx*4 + 2] = globs.myDyy[idx*4 + 2];
		//shared_myDxx[idx*4 + 3] = globs.myDxx[idx*4 + 3];
	}
	
	if (i >= globs.numX) return;
	if (o >= globs.outer) return;
	
	REAL dtInv = 1.0 / (globs.myTimeline[g+1] - globs.myTimeline[g]);
	
	REAL *a = globs.a
			+ o * numY * numY;
	REAL *b = globs.b
			+ o * numY * numY;
	REAL *c = globs.c
			+ o * numY * numY;
	
	for(int j=0; j<numY; j++) { // here a, b, c should have size [numY]
	
		REAL myVarY = globs.myVarY[i*globs.numY + j];
		a[j*numX + i] = -0.5 * (0.5 * myVarY
				* shared_myDyy[j*4 + 0]);
		b[j*numX + i] = dtInv - 0.5 * (0.5 * myVarY
				* shared_myDyy[j*4 + 1]);
		c[j*numX + i] = -0.5 * (0.5 * myVarY
				* shared_myDyy[j*4 + 2]);
	}
}

__global__ void rollback3_1_kernel(unsigned int g, PrivGlobs &globs) {
	unsigned numX = globs.numX;
	unsigned numY = globs.numY;

	unsigned int j = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int o = threadIdx.y + blockDim.y * blockIdx.y;
	if (j >= globs.numY) return;
	if (o >= globs.outer) return;

	REAL dtInv = 1.0 / (globs.myTimeline[g+1] - globs.myTimeline[g]);

	REAL *u = globs.u + o * numY * numY; // [outer][numY][numX]
	REAL *v = globs.v + o * numY * numX; // [outer][numY][numX]
	REAL *y = globs.y + o * numY * numY;
	
	for(int i=0; i<numX; i++) {
		y[i*numY + j] = dtInv*u[i*numY + j] - 0.5*v[i*numY + j];
	}
}
__global__ void rollback3_tridag_kernel(unsigned int g, PrivGlobs &globs) {
	unsigned numX = globs.numX;
	unsigned numY = globs.numY;
	
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int o = threadIdx.y + blockDim.y * blockIdx.y;
	if (i >= globs.numX) return;
	if (o >= globs.outer) return;
	
	REAL *myResult = globs.myResult_trans + o * numY * numX;
	REAL *a = globs.a + o * numY * numY;
	REAL *b = globs.b + o * numY * numY;
	REAL *c = globs.c + o * numY * numY;
	REAL *y = globs.y_trans + o * numY * numY;
	REAL *yy = globs.yy + o * numY * numY;
	
	d_tridag_3(a,b,c,y,numY,myResult,yy,i,numX,numY);
}
__global__ void transpose_kernel(
		REAL *result, REAL *input, int width, int height, int outer) {
	__shared__ REAL shared_tile[TILE_DIMENSION * TILE_DIMENSION];
	
	const unsigned int o = blockIdx.z;
	if(o >= outer) {
		return;
	}
	const unsigned int outer_offset = o * width * height;
	
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
				input[outer_offset + tile_offset + offset_in_tile];
	}
	__syncthreads();
	if(j < height && i < width) {
		result[outer_offset
				+ tile_offset_transposed + offset_in_tile_transposed] =
				shared_tile[id_in_tile];
	}
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

__device__ inline void d_tridag_2(
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
	
	u[0 + j] = r[0 + j];
	uu[j] = b[0];
	
	int this_index = numY + j;
	for(i=1; i<n; i++) {
		beta  = a[this_index] / uu[this_index - numY];
		
		uu[this_index] = b[this_index] - beta*c[this_index - numY];
		u[this_index]  = r[this_index] - beta*u[this_index - numY];
		
		this_index += numY;
	}
	
	this_index -= numY;
	u[this_index] = u[this_index] / uu[this_index];
	for(i=n-2; i>=0; i--) {
		this_index -= numY;
		u[this_index] = (u[this_index] - c[this_index]*u[this_index + numY]) / uu[this_index];
	}
	
	/*/ Hint: X) can be written smth like (once you make a non-constant)
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
	*/
}
__device__ inline void d_tridag_3(
	const REAL *a,   // size [n]
	const REAL *b,   // size [n]
	const REAL *c,   // size [n]
	const REAL *r,   // size [n] //TRANS
	const int n,
	REAL *u,   // size [n] //TRANS
	REAL *uu,   // size [n] temporary
	int i,
	int numX,
	int numY
) {
	REAL beta;
	
	u[i] = r[i];
	uu[i] = b[0];
	
	int this_index = numX + i; // j*numX + i
	for(int j=1; j<n; j++) {
		beta  = a[this_index] / uu[this_index - numX];
		
		uu[this_index] = b[this_index] - beta*c[this_index - numX];
		u[this_index]  = r[j*numY + i] - beta*u[this_index - numX];
		
		this_index += numX;
	}
	
	this_index -= numX;
	u[this_index] = u[this_index] / uu[this_index];
	for(int j=n-2; j>=0; j--) {
		this_index -= numX;
		u[this_index] = (u[this_index] - c[this_index]*u[this_index + numX]) / uu[this_index];
	}

	//REAL a_flip[256];
	//for(int j=0; j<n; j++) {
	//	a_flip[j] = u[(n-1-j)*numX + i];
	//}
	//a_flip[0] = a_flip[0] / uu[(n-1)*numX + i];
	//for(int j=1; j<n; j++) {
	//	a_flip[j] = (a_flip[j] - c[(n-1-j)*numX + i]*a_flip[j-1]) / uu[(n-1-j)*numX + i];
	//}
	//
	//for(int j=0; j<n; j++) {
	//	u[j*numX + i] = a_flip[n-1-j];
	//}
	
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
	dim3 threads = dim3(32,32);
	dim3 blocks = dim3(ceil((globs.numX+0.f)/32),ceil((globs.outer+0.f)/32));
	rollback3_0_kernel <<< blocks, threads, sizeof(REAL)*globs.numY*4 >>> (g, *globs.d_globs);
	blocks = dim3(ceil((globs.numY+0.f)/32),ceil((globs.outer+0.f)/32));
	rollback3_1_kernel <<< blocks, threads >>> (g, *globs.d_globs);
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
void transpose_before_tridag_3 (unsigned int g, PrivGlobs &globs) {
	dim3 blocks = dim3(
			ceil((float) globs.numY / TILE_DIMENSION),
			ceil((float) globs.numY / TILE_DIMENSION),
			ceil((float) globs.outer)
			);
	dim3 threads = dim3(TILE_DIMENSION, TILE_DIMENSION, 1);
	transpose_kernel <<< blocks, threads >>>
				(globs.device->y_trans,
				globs.device->y,
				globs.numY, globs.numY, globs.outer);
	
	cudaDeviceSynchronize();
	report_cuda_error("transpose");
}
void transpose_end (unsigned int g, PrivGlobs &globs) {
	dim3 blocks = dim3(
			ceil((float) globs.numX / TILE_DIMENSION),
			ceil((float) globs.numY / TILE_DIMENSION),
			ceil((float) globs.outer)
			);
	dim3 threads = dim3(TILE_DIMENSION, TILE_DIMENSION);
	transpose_kernel <<< blocks, threads >>>
				(globs.device->myResult,
				globs.device->myResult_trans,
				globs.numX, globs.numY, globs.outer);
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
		
		TIMER_START(rollback_2_tridag);
		rollback2_tridag_host(t, globs);
		TIMER_STOP(rollback_2_tridag);
		
		//transpose_u_host(t, globs);
		//transpose_u_back_host(t, globs);
		
		// implicit y
		TIMER_START(rollback_3);
		rollback3_host(t, globs);
		TIMER_STOP(rollback_3);
		
		TIMER_START(transpose);
		transpose_before_tridag_3(t, globs);
		TIMER_STOP(transpose);
		
		TIMER_START(rollback_3_tridag);
		rollback3_tridag_host(t, globs);
		TIMER_STOP(rollback_3_tridag);
		
		TIMER_START(transpose);
		transpose_end(t, globs);
		TIMER_STOP(transpose);
		
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
	TIMER_REPORT(rollback_2_tridag);
	TIMER_REPORT(rollback_3);
	TIMER_REPORT(rollback_3_tridag);
	TIMER_REPORT(transpose);
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
