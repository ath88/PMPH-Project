#ifndef PROJ_HELPER_FUNS
#define PROJ_HELPER_FUNS

//#include <vector>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include "Constants.h"

using namespace std;


struct PrivGlobs {
	
	int numX;
	int numY;
	int numT;
	
	// grid
	REAL *myX; // [numX]
	REAL *myY; // [numY]
	REAL *myTimeline; // [numT]
	unsigned myXindex;
	unsigned myYindex;
	
	// variable
	REAL *myResult; // [numX][numY]
	
	// coeffs
	REAL *myVarX; // [numX][numY]
	REAL *myVarY; // [numX][numY]
	
	// operators
	//vector<vector<REAL> > myDxx;  // [numX][4]
	REAL *myDxx; // [numX][4]
	REAL *myDyy; // [numY][4]
	//vector<vector<REAL> > myDyy;  // [numY][4]
	
	// vectors for rollback()
	REAL *u; // [numY][numX]
	REAL *v; // [numX][numY]
	REAL *a; // [max(numX,numY)]
	REAL *b; // [max(numX,numY)]
	REAL *c; // [max(numX,numY)]
	REAL *y; // [max(numX,numY)]
	REAL *yy; // [max(numX,numY)]
	
	
	PrivGlobs() {
		printf("Invalid Contructor: need to provide the array sizes! EXITING...!\n");
		exit(0);
	}
	
	//void init(
	PrivGlobs(
			const unsigned int &numX,
			const unsigned int &numY,
			const unsigned int &numT) {
		this->numX = numX;
		this->numY = numY;
		this->numT = numT;
		
		this->myX = (REAL *) malloc(sizeof(REAL) * numX);
		this->myY = (REAL *) malloc(sizeof(REAL) * numY);
		
		this->myDxx = (REAL *) malloc(sizeof(REAL) * numX * 4);
		this->myDyy = (REAL *) malloc(sizeof(REAL) * numY * 4);
				
		this->myTimeline = (REAL *) malloc(sizeof(REAL) * numT);
		
		this->myResult = (REAL *) malloc(sizeof(REAL) * numX * numY);
		this->myVarX = (REAL *) malloc(sizeof(REAL) * numX * numY);
		this->myVarY = (REAL *) malloc(sizeof(REAL) * numX * numY);
		
		
		this->u = (REAL *) malloc(sizeof(REAL) * numY * numX);
		this->v = (REAL *) malloc(sizeof(REAL) * numX * numY);
		
		this->a = (REAL *) malloc(sizeof(REAL) * (numX > numY ? numX : numY));
		this->b = (REAL *) malloc(sizeof(REAL) * (numX > numY ? numX : numY));
		this->c = (REAL *) malloc(sizeof(REAL) * (numX > numY ? numX : numY));
		this->y = (REAL *) malloc(sizeof(REAL) * (numX > numY ? numX : numY));
		this->yy = (REAL *) malloc(sizeof(REAL) * (numX > numY ? numX : numY));
	}
} __attribute__ ((aligned (128)));


void initGrid(const REAL s0, const REAL alpha, const REAL nu,const REAL t,
		const unsigned numX, const unsigned numY, const unsigned numT, PrivGlobs& globs);

//void initOperator(const vector<REAL> &x, vector<vector<REAL> > &Dxx);
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
