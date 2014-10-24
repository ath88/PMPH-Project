#include "ProjHelperFun.h"
#include "Constants.h"
#include "timers.h"

TIMER_DEFINE(run_OrigCPU);
	TIMER_DEFINE(init);
	TIMER_DEFINE(setPayoff);
	TIMER_DEFINE(updateParams);
	TIMER_DEFINE(rollback);
		TIMER_DEFINE(rollback_0);
		TIMER_DEFINE(rollback_1);
		TIMER_DEFINE(rollback_2);
			TIMER_DEFINE(tridag_0);
		TIMER_DEFINE(rollback_3);
			TIMER_DEFINE(tridag_1);

void updateParams(const unsigned g, const REAL alpha, const REAL beta, const REAL nu, PrivGlobs& globs) {
	TIMER_START(updateParams);
	
	for(unsigned i=0;i<globs.myX.size();++i) {
		for(unsigned j=0;j<globs.myY.size();++j) {
			globs.myVarX[i][j] = exp(2.0 * (
					beta*log(globs.myX[i])
					+ globs.myY[j]
					- 0.5*nu*nu*globs.myTimeline[g]));
			globs.myVarY[i][j] = exp(2.0 * (
					alpha*log(globs.myX[i])
					+ globs.myY[j]
					- 0.5*nu*nu*globs.myTimeline[g])); // nu*nu
		}
	}
	
	TIMER_STOP(updateParams);
}

void setPayoff(const REAL strike, PrivGlobs& globs) {
	TIMER_START(setPayoff);

	for(unsigned i = 0; i < globs.myX.size(); ++i) {
		REAL payoff = max(globs.myX[i] - strike, (REAL)0.0);
		for(unsigned j = 0; j < globs.myY.size(); ++j) {
			globs.myResult[i][j] = payoff;
		}
	}

	TIMER_STOP(setPayoff);
}

inline void tridag(
	const vector<REAL> &a,   // size [n]
	const vector<REAL> &b,   // size [n]
	const vector<REAL> &c,   // size [n]
	const vector<REAL> &r,   // size [n]
	const int n,
	vector<REAL> &u,   // size [n]
	vector<REAL> &uu   // size [n] temporary
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
	
	unsigned numX = globs.myX.size();
	unsigned numY = globs.myY.size();
	
	unsigned numZ = max(numX,numY);
	
	unsigned i, j;
	
	REAL dtInv = 1.0 / (globs.myTimeline[g+1] - globs.myTimeline[g]);
	
	vector<vector<REAL> > u(numY, vector<REAL>(numX));   // [numY][numX]
	vector<vector<REAL> > v(numX, vector<REAL>(numY));   // [numX][numY]
	vector<REAL> a(numZ), b(numZ), c(numZ), y(numZ);	 // [max(numX,numY)] 
	vector<REAL> yy(numZ);  // temporary used in tridag  // [max(numX,numY)]
	
	//	explicit x
	TIMER_START(rollback_0);
	for(i=0; i<numX; i++) {
		for(j=0; j<numY; j++) {
			u[j][i] = dtInv*globs.myResult[i][j];
			
			if(i > 0) { 
				u[j][i] += 0.5 * (0.5 * globs.myVarX[i][j] * globs.myDxx[i][0])
						* globs.myResult[i-1][j];
			}
			u[j][i] += 0.5 * (0.5 * globs.myVarX[i][j] * globs.myDxx[i][1])
					* globs.myResult[i][j];
			if(i < numX - 1) {
				u[j][i] += 0.5 * (0.5 * globs.myVarX[i][j] * globs.myDxx[i][2])
						* globs.myResult[i+1][j];
			}
		}
	}
	TIMER_STOP(rollback_0);
	
	//	explicit y
	TIMER_START(rollback_1);
	for(j=0; j<numY; j++) {
		for(i=0; i<numX; i++) {
			v[i][j] = 0.0;
			
			if(j > 0) {
				v[i][j] += (0.5 * globs.myVarY[i][j] * globs.myDyy[j][0])
						* globs.myResult[i][j-1];
			}
			v[i][j] += (0.5 * globs.myVarY[i][j] * globs.myDyy[j][1])
					* globs.myResult[i][j];
			if(j < numY - 1) {
				v[i][j] += (0.5 * globs.myVarY[i][j] * globs.myDyy[j][2])
						* globs.myResult[i][j+1];
			}
			u[j][i] += v[i][j];
		}
	}
	TIMER_STOP(rollback_1);
	
	// implicit x
	TIMER_START(rollback_2);
	for(j=0; j<numY; j++) {
		for(i=0; i<numX; i++) { // here a, b,c should have size [numX]
			a[i] = -0.5 * (0.5 * globs.myVarX[i][j] * globs.myDxx[i][0]);
			b[i] = dtInv - 0.5 * (0.5 * globs.myVarX[i][j] * globs.myDxx[i][1]);
			c[i] = -0.5 * (0.5 * globs.myVarX[i][j] * globs.myDxx[i][2]);
		}
		// here yy should have size [numX]
		TIMER_START(tridag_0);
		tridag(a,b,c,u[j],numX,u[j],yy);
		TIMER_STOP(tridag_0);
	}
	TIMER_STOP(rollback_2);
	
	// implicit y
	TIMER_START(rollback_3);
	for(i=0; i<numX; i++) { 
		for(j=0; j<numY; j++) { // here a, b, c should have size [numY]
			a[j] =  - 0.5*(0.5*globs.myVarY[i][j]*globs.myDyy[j][0]);
			b[j] = dtInv - 0.5*(0.5*globs.myVarY[i][j]*globs.myDyy[j][1]);
			c[j] = - 0.5*(0.5*globs.myVarY[i][j]*globs.myDyy[j][2]);
		}
		
		for(j=0; j<numY; j++) {
			y[j] = dtInv*u[j][i] - 0.5*v[i][j];
		}
		
		// here yy should have size [numY]
		TIMER_START(tridag_1);
		tridag(a,b,c,y,numY,globs.myResult[i],yy);
		TIMER_STOP(tridag_1);
	}
	TIMER_STOP(rollback_3);
	
	TIMER_STOP(rollback);
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
	TIMER_START(init);
	initGrid(s0,alpha,nu,t, numX, numY, numT, globs);
	initOperator(globs.myX,globs.myDxx);
	initOperator(globs.myY,globs.myDyy);
	TIMER_STOP(init);

	TIMER_START(setPayoff);
	setPayoff(strike, globs);
	TIMER_STOP(setPayoff);
	
	for(int i = globs.myTimeline.size()-2; i>=0; --i) {
		updateParams(i,alpha,beta,nu,globs);
		rollback(i, globs);
	}
	
	
	return globs.myResult[globs.myXindex][globs.myYindex];
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
	TIMER_INIT(init);
	TIMER_INIT(setPayoff);
	TIMER_INIT(updateParams);
	TIMER_INIT(rollback);
	TIMER_INIT(rollback_0);
	TIMER_INIT(rollback_1);
	TIMER_INIT(rollback_2);
	TIMER_INIT(tridag_0);
	TIMER_INIT(rollback_3);
	TIMER_INIT(tridag_1);
	
	REAL strike;
	PrivGlobs globs(numX, numY, numT);
	
	
	TIMER_START(run_OrigCPU);
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
	TIMER_REPORT(init);
	TIMER_REPORT(setPayoff);
	TIMER_REPORT(updateParams);
	TIMER_REPORT(rollback);
	TIMER_GROUP();
	TIMER_REPORT(rollback_0);
	TIMER_REPORT(rollback_1);
	TIMER_REPORT(rollback_2);
	TIMER_GROUP();
	TIMER_REPORT(tridag_0);
	TIMER_GROUP_END();
	TIMER_REPORT(rollback_3);
	TIMER_GROUP();
	TIMER_REPORT(tridag_1);
	TIMER_GROUP_END();
	TIMER_GROUP_END();
	TIMER_GROUP_END();
}

//#endif // PROJ_CORE_ORIG
