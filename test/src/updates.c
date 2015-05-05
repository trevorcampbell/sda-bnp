#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <gsl/gsl_sf_psi.h>
#include <gsl/gsl_sf_gamma.h>
#include "updates.h"

void updateLabelDist(double* const zeta,
		const double * const stat, 
		const double * const dlogh_deta, 
		const double * const dlogh_dnu, 
		const double * const psisum, 
		const uint32_t M, 
		const uint32_t K){

	uint32_t j, k;

	/*update the label distribution*/
	/*compute the log of the weights, storing the maximum so far*/
	double logpmax = -INFINITY;
	for (k = 0; k < K; k++){
		zeta[k] = psisum[k] - dlogh_dnu[k];
		for (j = 0; j < M; j++){
			zeta[k] -= stat[j]*dlogh_deta[k*M+j];
		}
		logpmax = (zeta[k] > logpmax ? zeta[k] : logpmax);
	}
	/*make numerically stable by subtracting max, take exp, sum them up*/
	double psum = 0.0;
	for (k = 0; k < K; k++){
		zeta[k] -= logpmax;
		zeta[k] = exp(zeta[k]);
		psum += zeta[k];
	}
	/*normalize*/
	for (k = 0; k < K; k++){
		zeta[k] /= psum;
	}

	return;
}

void updateParamDist(double* const eta, double* const nu, 
		const double* const eta0, 
		const double nu0, 
		const double sumzeta, 
		const double* const sumzetaT, 
		const uint32_t M){

	uint32_t j;
	/*Update the parameters*/
	/*Note that eta[i*M+j] is indexing for a 2D array (K clusters, M components) stored as 1D*/
	for (j = 0; j < M; j++){
		eta[j] = eta0[j]+sumzetaT[j];
	}
	*nu = nu0 + sumzeta;
	return;
}

void updateWeightDist(double* a, double* b, double* psisum,
		const double* const sumzeta, 
		const double alpha, 
		const uint32_t K){
	uint32_t k, j;
	/*Update a, b, and psisum*/
	double psibk = 0.0;
	for (k = 0; k < K; k++){
		a[k] = 1.0+sumzeta[k];
		b[k] = alpha;
		for (j = k+1; j < K; j++){
			b[k] += sumzeta[j];
		}
    if(psisum!=NULL) 
    {
      double psiak = gsl_sf_psi(a[k]) - gsl_sf_psi(a[k]+b[k]);
      psisum[k] = psiak + psibk;
      psibk += gsl_sf_psi(b[k]) - gsl_sf_psi(a[k]+b[k]);
    }
	}

	return;
}


void computeRho(double* const r, 
		const double* const w,
		const double* const stat, 
		const double* const eta, 
		const double* const nu, 
		const double* const eta0, 
		const double nu0, 
		double * const etatmp,
		void (*getLogH)(double*, double* const, double* const, const double * const, const double, const uint32_t, bool),
		const double alpha, 
		const uint32_t M,
		const uint32_t D,
		const uint32_t K){
	uint32_t k, j;
	double maxr = -INFINITY;
	double logh0, logh1;
	for(k = 0; k< K; k++){
		getLogH(&logh0, NULL, NULL, &(eta[M*k]), nu[k], D, false);
		for(j=0; j < M; j++){
			etatmp[j] = eta[M*k+j] + stat[j];
		}
		getLogH(&logh1, NULL, NULL, etatmp, nu[k]+1.0, D, false);
		r[k] = log(w[k]) + logh1-logh0;
		if(r[k] > maxr){
			maxr=r[k];
		}
	}
	getLogH(&logh0, NULL, NULL, eta0, nu0, D, false);
	for(j=0; j < M; j++){
		etatmp[j] = eta0[j] + stat[j];
	}
	getLogH(&logh1, NULL, NULL, etatmp, nu0+1.0, D, false);
	r[K] = log(alpha) + logh1 - logh0;
	if(r[K]>maxr){
		maxr=r[K];
	}
	double rsum = 0.0;
	for(k=0; k < K+1; k++){
		r[k] -= maxr;
		r[k] = exp(r[k]);
		rsum += r[k];
	}
	//printf("r: ");
	for(k=0; k < K+1; k++){
		r[k] /= rsum;
		//printf("%f ", r[k]);
	}
	//printf("\n");
	return;
}




