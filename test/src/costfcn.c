#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <stdbool.h>
#include <specialFunctions.h>
#include "costfcn.h"
#include "updates.h"

/* This file contains the cost function computations for varDP
 * The computations are split into the negative variational entropies,
 * the likelihood cross entropy, and the prior cross entropies. 
 * At the bottom of the file is the only exposed function in this .c file, 
 * double varBayesCost(...) -- it returns the variational upper bound on the KL divergence 
 * to the posterior. See that function for descriptions of the inputs to all the functions
 * in this file.
 */
double varLabelEntropy(const double * const zeta, const uint32_t N, const uint32_t K){
	double ent = 0.0;
	uint32_t i;
	for (i = 0; i < N*K; i++){
		ent += (zeta[i] > 1.0e-16 ? zeta[i]*log(zeta[i]) : 0.0);
	}
	return ent;
}

double varBetaEntropy(
		const double * const a, 
		const double * const b, 
		const uint32_t K){
	double ent = 0.0;
	uint32_t k;
	for (k = 0; k < K; k++){
    ent+= -gsl_sf_lnbeta(a[k], b[k]) + (a[k]-1.0)*gsl_sf_psi(a[k])
      +(b[k]-1.0)*gsl_sf_psi(b[k])-(a[k]+b[k]-2.0)*gsl_sf_psi(a[k]+b[k]);
	}
	return ent;
}

double varExpEntropy(
		const double * const logh, 
		const double * const eta, 
		const double * const nu, 
		const double * const dlogh_deta, 
		const double * const dlogh_dnu, 
		const uint32_t M, 
		const uint32_t K){
	double ent = 0.0;
	uint32_t k, j;
	for (k = 0; k < K; k++){
		ent += logh[k] - nu[k]*dlogh_dnu[k];
	}
	for (j = 0; j < M*K; j++){
		ent -= eta[j]*dlogh_deta[j];
	}

	return ent;
}

double likeCrossEntropy(
		const double * const sumzeta,
		const double * const sumzetaT,
		const double * const dlogh_deta,
		const double * const dlogh_dnu,
		const uint32_t M,
		const uint32_t K){
	double ent = 0.0;
	uint32_t k, j; 
	for (k = 0; k < K; k++){
		ent -= sumzeta[k]*dlogh_dnu[k];
	}
	for (j = 0; j < K*M; j++){
		ent -= sumzetaT[j]*dlogh_deta[j];
	}
	return ent;
}

double priorLabelCrossEntropy(
		const double * const a,
		const double * const b,
		const double * const sumzeta,
		const uint32_t K){
	double ent = 0.0;
	uint32_t k;
	double psibk = 0.0;
	for (k = 0; k < K; k++){
		double psiak = gsl_sf_psi(a[k]) - gsl_sf_psi(a[k]+b[k]);
		ent += sumzeta[k]*(psiak + psibk);
		psibk += gsl_sf_psi(b[k]) - gsl_sf_psi(a[k]+b[k]);
	}
	return ent;
}

double priorExpCrossEntropy(
		const double logh0,
		const double * const eta0,
		const double nu0,
		const double * const dlogh_deta,
		const double * const dlogh_dnu,
		const uint32_t M,
		const uint32_t K){

	double ent = K*logh0;
	uint32_t k, j;
	for (k = 0; k < K; k++){
		ent -= nu0*dlogh_dnu[k];
	}
	for (j=0; j < M*K; j++){
		ent -= eta0[j%M]*dlogh_deta[j];
	}
	return ent;
}

double priorBetaCrossEntropy(
		const double * const a,
		const double * const b,
		const double alpha,
		const uint32_t K){
	double ent = -K*gsl_sf_lnbeta(1.0, alpha);
	uint32_t k;
	for (k = 0; k < K; k++){
		ent += (alpha-1.0)*(gsl_sf_psi(b[k]) - gsl_sf_psi(a[k]+b[k]));
	}
	return ent;
}

/* This is the objective computation function
 * Inputs:
 * zeta -- label distribution for all data
 * sumzeta, sumzetaT -- the sum of categorical weights, and the sum of the weights*data sufficient stats
 * a, b -- the stick-breaking beta weights
 * eta, nu -- the natural parameters for the parameter exponential families
 * eta0, nu0 -- the prior natural parameters
 * logh, logh0 -- the base measure of each parameter exponential family distribution, and the prior one
 * dlogh_deta, dlogh_dnu -- the derivatives of the log base function
 * alpha -- the DP concentration parameter
 * M -- the dimension of the sufficient stat/natural parameters
 * K -- the number of clusters
 *
 * Output: A single double value for the overall variational KL upper bound
 */
double varBayesCost(
		const double * const zeta,
		const double * const sumzeta,
		const double * const sumzetaT,
		const double * const a,
		const double * const b,
		const double * const eta,
		const double * const eta0,
		const double * const nu,
		const double nu0,
		const double * const logh,
		const double logh0,
		const double * const dlogh_deta,
		const double * const dlogh_dnu,
		const double alpha,
		const uint32_t N,
		const uint32_t M,
		const uint32_t K){

//  printf("varLabelEntropy = %f\n",varLabelEntropy(zeta, N, K));
//  printf("varBetaEntropy = %f\n",varBetaEntropy(a, b, K));
//  printf("varExpEntropy = %f\n",varExpEntropy(logh, eta, nu, dlogh_deta, dlogh_dnu, M, K));
//  printf("likeCrossEntropy = %f\n",-likeCrossEntropy(sumzeta, sumzetaT, dlogh_deta, dlogh_dnu, M, K));
//  printf("priorExpCrossEntropy = %f\n",- priorExpCrossEntropy(logh0, eta0, nu0, dlogh_deta, dlogh_dnu, M, K));
//  printf("priorLabelCrossEntropy = %f\n",- priorLabelCrossEntropy(a, b, sumzeta, K));
//  printf("priorBetaCrossEntropy = %f\n",- priorBetaCrossEntropy(a, b, alpha, K));

    double cost = varBetaEntropy(a, b, K)
			+ varExpEntropy(logh, eta, nu, dlogh_deta, dlogh_dnu, M, K)
			- likeCrossEntropy(sumzeta, sumzetaT, dlogh_deta, dlogh_dnu, M, K)
			- priorExpCrossEntropy(logh0, eta0, nu0, dlogh_deta, dlogh_dnu, M, K)
			- priorLabelCrossEntropy(a, b, sumzeta, K)
			- priorBetaCrossEntropy(a, b, alpha, K);
    if(zeta != NULL)
      cost += varLabelEntropy(zeta, N, K);
    return cost;
}

/*
 * compute the combined varBayesCost after a merge of clusters from the
 * parameters of the merged cluster and the combined label entropy (sum over
 * all labelEntropies of the involved clusters)
 */
double varBayesCostAfterMerge(
		const double combinedLabelEntropy,
		const double * const sumzeta,
		const double * const sumzetaT,
		const double * const eta,
		const double * const eta0,
		const double * const nu,
		const double nu0,
		const double alpha,
		void (*getLogH)(double*, double* const, double* const, const double * const, const double, const uint32_t, bool),
		const uint32_t D,
		const uint32_t M,
		const uint32_t K)
{
  double *a = (double*) malloc(sizeof(double)*K);
  double *b = (double*) malloc(sizeof(double)*K);
  double *logh = (double*) malloc(sizeof(double)*K);
  double *dlogh_dnu = (double*) malloc(sizeof(double)*K);
  double *dlogh_deta = (double*) malloc(sizeof(double)*K*M);
//    VectorXd a(K); VectorXd b(K);
//    VectorXd logh(K); VectorXd dlogh_dnu(K);
//    VectorXd dlogh_deta(K*M);
    updateWeightDist(a, b, NULL, sumzeta, alpha, K);
    double logh0 = 0.0; 
  	getLogH(&logh0, NULL, NULL, eta0, nu0, D, false);
    uint32_t k = 0;
    for(k=0; k<K; ++k)
  	  getLogH(logh+k, dlogh_deta+M*k, dlogh_dnu+k, eta+M*k, nu[k], D, true);
    double elbo = varBayesCost(NULL, sumzeta, sumzetaT,
        a, b, eta, eta0, nu, nu0,
        logh, logh0,
        dlogh_deta, dlogh_dnu,
        alpha, 0, M, K) + combinedLabelEntropy; 
//    for(uint32_t i=0; i< NThreads; ++i)
//      elbo += labelEntropy[i];
  free(a); free(b); free(logh); free(dlogh_dnu); free(dlogh_deta);
    return elbo;
}

