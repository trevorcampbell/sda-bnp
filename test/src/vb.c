#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#include "../include/costfcn.h"
#include "../include/updates.h"
#include "../include/vb.h"
#include <gsl/gsl_sf_psi.h>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_rng.h>

double computeTestLogLikelihood(
		const double* const T, 
		double* eta, double* nu, double* a, double* b, 
		double* (*getLogPostPred)(const double* const, const double* const, const double* const, const uint32_t, const uint32_t, const uint32_t),
		const uint32_t Nt, 
		const uint32_t D,
		const uint32_t M,
		uint32_t K);
		
/* Private function in this c file (not exposed in h) */
void initializeZeta(double * const, double * const, double * const, const double * const, 
		void (*getStat)(double*, const double* const, const uint32_t), const uint32_t, const uint32_t, const uint32_t, const uint32_t);


//void varDP_init(double* zeta, double* sumzeta, double* sumzetaT, double* logh0, double *prevobj,
//		const double* const T, 
//		const double* const eta0, 
//		const double nu0,
//		void (*getLogH)(double*, double* const, double* const, const double * const, const double, const uint32_t, bool),
//		void (*getStat)(double*, const double* const, const uint32_t),
//		double* (*getLogPostPred)(const double* const, const double* const, const double* const, const uint32_t, const uint32_t, const uint32_t),
//		const uint32_t N, 
//		const uint32_t M,
//		const uint32_t D,
//		uint32_t K, uint32_t id){
//
//	/*Initialize zeta randomly for K clusters*/
//	initializeZeta(zeta, sumzeta, sumzetaT, T, getStat, N, M, D, K);
//
//	/*Get the prior logh*/
//	getLogH(logh0, NULL, NULL, eta0, nu0, D, false);
//
//	/*Proceed to the VB iteration*/
//	*prevobj = INFINITY;
//
//}
//
//double varDP_iteration(double* zeta, double* sumzeta, double* sumzetaT, double * logh0,
//    double* eta, double* nu, double* a, double* b, unsigned int* out_K,
//		const double* const T, 
//    double* prevobj,
//		const double alpha, 
//		const double* const eta0, 
//		const double nu0,
//		void (*getLogH)(double*, double* const, double* const, const double * const, const double, const uint32_t, bool),
//		void (*getStat)(double*, const double* const, const uint32_t),
//		double* (*getLogPostPred)(const double* const, const double* const, const double* const, const uint32_t, const uint32_t, const uint32_t),
//		const uint32_t N, 
//		const uint32_t M,
//		const uint32_t D,
//		uint32_t K,uint32_t id){
//
//	double* dlogh_deta = (double*) malloc(sizeof(double)*K*M); /*Stores d(logh)/d(eta) for each cluster*/
//	double* dlogh_dnu = (double*) malloc(sizeof(double)*K);/*Stores d(logh)/d(nu) for each cluster*/
//	double* logh = (double*) malloc(sizeof(double)*K);/*Stores logh for each cluster*/
//	double* stat = (double*) malloc(sizeof(double)*M);
//	double* psisum = (double*) malloc(sizeof(double)*K);
//
////  double tol = 1e-8;
//  double diff = 1.0;
//  int i,j,k;
//
//		/*Update the stick-breaking distribution and psisum*/
//		updateWeightDist(a, b, psisum, sumzeta, alpha, K);
//	
//		/*Update the exponential family parameter distributions*/
//		/*Compute the values of logh, derivatives*/
//		for (k=0; k< K; k++){
//			updateParamDist(&(eta[k*M]), &(nu[k]), eta0, nu0, sumzeta[k], &(sumzetaT[k*M]), M);
//			getLogH(&(logh[k]), &(dlogh_deta[M*k]), &(dlogh_dnu[k]), &(eta[M*k]), nu[k], D, true);
//		}
//
//		/*Initialize sumzeta to 0*/
//		for(k = 0; k < K; k++){
//			sumzeta[k] = 0.0;
//			for (j = 0; j < M; j++){
//				sumzetaT[k*M+j] = 0.0;
//			}
//		}
//
//		/*Update the label distributions*/
//		for (i=0; i < N; i++){
//			/*Compute the statistic for datapoint*/
//			getStat(stat, &(T[i*D]), D);
//			/*Get the new label distribution*/
//			updateLabelDist(&(zeta[i*K]), stat, dlogh_deta, dlogh_dnu, psisum, M,  K);
//			/*Update sumzeta*/
//			for(k = 0; k < K; k++){
//				sumzeta[k] += zeta[i*K+k];
//				for (j = 0; j < M; j++){
//					sumzetaT[k*M+j] += zeta[i*K+k]*stat[j];
//				}
//			}
//		}
//
//		/*Compute the variational objective*/
//    double obj= varBayesCost(zeta, sumzeta, sumzetaT, a, b, eta, eta0, nu, nu0,
//        logh, *logh0, dlogh_deta, dlogh_dnu, alpha, N, M, K);
//		diff = fabs( (obj- (*prevobj))/obj);
//		*prevobj = obj;
//	  *out_K = K;
////		printf("@id %d: Obj %f\tdelta %f\n", id,obj,diff);
//
//    free(dlogh_deta); free(dlogh_dnu); free(stat); free(psisum);
//    return diff;
//}

/* This function runs batch variational inference for DP mixtures of exponential families
 *
 * Inputs:
 * T (size N*D) -- the flattened array of data
 *      i.e. T[i*D+j] is the jth component of datapoint i
 * alpha -- DP concentration parameter
 * eta0 (size M) -- the M-dimensional natural parameter for the exponential family parameter prior
 * nu0 -- the prior weight
 * void (*getLogH)() -- a function pointer to a function that computes the base measure and log derivatives for an exponential family
 * 						see vbfuncs_gaussian.c for a description of the requirements of this function
 * void (*getStat)() -- a function pointer to a function that computes the sufficient statistic for a datapoint
 * 						see vbfuncs_gaussian.c for a description of the requirements of this function
 * N -- number of datapoints (major steps in T)
 * D -- dimension of the data(minor steps in T)
 * M -- dimension of suff stat 
 * K -- the initial number of clusters.
 *
 * Outputs: To get the results out of this function, pass in pointer addresses to all out_X arguments (   e.g. double* res_zeta; varDP(&res_zeta, ...);  )
 * out_zeta (size N*K) -- flattened label distributions for all datapoints
 * out_eta (size K*M) -- natural parameters for all clusters;  each consecutive block of size M is for a different cluster
 * out_nu (size K) -- distribution strengths for all clusters
 * out_a (size K) --stick breaking beta distribution params (1st param)
 * out_b (size K) --stick breaking beta distribution params (2nd param)
 * out_K -- final number of clusters
 *
 * In LaTeX notation:  
 * The prior on parameters is P(\theta) = h(\eta_0, \nu_0)\exp\left( \phi(\theta)^T\eta_0 - \nu_0A(\phi(\theta))\right)
 * The data likelihood is P(y) = f(y)\exp\left(\phi(\theta)^T T(y) - A(\phi(\theta))\right)
 * A( ) is the log-partition function
 * h, f are base measures
 */
double varDP_noAlloc(double* zeta, double* eta, double* nu, double* a, double* b, unsigned int* out_K,
		double* times, double* testlls, uint32_t* out_nTrace,
		const double* const T, 
		const double* const Ttest, 
		const double alpha, 
		const double* const eta0, 
		const double nu0,
		void (*getLogH)(double*, double* const, double* const, const double * const, const double, const uint32_t, bool),
		void (*getStat)(double*, const double* const, const uint32_t),
		double* (*getLogPostPred)(const double* const, const double* const, const double* const, const uint32_t, const uint32_t, const uint32_t),
		const uint32_t N, const uint32_t Nt, 
		const uint32_t M,
		const uint32_t D,
		uint32_t K,uint32_t id)
{

//  printf("@id=%d:varDP_noAlloc N=%d M=%d D=%d K=%d \n",id,N,M,D,K);
	double* sumzeta = (double*) malloc(sizeof(double)*K); /*Stores sum over all data(zeta) for each cluster*/
	double* sumzetaT = (double*) malloc(sizeof(double)*K*M); /*Stores sum over all data(zeta*T) for each cluster*/

  double obj = varDP_noAllocSumZeta(zeta, sumzeta, sumzetaT, eta, nu, a, b, out_K, times, testlls, out_nTrace, T, Ttest,
    alpha, eta0, nu0, getLogH, getStat, getLogPostPred, N, Nt, M, D, K,id);

	free(sumzeta); free(sumzetaT);
  return obj;
}

double varDP_noAllocSumZeta(double* zeta, double* sumzeta, double* sumzetaT,
    double* eta, double* nu, double* a, double* b, unsigned int* out_K,
		double* times, double* testlls, uint32_t* out_nTrace,
		const double* const T, 
		const double* const Ttest, 
		const double alpha, 
		const double* const eta0, 
		const double nu0,
		void (*getLogH)(double*, double* const, double* const, const double * const, const double, const uint32_t, bool),
		void (*getStat)(double*, const double* const, const uint32_t),
		double* (*getLogPostPred)(const double* const, const double* const, const double* const, const uint32_t, const uint32_t, const uint32_t),
		const uint32_t N, const uint32_t Nt, 
		const uint32_t M,
		const uint32_t D,
		uint32_t K,uint32_t id)
{
	double* dlogh_deta = (double*) malloc(sizeof(double)*K*M); /*Stores d(logh)/d(eta) for each cluster*/
	double* dlogh_dnu = (double*) malloc(sizeof(double)*K);/*Stores d(logh)/d(nu) for each cluster*/
	double* logh = (double*) malloc(sizeof(double)*K);/*Stores logh for each cluster*/
	double logh0 = 0.0;
	double* stat = (double*) malloc(sizeof(double)*M);
	double* psisum = (double*) malloc(sizeof(double)*K);

	int i, j, k;

	*out_nTrace = 0;
	struct timespec ts, tf;
	clock_gettime(CLOCK_MONOTONIC, &ts);

	/*Initialize zeta randomly for K clusters*/
	initializeZeta(zeta, sumzeta, sumzetaT, T, getStat, N, M, D, K);


	/*Get the prior logh*/
	getLogH(&logh0, NULL, NULL, eta0, nu0, D, false);

	/*Proceed to the VB iteration*/
	double prevobj = INFINITY;
	double tol = 1e-8;
	double diff = 1.0;
	while(diff > tol){
		/*Update the stick-breaking distribution and psisum*/
		updateWeightDist(a, b, psisum, sumzeta, alpha, K);
	
		/*Update the exponential family parameter distributions*/
		/*Compute the values of logh, derivatives*/
		for (k=0; k< K; k++){
			updateParamDist(&(eta[k*M]), &(nu[k]), eta0, nu0, sumzeta[k], &(sumzetaT[k*M]), M);
			getLogH(&(logh[k]), &(dlogh_deta[M*k]), &(dlogh_dnu[k]), &(eta[M*k]), nu[k], D, true);
		}

		/*Initialize sumzeta to 0*/
		for(k = 0; k < K; k++){
			sumzeta[k] = 0.0;
			for (j = 0; j < M; j++){
				sumzetaT[k*M+j] = 0.0;
			}
		}

		/*Update the label distributions*/
		for (i=0; i < N; i++){
			/*Compute the statistic for datapoint*/
			getStat(stat, &(T[i*D]), D);
			/*Get the new label distribution*/
			updateLabelDist(&(zeta[i*K]), stat, dlogh_deta, dlogh_dnu, psisum, M,  K);
			/*Update sumzeta*/
			for(k = 0; k < K; k++){
				sumzeta[k] += zeta[i*K+k];
				for (j = 0; j < M; j++){
					sumzetaT[k*M+j] += zeta[i*K+k]*stat[j];
				}
			}
	}

		
		
		clock_gettime(CLOCK_MONOTONIC, &tf);
		if (*out_nTrace == 0){
			times[*out_nTrace] = (tf.tv_sec-ts.tv_sec) + (tf.tv_nsec - ts.tv_nsec)/1.0e9;
		} else {
			times[*out_nTrace] = times[*out_nTrace - 1] + (tf.tv_sec-ts.tv_sec) + (tf.tv_nsec - ts.tv_nsec)/1.0e9;
		}
		testlls[*out_nTrace] = computeTestLogLikelihood(Ttest, eta, nu, a, b, getLogPostPred, Nt, D, M, K);
		(*out_nTrace)++;
		/*Compute the variational objective*/
		double obj= varBayesCost(zeta, sumzeta, sumzetaT, a, b, eta, eta0, nu, nu0, logh, logh0, dlogh_deta, dlogh_dnu, alpha, N, M, K);
		diff = fabs( (obj-prevobj)/obj);
		prevobj = obj;
//		printf("@id %d: Obj %f\tdelta %f\n", id,obj,diff);
		clock_gettime(CLOCK_MONOTONIC, &ts);
	}

	//Remove empty clusters
	removeEmptyClustersX(zeta, sumzeta, sumzetaT, eta, nu, logh, dlogh_deta, dlogh_dnu, nu0, a, b, &K, N, M, K, false);
	double finalobj= varBayesCost(zeta, sumzeta, sumzetaT, a, b, eta, eta0, nu, nu0, logh, logh0, dlogh_deta, dlogh_dnu, alpha, N, M, K);
	*out_K = K;

	/*Free non-output memory*/
	free(dlogh_deta); free(dlogh_dnu); free(logh); free(stat);
  	free(psisum);

	return finalobj;
}

double varDP(double** out_zeta, double** out_eta, double** out_nu, double** out_a, double** out_b, uint32_t* out_K,
		double** out_times, double** out_testlls, uint32_t* out_nTrace,
		const double* const T, 
		const double* const Ttest, 
		const double alpha, 
		const double* const eta0, 
		const double nu0,
		void (*getLogH)(double*, double* const, double* const, const double * const, const double, const uint32_t, bool),
		void (*getStat)(double*, const double* const, const uint32_t),
		double* (*getLogPostPred)(const double* const, const double* const, const double* const, const uint32_t, const uint32_t, const uint32_t),
		const uint32_t N, const uint32_t Nt, 
		const uint32_t M,
		const uint32_t D,
		uint32_t K){

	double* zeta = (double*) malloc(sizeof(double)*N*K); /*Stores sum over all data(zeta) for each cluster*/
	double* eta = (double*) malloc(sizeof(double)*K*M); /*Stores d(logh)/d(eta) for each cluster*/
	double* nu = (double*) malloc(sizeof(double)*K); /*Stores d(logh)/d(eta) for each cluster*/
	double* a = (double*) malloc(sizeof(double)*K); /*Stores 1st beta parameters for weights*/
	double* b = (double*) malloc(sizeof(double)*K); /*Stores 2nd beta parameters for weights*/
	double* times = (double*) malloc(sizeof(double)*100000); /*Stores log times*/
	double* testlls = (double*) malloc(sizeof(double)*100000); /*Stores log testloglikes*/

  double obj = varDP_noAlloc(zeta,eta,nu,a,b,out_K,times,testlls,out_nTrace,T,Ttest,alpha,eta0,nu0,getLogH,getStat,getLogPostPred,N,Nt,M,D,K,0);

	/*output*/
	*out_zeta = zeta;
	*out_eta = eta;
	*out_nu = nu;
	*out_a = a;
	*out_b = b;
	*out_times = times;
	*out_testlls = testlls;
  return obj;
}

void removeEmptyClustersX(double* zeta, double* sumzeta, double* sumzetaT, double* eta, double* nu, 
			double* logh, double* dlogh_deta, double* dlogh_dnu, const double nu0, double* a, double* b, uint32_t* out_K,
		const uint32_t N,  const uint32_t M, const uint32_t K, bool compress){
	int kr = K, kl = -1;
	uint32_t j;
	while(kr > kl){
		//find first nonempty cluster from the right
		do{kr--;} while(kr > kl && nu[kr] - nu0 < 1.0);
	
		//find first empty cluster from the left
		do{kl++;} while(kl <= kr && nu[kl] - nu0 >= 1.0);

		if (kl < kr){
			//swap
			for(j=0; j < M; j++){
				eta[kl*M+j] = eta[kr*M+j];
				dlogh_deta[kl*M+j] = dlogh_deta[kr*M+j];
				sumzetaT[kl*M+j] = sumzetaT[kr*M+j];
			}
			nu[kl] = nu[kr];
			logh[kl] = logh[kr];
			dlogh_dnu[kl] = dlogh_dnu[kr];
			sumzeta[kl] = sumzeta[kr];
			a[kl] = a[kr];
			b[kl] = b[kr];
			for(j = 0; j < N; j++){
				zeta[j*K+kl] = zeta[j*K+kr];
			}
		}
	}
	(*out_K) = kl;
	//compress zeta
	if ( (*out_K) < K){
		for (j = 1; j < N; j++){ //j=0 is already in the right spot, and don't want to assign things to themselves
			for(kl=0; kl < (*out_K); kl++){
				zeta[j*(*out_K)+kl] = zeta[j*K+kl];
			}
		}
	}
#ifndef __CUDACC__
	if((*out_K) < K && compress){
		//compress memory for eta/nu/a/b
		eta = (double*) realloc(eta, sizeof(double)*M*(*out_K));
		nu = (double*) realloc(nu, sizeof(double)*(*out_K));
		a = (double*) realloc(a, sizeof(double)*(*out_K));
		b = (double*) realloc(b, sizeof(double)*(*out_K));
		zeta = (double*) realloc(zeta, sizeof(double)*N*(*out_K));
	}
#endif
	return;
}

void removeEmptyClusters(double* zeta, double* eta, double* nu, const double nu0, double* a, double* b, uint32_t* out_K,
		const uint32_t N,  const uint32_t M, const uint32_t K, bool compress){
	int kr = K, kl = -1;
	uint32_t j;
	while(kr > kl){
		//find first nonempty cluster from the right
		do{kr--;} while(nu[kr] - nu0 < 1.0 && kr > kl);
		//find first empty cluster from the left
		do{kl++;} while(nu[kl] - nu0 >= 1.0 && kl <= kr);

		if (kl < kr){
			//swap
			for(j=0; j < M; j++){
				eta[kl*M+j] = eta[kr*M+j];
			}
			nu[kl] = nu[kr];
			a[kl] = a[kr];
			b[kl] = b[kr];
			for(j = 0; j < N; j++){
				zeta[j*K+kl] = zeta[j*K+kr];
			}
		}
	}
	(*out_K) = kl;
	if( (*out_K) < K){
		//compress zeta
		for (j = 1; j < N; j++){ //j=0 is already in the right spot, and don't want to assign things to themselves
			for(kl=0; kl < (*out_K); kl++){
				zeta[j*(*out_K)+kl] = zeta[j*K+kl];
			}
		}
	}
#ifndef __CUDACC__
	if((*out_K) < K && compress){
		//compress memory for eta/nu/a/b
		eta = (double*) realloc(eta, sizeof(double)*M*(*out_K));
		nu = (double*) realloc(nu, sizeof(double)*(*out_K));
		a = (double*) realloc(a, sizeof(double)*(*out_K));
		b = (double*) realloc(b, sizeof(double)*(*out_K));
		zeta = (double*) realloc(zeta, sizeof(double)*N*(*out_K));
	}
#endif
	return;
}


void initializeZeta(double * const zeta, double * const sumzeta, double * const sumzetaT,
			const double * const T, 
			void (*getStat)(double*, const double* const, const uint32_t),
			const uint32_t N, 
			const uint32_t M,
			const uint32_t D,
			const uint32_t K){
	int i, j, k;
	double ratioNK = (double)N/(double)K;
	double * clusstats = (double*) malloc(sizeof(double)*M*K);
	double * stat = (double*) malloc(sizeof(double)*M);

	/*initialize sumzeta, sumzetaT to 0*/
	for (k = 0; k < K; k++){
		sumzeta[k] = 0.0;
		for(j = 0; j < M; j++){
			sumzetaT[k*M+j] = 0.0;
		}
	}

	/*initialize in the cluster stats*/
	for(k=0; k< K; k++){
		int idxk = floor(k*ratioNK);
		getStat(&(clusstats[M*k]), &(T[idxk*D]), D);
	}

	/*compute dist between cluster stats and data stats and use exp(-dist^2) as similarity*/
	for(i = 0; i < N; i++){
		getStat(stat, &(T[i*D]), D);
		double rwsum = 0;
    double minDistSq = INFINITY;
		for(k=0; k< K; k++){
			double distsq = 0.0;
			for(j=0; j<M; j++){
				distsq += (stat[j]-clusstats[k*M+j])*(stat[j]-clusstats[k*M+j]);
			}
			zeta[i*K+k] = distsq;
      if (minDistSq > distsq) minDistSq = distsq;
		}

		for(k=0; k < K; k++){
			zeta[i*K+k] = exp(-(zeta[i*K+k]-minDistSq));
			rwsum += zeta[i*K+k];
		}

		for(k=0; k < K; k++){
			zeta[i*K+k] /= rwsum;
			sumzeta[k] += zeta[i*K+k];
			for (j=0; j < M; j++){
				sumzetaT[k*M+j] += zeta[i*K+k]*stat[j];
			}
		}
	}

	free(clusstats);
	free(stat);
	return;
}


void convertSVAtoVB(double*const, double*const, double*const, double*const , double*const, double*const, double*const,  double*const,
				 const double* const, const double* const, const double* const, const double* const, 
					void (*getLogH)(double*, double* const, double* const, const double * const, const double, const uint32_t, bool),
					void (*getStat)(double*, const double* const, const uint32_t),
		double* (*getLogPostPred)(const double* const, const double* const, const double* const, const uint32_t, const uint32_t, const uint32_t),
							const double, const uint32_t,const uint32_t,const uint32_t, const uint32_t);

double svaDP(double** out_zeta, double** out_eta, double** out_nu, double** out_a, double** out_b, uint32_t* out_K,
		double** out_times, double** out_testlls, uint32_t* out_nTrace,
		const double* const T, 
		const double* const Ttest, 
		const double alpha, 
		const double epsclus,
		const double epspm,
		const double* const eta0, 
		const double nu0,
		void (*getLogH)(double*, double* const, double* const, const double * const, const double, const uint32_t, bool),
		void (*getStat)(double*, const double* const, const uint32_t),
		double* (*getLogPostPred)(const double* const, const double* const, const double* const, const uint32_t, const uint32_t, const uint32_t),
		const uint32_t N, const uint32_t Nt, 
		const uint32_t M,
		const uint32_t D,
		uint32_t K){

	int i, j, k, kk, m;
	uint32_t Ktmp;

	/*Initialize*/
	K = 1;
	double* times = (double*) malloc(sizeof(double)*2*N); /*stores trace times*/
	double* testlls = (double*) malloc(sizeof(double)*2*N); /*stores testll trace*/

	*out_nTrace = 0;
	struct timespec ts, tf;
	clock_gettime(CLOCK_MONOTONIC, &ts);


	double* eta = (double*) malloc(sizeof(double)*K*M); /*Stores the natural params*/
	double* nu = (double*) malloc(sizeof(double)*K); /*Stores the natural param strengths*/
	double* w = (double*) malloc(sizeof(double)*K); /*Stores cluster weights*/
	double* r = (double*) malloc(sizeof(double)*(K+1)); /*Stores normalized marginal log likelihoods*/
	double** rdiffsum = (double**) malloc(sizeof(double*)*K); /*Stores normalized marginal log likelihoods*/
	for(k=0; k <K; k++){
		rdiffsum[k] = (double*) malloc(sizeof(double*)*K);
	}
	double** ndiffsum = (double**) malloc(sizeof(double*)*K); /*Stores normalized marginal log likelihoods*/
	for(k=0; k <K; k++){
		ndiffsum[k] = (double*) malloc(sizeof(double*)*K);
	}
	double* stat = (double*) malloc(sizeof(double)*M);/*working memory to hold stats*/
	double* etatmp = (double*) malloc(sizeof(double)*M);/*working memory to hold stats*/
	/*Compute the statistic for the first datapoint*/
	getStat(stat, T, D);
	for(j=0; j < M; j++){
		eta[j] = eta0[j] + stat[j];
	}
	nu[0] = nu0 + 1.0;
	w[0] = 1.0;

	/*For storing conversion to VB from SVA*/
	double* zeta = (double*) malloc(sizeof(double)*N*K); /*Stores sum over all data(zeta) for each cluster*/
	double* sumzeta = (double*) malloc(sizeof(double)*K);
	double* sumzetaT = (double*) malloc(sizeof(double)*K*M);
	double* a = (double*) malloc(sizeof(double)*K); /*Stores 1st beta parameters for weights*/
	double* b = (double*) malloc(sizeof(double)*K); /*Stores 2nd beta parameters for weights*/
	double* dlogh_deta = (double*) malloc(sizeof(double)*K*M); /*Stores d(logh)/d(eta) for each cluster*/
	double* dlogh_dnu = (double*) malloc(sizeof(double)*K);/*Stores d(logh)/d(nu) for each cluster*/
	double* logh = (double*) malloc(sizeof(double)*K);/*Stores logh for each cluster*/
	uint32_t csz = K;
	double logh0; getLogH(&logh0, NULL, NULL, eta0, nu0, D, false);

	for(i=1; i < N; i++){
//		printf("Processing datapoint %d/%d, K= %d\n", i, N, K); 
		getStat(stat, &(T[i*D]), D);
		computeRho(r,w,stat, eta, nu, eta0, nu0, etatmp, getLogH, alpha, M, D, K);
		if( r[K] > epsclus){
			K++;
			eta = (double*) realloc(eta, sizeof(double)*K*M); /*Stores the natural params*/
			nu = (double*) realloc(nu, sizeof(double)*K); /*Stores the natural param strengths*/
			w = (double*) realloc(w, sizeof(double)*K); /*Stores cluster weights*/
			r = (double*) realloc(r, sizeof(double)*(K+1)); /*Stores normalized marginal log likelihoods*/
			rdiffsum = (double**) realloc(rdiffsum, sizeof(double*)*K); /*Stores normalized marginal log likelihoods*/
			ndiffsum = (double**) realloc(ndiffsum, sizeof(double*)*K); /*Stores normalized marginal log likelihoods*/
			for(k = 0; k < K-1; k++){
				rdiffsum[k] = (double*) realloc(rdiffsum[k], sizeof(double)*K);
				rdiffsum[k][K-1] = 1.0;
				ndiffsum[k] = (double*) realloc(ndiffsum[k], sizeof(double)*K);
				ndiffsum[k][K-1] = 1.0;
			}
			rdiffsum[K-1] = (double*) malloc(sizeof(double)*K);
			ndiffsum[K-1] = (double*) malloc(sizeof(double)*K);
			for(k=0; k < K; k++){
				rdiffsum[K-1][k] = 1.0;
				ndiffsum[K-1][k] = 1.0;
			}
			w[K-1] = 0.0;
			nu[K-1] = nu0;
			for(j=0;j< M; j++){
				eta[(K-1)*M+j] = eta0[j];
			}
		} else {
			double rsum = 0.0;
			for(k=0; k < K; k++){
				rsum += r[k];
			}
			for(k=0; k < K; k++){
				r[k]/= rsum;
			}
		}
		for(k=0; k < K; k++){
			w[k] += r[k];
			for(j=0; j < M; j++){
				eta[k*M+j] += r[k]*stat[j];
			}
			nu[k] += r[k];
			for(kk=0; kk < K; kk++){
				ndiffsum[k][kk] += 1.0;
				rdiffsum[k][kk] += fabs(r[k]-r[kk]);
			}
		}
		//prune low weight clusters
		double sumw = 0.;
		for(k=0; k < K; k++){
			sumw+= w[k];
		}
		uint32_t oldK = K;
		for(k=0;k < K; k++){
			if (w[k]/sumw < epspm && K>1){
				//prune the cluster -- shift everything down
				for(kk = k+1; kk < K; kk++){
					nu[kk-1] = nu[kk];
					w[kk-1] = w[kk];
					for(j=0; j < M; j++){
						eta[(kk-1)*M+j] = eta[kk*M+j];
					}
					for(j = 0; j < K; j++){
						rdiffsum[j][kk-1] = rdiffsum[j][kk];
						ndiffsum[j][kk-1] = ndiffsum[j][kk];
						rdiffsum[kk-1][j] = rdiffsum[kk][j];
						ndiffsum[kk-1][j] = ndiffsum[kk][j];
					}
				}
				k--;
				//if(k==K-1){k--;} 
				K--;
			}
		}
		//merge similar clusters
		for(k=0; k < K; k++){
		for(kk=k+1; kk < K; kk++){
			if(rdiffsum[k][kk]/ndiffsum[k][kk] < epspm && K>1){
				nu[k] += nu[kk]-nu0;
				w[k] += w[kk];
				for(j=0; j < M; j++){
					eta[k*M+j] += eta[kk*M+j] - eta0[j];
				}
				for(m=kk+1; m < K; m++){
					for(j=0; j < K; j++){
						rdiffsum[j][m-1] = rdiffsum[j][m];
						ndiffsum[j][m-1] = ndiffsum[j][m];
						rdiffsum[m-1][j] = rdiffsum[m][j];
						ndiffsum[m-1][j] = ndiffsum[m][j];
					}
				}
				kk--;// if(kk==K-1){kk--;} 
				K--;
			}
		}
		}
		if (K != oldK){
			eta = (double*) realloc(eta, sizeof(double)*K*M); /*Stores the natural params*/
			nu = (double*) realloc(nu, sizeof(double)*K); /*Stores the natural param strengths*/
			w = (double*) realloc(w, sizeof(double)*K); /*Stores cluster weights*/
			r = (double*) realloc(r, sizeof(double)*(K+1)); /*Stores normalized marginal log likelihoods*/
			for(k=K; k < oldK; k++){
				free(rdiffsum[k]); free(ndiffsum[k]);
			}
			rdiffsum = (double**) realloc(rdiffsum, sizeof(double*)*K); /*Stores normalized marginal log likelihoods*/
			ndiffsum = (double**) realloc(ndiffsum, sizeof(double*)*K); /*Stores normalized marginal log likelihoods*/
			for(k=0; k < K; k++){
				rdiffsum[k] = (double*) realloc(rdiffsum[k], sizeof(double)*K);
				ndiffsum[k] = (double*) realloc(ndiffsum[k], sizeof(double)*K);
			}
		}

		if (csz != K){
			zeta = (double*) realloc(zeta, sizeof(double)*N*K); /*Stores sum over all data(zeta) for each cluster*/
			sumzeta = (double*) realloc(sumzeta, sizeof(double)*K);
			sumzetaT = (double*) realloc(sumzetaT, sizeof(double)*K*M);
			a = (double*) realloc(a, sizeof(double)*K); /*Stores 1st beta parameters for weights*/
			b = (double*) realloc(b, sizeof(double)*K); /*Stores 2nd beta parameters for weights*/
			dlogh_deta = (double*) realloc(dlogh_deta, sizeof(double)*K*M); /*Stores d(logh)/d(eta) for each cluster*/
			dlogh_dnu = (double*) realloc(dlogh_dnu, sizeof(double)*K);/*Stores d(logh)/d(nu) for each cluster*/
			logh = (double*) realloc(logh, sizeof(double)*K);/*Stores logh for each cluster*/
			csz = K;
		}

		clock_gettime(CLOCK_MONOTONIC, &tf);
		if (*out_nTrace == 0){
			times[*out_nTrace] = (tf.tv_sec-ts.tv_sec) + (tf.tv_nsec - ts.tv_nsec)/1.0e9;
		} else {
			times[*out_nTrace] = times[*out_nTrace - 1] + (tf.tv_sec-ts.tv_sec) + (tf.tv_nsec - ts.tv_nsec)/1.0e9;
		}
		//compute test log likelihood
		convertSVAtoVB(zeta, sumzeta, sumzetaT, a, b, logh, dlogh_deta, dlogh_dnu, T, w, eta, nu, getLogH, getStat,getLogPostPred, alpha, M, D, N, K);
		//Remove empty clusters
		removeEmptyClustersX(zeta, sumzeta, sumzetaT, eta, nu, logh, dlogh_deta, dlogh_dnu, nu0, a, b, &Ktmp, N, M, K, false);
		testlls[*out_nTrace] = computeTestLogLikelihood(Ttest, eta, nu, a, b, getLogPostPred, Nt, D, M, K);
		(*out_nTrace)++;
		clock_gettime(CLOCK_MONOTONIC, &ts);
	}

	convertSVAtoVB(zeta, sumzeta, sumzetaT, a, b, logh, dlogh_deta, dlogh_dnu, T, w, eta, nu, getLogH, getStat,getLogPostPred, alpha, M, D, N, K);
	//Remove empty clusters
	removeEmptyClustersX(zeta, sumzeta, sumzetaT, eta, nu, logh, dlogh_deta, dlogh_dnu, nu0, a, b, &Ktmp, N, M, K, false);
	double finalobj= varBayesCost(zeta, sumzeta, sumzetaT, a, b, eta, eta0, nu, nu0, logh, logh0, dlogh_deta, dlogh_dnu, alpha, N, M, Ktmp);
	*out_K = Ktmp;

	/*Free non-output memory*/
	free(w); free(r); free(stat); free(etatmp); 
	free(sumzeta); free(sumzetaT);
	free(dlogh_deta); free(dlogh_dnu); free(logh);
	for(k=0; k < K; k++){
		free(rdiffsum[k]);
		free(ndiffsum[k]);
	}
	free(rdiffsum); free(ndiffsum);


	/*output*/
	*out_zeta = zeta;
	*out_eta = eta;
	*out_nu = nu;
	*out_a = a;
	*out_b = b;
	*out_K = K;
	*out_testlls = testlls;
	*out_times = times;

  	return finalobj;
}

void convertSVAtoVB(double*const zeta, double*const sumzeta, double*const sumzetaT, double*const a, double*const b, 
		    double*const logh, double*const dlogh_deta, double*const dlogh_dnu,
		const double* const T,
		const double* const w,
		const double* const eta, 
		const double* const nu, 
		void (*getLogH)(double*, double* const, double* const, const double * const, const double, const uint32_t, bool),
		void (*getStat)(double*, const double* const, const uint32_t),
		double* (*getLogPostPred)(const double* const, const double* const, const double* const, const uint32_t, const uint32_t, const uint32_t),
		const double alpha, 
		const uint32_t M,
		const uint32_t D,
		const uint32_t N,
		const uint32_t K){

	double* stat = (double*) malloc(sizeof(double)*M);
	double* psisum = (double*) malloc(sizeof(double)*K);

	uint32_t k, i, j;

	/*Compute the values of logh, derivatives*/
	for (k=0; k< K; k++){
		getLogH(&(logh[k]), &(dlogh_deta[M*k]), &(dlogh_dnu[k]), &(eta[M*k]), nu[k], D, true);
	}
	/*Compute the stick-breaking distribution and psisum*/
	double psibk = 0.0;
	for(k=0; k < K; k++){
		a[k] = w[k];
		b[k] = alpha;
		for(i=k+1; i < K; i++){
			b[k] += w[i];
		}
		double psiak = gsl_sf_psi(a[k]) - gsl_sf_psi(a[k]+b[k]);
		psisum[k] = psiak + psibk;
		psibk += gsl_sf_psi(b[k]) - gsl_sf_psi(a[k]+b[k]);
	}

	//initialize sumzeta
	for(k=0; k < K; k++){
		sumzeta[k] = 0.0;
		for(j=0; j < M; j++){
			sumzetaT[k*M+j] = 0.0;
		}
	}

	/*Compute the label distributions*/
	for (i=0; i < N; i++){
		/*Compute the statistic for datapoint*/
		getStat(stat, &(T[i*D]), D);
		/*Get the new label distribution*/
		updateLabelDist(&(zeta[i*K]), stat, dlogh_deta, dlogh_dnu, psisum, M,  K);
		for(k=0; k < K; k++){
			sumzeta[k] += zeta[i*K+k];
			for(j=0; j < M; j++){
				sumzetaT[k*M+j] += zeta[i*K+k]*stat[j];
			}
		}
	}

	free(psisum); free(stat);
	return;
}


double moVBDP_noAllocSumZeta(double* zeta, double* sumzeta, double* sumzetaT,
    double* eta, double* nu, double* a, double* b, unsigned int* out_K,
		double* times, double* testlls, uint32_t* out_nTrace,
		const double* const T, 
		const double* const Ttest, 
		const double alpha, 
		const double* const eta0, 
		const double nu0,
		void (*getLogH)(double*, double* const, double* const, const double * const, const double, const uint32_t, bool),
		void (*getStat)(double*, const double* const, const uint32_t),
		double* (*getLogPostPred)(const double* const, const double* const, const double* const, const uint32_t, const uint32_t, const uint32_t),
		const uint32_t N, const uint32_t Nt, 
		const uint32_t M,
		const uint32_t D,
		uint32_t K, uint32_t NBatch, uint32_t id)
{

	// get NBatch and initialize subbatch zetas
	uint32_t B = ceil((double)N/(double)NBatch);
	uint32_t finalNBatch = N % NBatch;
	if (finalNBatch == 0){
		finalNBatch = NBatch;
	}


	double* dlogh_deta = (double*) malloc(sizeof(double)*K*M); /*Stores d(logh)/d(eta) for each cluster*/
	double* dlogh_dnu = (double*) malloc(sizeof(double)*K);/*Stores d(logh)/d(nu) for each cluster*/
	double* logh = (double*) malloc(sizeof(double)*K);/*Stores logh for each cluster*/
	double logh0 = 0.0;
	double* stat = (double*) malloc(sizeof(double)*M);
	double* psisum = (double*) malloc(sizeof(double)*K);

	double* sumzetaB = (double*) malloc(sizeof(double)*K*B);
	double* sumzetaBT = (double*) malloc(sizeof(double)*K*M*B);

	*out_nTrace = 0;
	struct timespec ts, tf;
	clock_gettime(CLOCK_MONOTONIC, &ts);

	int i, j, k, bb;

	/*Initialize zeta randomly for K clusters*/
	initializeZeta(zeta, sumzeta, sumzetaT, T, getStat, N, M, D, K);

	/*Get the prior logh*/
	getLogH(&logh0, NULL, NULL, eta0, nu0, D, false);

	for (i = 0; i < K*B; i++){
		sumzetaB[i] = 0.0;
		for (j = 0; j < M; j++){
			sumzetaBT[i*M+j] = 0.0;
		}
	}
	for(bb= 0; bb < B; bb++){
		uint32_t NBatchb = (bb == B-1 ? finalNBatch : NBatch); 
		for (i = bb*NBatch; i < bb*NBatch+NBatchb; i++){
			getStat(stat, &(T[i*D]), D);
			for(k = 0; k < K; k++){
				sumzetaB[bb*K+k] += zeta[i*K+k];
				for(j = 0; j < M; j++){
					sumzetaBT[bb*K*M+k*M+j] += zeta[i*K+k]*stat[j];
				}
			}
		}
	}

	//initialize global parameters
	updateWeightDist(a, b, psisum, sumzeta, alpha, K);
	
	/*Update the exponential family parameter distributions*/
	/*Compute the values of logh, derivatives*/
	for (k=0; k< K; k++){
		updateParamDist(&(eta[k*M]), &(nu[k]), eta0, nu0, sumzeta[k], &(sumzetaT[k*M]), M);
		getLogH(&(logh[k]), &(dlogh_deta[M*k]), &(dlogh_dnu[k]), &(eta[M*k]), nu[k], D, true);
	}

	/*Proceed to the VB iteration*/
	double prevobj = INFINITY;
	double tol = 1e-8;
	double diff = 1.0;
	while(diff > tol){
		for(bb = 0; bb < B; bb++){
			//first remove the subbatch stats from the global statistics and initialize subbatch stats to 0
			for(k = 0; k < K; k++){
				sumzeta[k] -= sumzetaB[bb*K+k];
				sumzetaB[bb*K+k] = 0.0;
				for(j = 0; j < M; j++){
					sumzetaT[k*M+j] -= sumzetaBT[bb*K*M+k*M+j];
					sumzetaBT[bb*K*M+k*M+j] = 0.0;
				} 
			}
			//compute the subbatch label distribution using the current global parameters
			uint32_t NBatchb = (bb == B-1 ? finalNBatch : NBatch); 
			for (i=bb*NBatch; i < bb*NBatch+NBatchb; i++){
				/*Compute the statistic for datapoint*/
				getStat(stat, &(T[i*D]), D);
				/*Get the new label distribution*/
				updateLabelDist(&(zeta[i*K]), stat, dlogh_deta, dlogh_dnu, psisum, M,  K);
				/*Update the batch sumzetaB*/
				for(k = 0; k < K; k++){
					sumzetaB[bb*K+k] += zeta[i*K+k];
					for (j = 0; j < M; j++){
						sumzetaBT[bb*K*M+k*M+j] += zeta[i*K+k]*stat[j];
					}
				}
			}
			//reinsert subbatch stats into the global stats
			for(k = 0; k < K; k++){
				sumzeta[k] += sumzetaB[bb*K+k];
				for(j = 0; j < M; j++){
					sumzetaT[k*M+j] += sumzetaBT[bb*K*M+k*M+j];
				} 
			}
			//compute the new global parameters
			updateWeightDist(a, b, psisum, sumzeta, alpha, K);
			for (k=0; k< K; k++){
				updateParamDist(&(eta[k*M]), &(nu[k]), eta0, nu0, sumzeta[k], &(sumzetaT[k*M]), M);
				getLogH(&(logh[k]), &(dlogh_deta[M*k]), &(dlogh_dnu[k]), &(eta[M*k]), nu[k], D, true);
			}

			clock_gettime(CLOCK_MONOTONIC, &tf);
			if (*out_nTrace == 0){
				times[*out_nTrace] = (tf.tv_sec-ts.tv_sec) + (tf.tv_nsec - ts.tv_nsec)/1.0e9;
			} else {
				times[*out_nTrace] = times[*out_nTrace - 1] + (tf.tv_sec-ts.tv_sec) + (tf.tv_nsec - ts.tv_nsec)/1.0e9;
			}
			testlls[*out_nTrace] = computeTestLogLikelihood(Ttest, eta, nu, a, b, getLogPostPred, Nt, D, M, K);
			(*out_nTrace)++;
			/*Compute the variational objective*/
			if (bb == B-1){
				double obj= varBayesCost(zeta, sumzeta, sumzetaT, a, b, eta, eta0, nu, nu0, logh, logh0, dlogh_deta, dlogh_dnu, alpha, N, M, K);
				diff = fabs( (obj-prevobj)/obj);
				prevobj = obj;
			}
			clock_gettime(CLOCK_MONOTONIC, &ts);
		}
	}

	//Remove empty clusters
	removeEmptyClustersX(zeta, sumzeta, sumzetaT, eta, nu, logh, dlogh_deta, dlogh_dnu, nu0, a, b, &K, N, M, K, false);
	double finalobj= varBayesCost(zeta, sumzeta, sumzetaT, a, b, eta, eta0, nu, nu0, logh, logh0, dlogh_deta, dlogh_dnu, alpha, N, M, K);
	*out_K = K;

	/*Free non-output memory*/
	free(dlogh_deta); free(dlogh_dnu); free(logh); free(stat);
	free(sumzetaB); free(sumzetaBT);
  	free(psisum);

	return finalobj;
}

double moVBDP_noAlloc(double* zeta, double* eta, double* nu, double* a, double* b, unsigned int* out_K,
		double* times, double* testlls, uint32_t* out_nTrace,
		const double* const T, 
		const double* const Ttest, 
		const double alpha, 
		const double* const eta0, 
		const double nu0,
		void (*getLogH)(double*, double* const, double* const, const double * const, const double, const uint32_t, bool),
		void (*getStat)(double*, const double* const, const uint32_t),
		double* (*getLogPostPred)(const double* const, const double* const, const double* const, const uint32_t, const uint32_t, const uint32_t),
		const uint32_t N, const uint32_t Nt, 
		const uint32_t M,
		const uint32_t D,
		uint32_t K, uint32_t NBatch, uint32_t id)
{

	double* sumzeta = (double*) malloc(sizeof(double)*K); /*Stores sum over all data(zeta) for each cluster*/
	double* sumzetaT = (double*) malloc(sizeof(double)*K*M); /*Stores sum over all data(zeta*T) for each cluster*/

  double obj = moVBDP_noAllocSumZeta(zeta, sumzeta, sumzetaT, eta, nu, a, b, out_K, times, testlls, out_nTrace, T, Ttest,
    alpha, eta0, nu0, getLogH, getStat,getLogPostPred, N, Nt, M, D, K, NBatch,id);

	free(sumzeta); free(sumzetaT);
  return obj;
}

double moVBDP(double** out_zeta, double** out_eta, double** out_nu, double** out_a, double** out_b, uint32_t* out_K,
		double** out_times, double** out_testlls, uint32_t* out_nTrace,
		const double* const T, 
		const double* const Ttest, 
		const double alpha, 
		const double* const eta0, 
		const double nu0,
		void (*getLogH)(double*, double* const, double* const, const double * const, const double, const uint32_t, bool),
		void (*getStat)(double*, const double* const, const uint32_t),
		double* (*getLogPostPred)(const double* const, const double* const, const double* const, const uint32_t, const uint32_t, const uint32_t),
		const uint32_t N, const uint32_t Nt, 
		const uint32_t M,
		const uint32_t D,
		uint32_t K, uint32_t NBatch){

	double* zeta = (double*) malloc(sizeof(double)*N*K); /*Stores sum over all data(zeta) for each cluster*/
	double* eta = (double*) malloc(sizeof(double)*K*M); /*Stores d(logh)/d(eta) for each cluster*/
	double* nu = (double*) malloc(sizeof(double)*K); /*Stores d(logh)/d(eta) for each cluster*/
	double* a = (double*) malloc(sizeof(double)*K); /*Stores 1st beta parameters for weights*/
	double* b = (double*) malloc(sizeof(double)*K); /*Stores 2nd beta parameters for weights*/
	double* times = (double*) malloc(sizeof(double)*100000); /*Stores log times*/
	double* testlls = (double*) malloc(sizeof(double)*100000); /*Stores log testloglikes*/


  double obj = moVBDP_noAlloc(zeta,eta,nu,a,b,out_K,times,testlls,out_nTrace,T,Ttest,alpha,eta0,nu0,getLogH,getStat,getLogPostPred,N,Nt,M,D,K,NBatch,0);

	/*output*/
	*out_zeta = zeta;
	*out_eta = eta;
	*out_nu = nu;
	*out_a = a;
	*out_b = b;
	*out_times = times;
	*out_testlls = testlls;
  return obj;
}

//double soVBDP_noAllocSumZetaOld(double* zeta, double* sumzeta, double* sumzetaT,
//    double* eta, double* nu, double* a, double* b, unsigned int* out_K,
//		double* times, double* testlls, uint32_t* out_nTrace,
//		const double* const T, 
//		const double* const Ttest, 
//		const double alpha, 
//		const double* const eta0, 
//		const double nu0,
//		void (*getLogH)(double*, double* const, double* const, const double * const, const double, const uint32_t, bool),
//		void (*getStat)(double*, const double* const, const uint32_t),
//		double* (*getLogPostPred)(const double* const, const double* const, const double* const, const uint32_t, const uint32_t, const uint32_t),
//		const uint32_t N, const uint32_t Nt, 
//		const uint32_t M,
//		const uint32_t D,
//		uint32_t K, uint32_t NBatch, uint32_t id)
//{
//
//	// get NBatch/finalNBatch
//	uint32_t B = ceil((double)N/(double)NBatch);
//	uint32_t finalNBatch = N % NBatch;
//	if (finalNBatch == 0){
//		finalNBatch = NBatch;
//	}
//
//
//	double* dlogh_deta = (double*) malloc(sizeof(double)*K*M); /*Stores d(logh)/d(eta) for each cluster*/
//	double* dlogh_dnu = (double*) malloc(sizeof(double)*K);/*Stores d(logh)/d(nu) for each cluster*/
//	double* logh = (double*) malloc(sizeof(double)*K);/*Stores logh for each cluster*/
//	double logh0 = 0.0;
//	double* stat = (double*) malloc(sizeof(double)*M);
//	double* psisum = (double*) malloc(sizeof(double)*K);
//
//	double* sumzetaB = (double*) malloc(sizeof(double)*K);
//	double* sumzetaBT = (double*) malloc(sizeof(double)*K*M);
//	double* tmpa = (double*) malloc(sizeof(double)*K);
//	double* tmpb = (double*) malloc(sizeof(double)*K);
//	double* tmpnu = (double*) malloc(sizeof(double)*K);
//	double* tmpeta = (double*) malloc(sizeof(double)*K*M);
//
//	int i, j, k, bb;
//
//	/*Initialize zeta randomly for K clusters*/
//	initializeZeta(zeta, sumzeta, sumzetaT, T, getStat, N, M, D, K);
//
//	/*Get the prior logh*/
//	getLogH(&logh0, NULL, NULL, eta0, nu0, D, false);
//
//	//initialize global parameters
//	updateWeightDist(a, b, psisum, sumzeta, alpha, K);
//	
//	/*Update the exponential family parameter distributions*/
//	/*Compute the values of logh, derivatives*/
//	for (k=0; k< K; k++){
//		updateParamDist(&(eta[k*M]), &(nu[k]), eta0, nu0, sumzeta[k], &(sumzetaT[k*M]), M);
//		getLogH(&(logh[k]), &(dlogh_deta[M*k]), &(dlogh_dnu[k]), &(eta[M*k]), nu[k], D, true);
//	}
//
//	*out_nTrace = 0;
//	struct timespec ts, tf;
//	clock_gettime(CLOCK_MONOTONIC, &ts);
//
//	/*Proceed to the VB iteration*/
//	double prevobj = INFINITY;
//	double tol = 1e-8;
//	double diff = 1.0;
//	uint32_t step = 0;
//	while(diff > tol){
//		//initialize the full global stats to 0 before iterating over 
//		//subbatches and accumulating zeta
//		for(k=0; k < K; k++){
//			sumzeta[k] = 0.0;
//			for(j=0; j < M; j++){
//				sumzetaT[k*M+j] = 0.0;
//			}
//		}
//		//iterate over subbatches
//		for(bb = 0; bb < B; bb++){
//			//first initialize subbatch stats to 0
//			for(k = 0; k < K; k++){
//				sumzetaB[k] = 0.0;
//				for(j = 0; j < M; j++){
//					sumzetaBT[k*M+j] = 0.0;
//				}
//			}
//			//compute the subbatch label distribution using the current global parameters
//			uint32_t NBatchb = (bb == B-1 ? finalNBatch : NBatch); 
//			for (i=bb*NBatch; i < bb*NBatch+NBatchb; i++){
//				/*Compute the statistic for datapoint*/
//				getStat(stat, &(T[i*D]), D);
//				/*Get the new label distribution*/
//				updateLabelDist(&(zeta[i*K]), stat, dlogh_deta, dlogh_dnu, psisum, M,  K);
//				/*Update the batch sumzetaB*/
//				for(k = 0; k < K; k++){
//					sumzetaB[k] += zeta[i*K+k];
//					sumzeta[k] += zeta[i*K+k];
//					for (j = 0; j < M; j++){
//						sumzetaBT[k*M+j] += zeta[i*K+k]*stat[j];
//						sumzetaT[k*M+j] += zeta[i*K+k]*stat[j];
//					}
//				}
//			}
//			//amplify the subbatch stats to the total dataset size
//			for(k = 0; k < K; k++){
//				sumzetaB[k] *= (double)N/(double)NBatchb;
//				for(j = 0; j < M; j++){
//					sumzetaBT[k*M+j] *=(double)N/(double)NBatchb;
//				} 
//			}
//			//compute the noisy subbatch global parameters
//			updateWeightDist(tmpa, tmpb, psisum, sumzetaB, alpha, K);
//			for (k=0; k< K; k++){
//				updateParamDist(&(tmpeta[k*M]), &(tmpnu[k]), eta0, nu0, sumzetaB[k], &(sumzetaBT[k*M]), M);
//			}
//			//add the noisy step to the full global parameters with learning rate rho
//			//also compute helper variables dlogh & psisum
//			step++;
//			double rho = exp(-0.5*log(step+10.0));
//			double psibk = 0.0;
//			for (k=0; k< K; k++){
//				a[k] = rho*tmpa[k] + (1.0-rho)*a[k]; 
//				b[k] = rho*tmpb[k] + (1.0-rho)*b[k]; 
//				nu[k] = rho*tmpnu[k] + (1.0-rho)*nu[k]; 
//				for(j=0; j < M; j++){
//					eta[k*M+j] = rho*tmpeta[k*M+j] + (1.0-rho)*eta[k*M+j];
//				}
//				getLogH(&(logh[k]), &(dlogh_deta[M*k]), &(dlogh_dnu[k]), &(eta[M*k]), nu[k], D, true);
//				double psiak = gsl_sf_psi(a[k]) - gsl_sf_psi(a[k]+b[k]);
//      				psisum[k] = psiak + psibk;
//      				psibk += gsl_sf_psi(b[k]) - gsl_sf_psi(a[k]+b[k]);
//			}
//		}
//		/*Compute the variational objective*/
//		double obj= varBayesCost(zeta, sumzeta, sumzetaT, a, b, eta, eta0, nu, nu0, logh, logh0, dlogh_deta, dlogh_dnu, alpha, N, M, K);
//		diff = fabs( (obj-prevobj)/obj);
//		prevobj = obj;
//
//		clock_gettime(CLOCK_MONOTONIC, &tf);
//		if (*out_nTrace == 0){
//			times[*out_nTrace] = (tf.tv_sec-ts.tv_sec) + (tf.tv_nsec - ts.tv_nsec)/1.0e9;
//		} else {
//			times[*out_nTrace] = times[*out_nTrace - 1] + (tf.tv_sec-ts.tv_sec) + (tf.tv_nsec - ts.tv_nsec)/1.0e9;
//		}
//		testlls[*out_nTrace] = computeTestLogLikelihood(Ttest, eta, nu, a, b, getLogPostPred, Nt, D, M, K);
//		(*out_nTrace)++;
//		clock_gettime(CLOCK_MONOTONIC, &ts);
//    //printf("obj %f\t step %d",obj,step);
//	}
//
//	//Remove empty clusters
//	removeEmptyClustersX(zeta, sumzeta, sumzetaT, eta, nu, logh, dlogh_deta, dlogh_dnu, nu0, a, b, &K, N, M, K, false);
//	double finalobj= varBayesCost(zeta, sumzeta, sumzetaT, a, b, eta, eta0, nu, nu0, logh, logh0, dlogh_deta, dlogh_dnu, alpha, N, M, K);
//	*out_K = K;
//
//	/*Free non-output memory*/
//	free(dlogh_deta); free(dlogh_dnu); free(logh); free(stat);
//	free(sumzetaB); free(sumzetaBT);
//	free(tmpa); free(tmpb); free(tmpnu); free(tmpeta);
//  	free(psisum);
//
//	return finalobj;
//}

double soVBDP_noAllocSumZeta(double* zeta, double* sumzeta, double* sumzetaT,
    double* eta, double* nu, double* a, double* b, unsigned int* out_K,
		double* times, double* testlls, uint32_t* out_nTrace,
		const double* const T, 
		const double* const Ttest, 
		const double alpha, 
		const double* const eta0, 
		const double nu0,
		void (*getLogH)(double*, double* const, double* const, const double * const, const double, const uint32_t, bool),
		void (*getStat)(double*, const double* const, const uint32_t),
		double* (*getLogPostPred)(const double* const, const double* const, const double* const, const uint32_t, const uint32_t, const uint32_t),
		const uint32_t N, const uint32_t Nt, 
		const uint32_t M,
		const uint32_t D,
		uint32_t K, uint32_t NBatch, uint32_t id)
{

	double* dlogh_deta = (double*) malloc(sizeof(double)*K*M); /*Stores d(logh)/d(eta) for each cluster*/
	double* dlogh_dnu = (double*) malloc(sizeof(double)*K);/*Stores d(logh)/d(nu) for each cluster*/
	double* logh = (double*) malloc(sizeof(double)*K);/*Stores logh for each cluster*/
	double logh0 = 0.0;
	double* stat = (double*) malloc(sizeof(double)*M);
	double* psisum = (double*) malloc(sizeof(double)*K);

	double* sumzetaB = (double*) malloc(sizeof(double)*K);
	double* sumzetaBT = (double*) malloc(sizeof(double)*K*M);
	double* tmpa = (double*) malloc(sizeof(double)*K);
	double* tmpb = (double*) malloc(sizeof(double)*K);
	double* tmpnu = (double*) malloc(sizeof(double)*K);
	double* tmpeta = (double*) malloc(sizeof(double)*K*M);

	uint32_t* subbatch = (uint32_t*) malloc(sizeof(uint32_t)*NBatch);

	//random stuff for subsampling the dataset
	const gsl_rng_type * grt;
	gsl_rng* gr;
	gsl_rng_env_setup();
	grt = gsl_rng_default;
	gr = gsl_rng_alloc(grt);

	int i, j, k;

	*out_nTrace = 0;
	struct timespec ts, tf;
	clock_gettime(CLOCK_MONOTONIC, &ts);


	/*Initialize zeta randomly for K clusters*/
	initializeZeta(zeta, sumzeta, sumzetaT, T, getStat, N, M, D, K);

	/*Get the prior logh*/
	getLogH(&logh0, NULL, NULL, eta0, nu0, D, false);

	//initialize global parameters
	updateWeightDist(a, b, psisum, sumzeta, alpha, K);
	
	/*Update the exponential family parameter distributions*/
	/*Compute the values of logh, derivatives*/
	for (k=0; k< K; k++){
		updateParamDist(&(eta[k*M]), &(nu[k]), eta0, nu0, sumzeta[k], &(sumzetaT[k*M]), M);
		getLogH(&(logh[k]), &(dlogh_deta[M*k]), &(dlogh_dnu[k]), &(eta[M*k]), nu[k], D, true);
	}

	
	//initialize the full global stats to 0
	for(k=0; k < K; k++){
		sumzeta[k] = 0.0;
		for(j=0; j < M; j++){
			sumzetaT[k*M+j] = 0.0;
		}
	}
	//accumulate zeta/zetaT for objective computation
	for (i=0; i <N ; i++){
		getStat(stat, &(T[i*D]), D);
		for(k = 0; k < K; k++){
			sumzeta[k] += zeta[i*K+k];
			for (j = 0; j < M; j++){
				sumzetaT[k*M+j] += zeta[i*K+k]*stat[j];
			}
		}
	}

	/*Proceed to the VB iteration*/
	double prevobj = INFINITY;
	double tol = 1e-8;
	double diff = 1.0;
	uint32_t step = 0;
	while(diff > tol){

		//get a random subset
		for (i = 0; i < NBatch; i++){
			subbatch[i] = (uint32_t)floor(N*gsl_rng_uniform(gr));
		}

		//initialize subbatch stats to 0
		for(k = 0; k < K; k++){
			sumzetaB[k] = 0.0;
			for(j = 0; j < M; j++){
				sumzetaBT[k*M+j] = 0.0;
			}
		}

		//compute subbatch stats
		for (i=0; i <NBatch ; i++){
			/*Compute the statistic for datapoint*/
			getStat(stat, &(T[subbatch[i]*D]), D);
			//subtract from sumzeta/zeta
			for(k = 0; k < K; k++){
				sumzeta[k] -= zeta[subbatch[i]*K+k];
				for (j = 0; j < M; j++){
					sumzetaT[k*M+j] -= zeta[subbatch[i]*K+k]*stat[j];
				}
			}
			/*Get the new label distribution*/
			updateLabelDist(&(zeta[subbatch[i]*K]), stat, dlogh_deta, dlogh_dnu, psisum, M,  K);

			/*Update the batch sumzetaB and sumzeta*/
			for(k = 0; k < K; k++){
				sumzetaB[k] += zeta[subbatch[i]*K+k];
				sumzeta[k] += zeta[subbatch[i]*K+k];
				for (j = 0; j < M; j++){
					sumzetaBT[k*M+j] += zeta[subbatch[i]*K+k]*stat[j];
					sumzetaT[k*M+j] += zeta[subbatch[i]*K+k]*stat[j];
				}
			}
		}

		//amplify the subbatch stats to the total dataset size
		for(k = 0; k < K; k++){
			sumzetaB[k] *= (double)N/(double)NBatch;
			for(j = 0; j < M; j++){
				sumzetaBT[k*M+j] *=(double)N/(double)NBatch;
			}
		}

		//compute the noisy subbatch global parameters
		updateWeightDist(tmpa, tmpb, psisum, sumzetaB, alpha, K);
		for (k=0; k< K; k++){
			updateParamDist(&(tmpeta[k*M]), &(tmpnu[k]), eta0, nu0, sumzetaB[k], &(sumzetaBT[k*M]), M);
		}

		//add the noisy step to the full global parameters with learning rate rho
		//also compute helper variables dlogh & psisum
		step++;
		double rho = exp(-0.5*log(step+10.0));
		double psibk = 0.0;
		for (k=0; k< K; k++){
			a[k] = rho*tmpa[k] + (1.0-rho)*a[k]; 
			b[k] = rho*tmpb[k] + (1.0-rho)*b[k]; 
			nu[k] = rho*tmpnu[k] + (1.0-rho)*nu[k]; 
			for(j=0; j < M; j++){
				eta[k*M+j] = rho*tmpeta[k*M+j] + (1.0-rho)*eta[k*M+j];
			}
			getLogH(&(logh[k]), &(dlogh_deta[M*k]), &(dlogh_dnu[k]), &(eta[M*k]), nu[k], D, true);
			double psiak = gsl_sf_psi(a[k]) - gsl_sf_psi(a[k]+b[k]);
      			psisum[k] = psiak + psibk;
      			psibk += gsl_sf_psi(b[k]) - gsl_sf_psi(a[k]+b[k]);
		}

		

		clock_gettime(CLOCK_MONOTONIC, &tf);
		if (*out_nTrace == 0){
			times[*out_nTrace] = (tf.tv_sec-ts.tv_sec) + (tf.tv_nsec - ts.tv_nsec)/1.0e9;
		} else {
			times[*out_nTrace] = times[*out_nTrace - 1] + (tf.tv_sec-ts.tv_sec) + (tf.tv_nsec - ts.tv_nsec)/1.0e9;
		}
		testlls[*out_nTrace] = computeTestLogLikelihood(Ttest, eta, nu, a, b, getLogPostPred, Nt, D, M, K);
		(*out_nTrace)++;
		/*Compute the variational objective*/
		double obj= varBayesCost(zeta, sumzeta, sumzetaT, a, b, eta, eta0, nu, nu0, logh, logh0, dlogh_deta, dlogh_dnu, alpha, N, M, K);
		diff = fabs( (obj-prevobj)/obj);
		prevobj = obj;
		clock_gettime(CLOCK_MONOTONIC, &ts);
    //printf("obj %f\t step %d",obj,step);
	}

	//Remove empty clusters
	removeEmptyClustersX(zeta, sumzeta, sumzetaT, eta, nu, logh, dlogh_deta, dlogh_dnu, nu0, a, b, &K, N, M, K, false);
	double finalobj= varBayesCost(zeta, sumzeta, sumzetaT, a, b, eta, eta0, nu, nu0, logh, logh0, dlogh_deta, dlogh_dnu, alpha, N, M, K);
	*out_K = K;

	/*Free non-output memory*/
	free(dlogh_deta); free(dlogh_dnu); free(logh); free(stat);
	free(sumzetaB); free(sumzetaBT);
	free(tmpa); free(tmpb); free(tmpnu); free(tmpeta);
  	free(psisum);
  	free(subbatch);
  	gsl_rng_free(gr);

	return finalobj;
}

double soVBDP_noAlloc(double* zeta, double* eta, double* nu, double* a, double* b, unsigned int* out_K,
		double* times, double* testlls, uint32_t* out_nTrace,
		const double* const T, 
		const double* const Ttest, 
		const double alpha, 
		const double* const eta0, 
		const double nu0,
		void (*getLogH)(double*, double* const, double* const, const double * const, const double, const uint32_t, bool),
		void (*getStat)(double*, const double* const, const uint32_t),
		double* (*getLogPostPred)(const double* const, const double* const, const double* const, const uint32_t, const uint32_t, const uint32_t),
		const uint32_t N, const uint32_t Nt, 
		const uint32_t M,
		const uint32_t D,
		uint32_t K, uint32_t NBatch, uint32_t id)
{

	double* sumzeta = (double*) malloc(sizeof(double)*K); /*Stores sum over all data(zeta) for each cluster*/
	double* sumzetaT = (double*) malloc(sizeof(double)*K*M); /*Stores sum over all data(zeta*T) for each cluster*/

  double obj = soVBDP_noAllocSumZeta(zeta, sumzeta, sumzetaT, eta, nu, a, b, out_K, times, testlls, out_nTrace, T, Ttest,
    alpha, eta0, nu0, getLogH, getStat,getLogPostPred, N, Nt, M, D, K, NBatch,id);


	free(sumzeta); free(sumzetaT);
  return obj;
}

double soVBDP(double** out_zeta, double** out_eta, double** out_nu, double** out_a, double** out_b, uint32_t* out_K,
		double** out_times, double** out_testlls, uint32_t* out_nTrace,
		const double* const T, 
		const double* const Ttest, 
		const double alpha, 
		const double* const eta0, 
		const double nu0,
		void (*getLogH)(double*, double* const, double* const, const double * const, const double, const uint32_t, bool),
		void (*getStat)(double*, const double* const, const uint32_t),
		double* (*getLogPostPred)(const double* const, const double* const, const double* const, const uint32_t, const uint32_t, const uint32_t),
		const uint32_t N, const uint32_t Nt, 
		const uint32_t M,
		const uint32_t D,
		uint32_t K, uint32_t NBatch){

	double* zeta = (double*) malloc(sizeof(double)*N*K); /*Stores sum over all data(zeta) for each cluster*/
	double* eta = (double*) malloc(sizeof(double)*K*M); /*Stores d(logh)/d(eta) for each cluster*/
	double* nu = (double*) malloc(sizeof(double)*K); /*Stores d(logh)/d(eta) for each cluster*/
	double* a = (double*) malloc(sizeof(double)*K); /*Stores 1st beta parameters for weights*/
	double* b = (double*) malloc(sizeof(double)*K); /*Stores 2nd beta parameters for weights*/
	double* times = (double*) malloc(sizeof(double)*100000); /*Stores 2nd beta parameters for weights*/
	double* testlls = (double*) malloc(sizeof(double)*100000); /*Stores 2nd beta parameters for weights*/

  double obj = soVBDP_noAlloc(zeta,eta,nu,a,b,out_K,times,testlls,out_nTrace,T,Ttest,alpha,eta0,nu0,getLogH,getStat,getLogPostPred,N,Nt,M,D,K,NBatch,0);
  

	/*output*/
	*out_zeta = zeta;
	*out_eta = eta;
	*out_nu = nu;
	*out_a = a;
	*out_b = b;
	*out_times = times;
	*out_testlls = testlls;
  return obj;
}

int cmpfunc(const void * a, const void * b){
	double ad = *(double*)a;
	double bd = *(double*)b;
	if (ad < bd){
		return -1;
	} else if (ad == bd)
		return 0;
	else {
		return 1;
	}
}

double computeTestLogLikelihood(
		const double* const T, 
		double* eta, double* nu, double* a, double* b, 
		double* (*getLogPostPred)(const double* const, const double* const, const double* const, const uint32_t, const uint32_t, const uint32_t),
		const uint32_t Nt, 
		const uint32_t D,
		const uint32_t M,
		uint32_t K){

	//DEBUGGING CODE -- similar found in sdadp_impl.hpp
    //double eta0[21] = { 518081,      -421473,      -421473,       346853,     -15409.5,      12600.5,      467.401,
    //  			   402553,       949145,       949145,  2.24606e+06,      10554.7,      24983.3,          286,
    //  				411719,       388877,       388877,       367628,      8890.26,      8394.19,          200 };
    //double nu0[3] = {459.401, 278, 192};
	//double a0[3] = {1.0, 1.0, 1.0};
	//double b0[3] = {3.0, 2.0, 1.0};


	//if (Nt == 0){
	//	printf("WARNING: Test Log Likelihood = NaN since Nt = 0");
	//}
	////first get average weights
	//double stick = 1.0;
	//double* weights = (double*) malloc(sizeof(double)*3);
	//for(uint32_t k = 0; k < 3-1; k++){
	//	weights[k] = stick*a0[k]/(a0[k]+b0[k]);
	//	stick *= b0[k]/(a0[k]+b0[k]);
	//}
	//weights[3-1] = stick;

	////now get the log likelihoods for all data
	//double loglike = 0.0;
	//double* loglikes = getLogPostPred(T, eta0, nu0, 3, 2, Nt);
	//for(uint32_t i = 0; i < Nt; i++){
	//	for (uint32_t k = 0; k < 3; k++){
	//		loglikes[3*i+k] += log(weights[k]);
	//	}
	//	qsort(&(loglikes[3*i]), 3, sizeof(double), cmpfunc);
	//	double like = 0.0;
	//	for (uint32_t k = 0; k < 3; k++){
	//		//subtract off the max first
	//		like += exp(loglikes[3*i+k] - loglikes[3*i+(3-1)]);
	//	}
	//	loglike += log(like)+loglikes[3*i+(3-1)];
	//}
	//printf("VB.C LOG LIKE: %f\n", loglike/Nt);



	if (Nt == 0){
		printf("WARNING: Test Log Likelihood = NaN since Nt = 0");
	}
	//first get average weights
	double stick = 1.0;
	double* weights = (double*) malloc(sizeof(double)*K);
	for(uint32_t k = 0; k < K-1; k++){
		weights[k] = stick*a[k]/(a[k]+b[k]);
		stick *= b[k]/(a[k]+b[k]);
	}
	weights[K-1] = stick;

	//now get the log likelihoods for all data
	double loglike = 0.0;
	double* loglikes = getLogPostPred(T, eta, nu, K, D, Nt);
	for(uint32_t i = 0; i < Nt; i++){
		for (uint32_t k = 0; k < K; k++){
			loglikes[K*i+k] += log(weights[k]);
		}
		qsort(&(loglikes[K*i]), K, sizeof(double), cmpfunc);
		double like = 0.0;
		for (uint32_t k = 0; k < K; k++){
			//subtract off the max first
			like += exp(loglikes[K*i+k] - loglikes[K*i+(K-1)]);
		}
		loglike += log(like)+loglikes[K*i+(K-1)];
	}

	/////////////////OLD WAY
	//double* loglikes = (double*) malloc(sizeof(double)*K);
	//for(uint32_t i = 0; i < Nt; i++){
	//	//get the log likelihoods for all clusters
	//	for (uint32_t k = 0; k < K; k++){
	//		loglikes[k] = log(weights[k]) + getLogPostPred(&(T[i*D]), &(eta[k*M]), nu[k], D);
	//	}
	//	//numerically stable sum
	//	//first sort in increasing order
	//	qsort(loglikes, K, sizeof(double), cmpfunc);
	//	//then sum in increasing order
	//	double like = 0.0;
	//	for (uint32_t k = 0; k < K; k++){
	//		//subtract off the max first
	//		like += exp(loglikes[k] - loglikes[K-1]);
	//	}
	//	//now multiply by exp(max), take the log, and add to running loglike total
	//	loglike += loglikes[K-1] + log(like);
	//}
	free(weights); free(loglikes);
	return loglike/Nt; //should return NaN if Nt == 0
}

/*TODO, use the below old code when doing split merge stuff*/

/*initialize memory for computing the split proposal indices*/
/*
uint32_t szIdcs = 10;
if (kSplit >= 0){
	*nIdcs = 0;
	(*splitIdcs) = (uint32_t*) malloc(sizeof(uint32_t)*szIdcs);
	if ( (*splitIdcs) == NULL){
		printf("Error (updateLabelDist): Cannot allocate splitIdcs memory");
		exit(0);
	}
}
if (kSplit >= 0 && (splitIdcs == NULL || nIdcs == NULL)){
		printf("Error (updateLabelDist): if kSplit>=0, splitIdcs and nIdcs must point to something (cannot be NULL)");
		exit(0);
}*/
/*Finally, check if it should be in the next split proposal*/
/*
	if (kSplit >= 0 && zetai[kSplit] > 0.1){
		(*splitIdcs)[*nIdcs] = i;
		(*nIdcs)++;
		if (*nIdcs == szIdcs){
			szIdcs *= 2;
			uint32_t* tmp = (uint32_t*) realloc(*splitIdcs, sizeof(uint32_t)*szIdcs);
			if (tmp == NULL){
				printf("Error (updateLabelDist): Cannot reallocate memory");
				exit(0);
			}
			*splitIdcs = tmp;
		}
	}
	*/
