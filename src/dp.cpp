#include "dp.hpp"

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


void VarDP::run(bool computeTestLL, double tol){

	double diff = 10.0*tol + 1.0;
	double obj = std::numeric_limits<double>::infinity();
	double prevobj = std::numeric_limits<double>::infinity();
	while(diff > tol){
		this->updateWeightDist();
		this->updateParamDist();
		this->updateLabelDist();
		prevobj = obj;
		obj = this->computeObjective();
		diff = (obj - prevobj)/obj;
		if (computeTestLL){
			double testll = this->computeTestLL();
		}
	}
}



		double computeObjective();
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

		/*Compute the variational objective*/
    double obj= varBayesCost(zeta, sumzeta, sumzetaT, a, b, eta, eta0, nu, nu0,
        logh, *logh0, dlogh_deta, dlogh_dnu, alpha, N, M, K);
		diff = fabs( (obj- (*prevobj))/obj);
		*prevobj = obj;
	  *out_K = K;
//		printf("@id %d: Obj %f\tdelta %f\n", id,obj,diff);

    free(dlogh_deta); free(dlogh_dnu); free(stat); free(psisum);
    return diff;
}

	//initialize global params to smoothed doc stats
	
	//while diff > tol
	//update local params
	//update global params
}

double VarDP::computeObjective(){

	//get the label entropy
	MXd mzero = MatrixXd::Zero(zeta.rows(), zeta.cols());
	MXd zlogz = zeta.array()*zeta.array().log();
	double labelEntropy = ((zeta.array() > 1.0e-16).select(zlogz, mzero)).sum();

	//get the variational beta entropy
	

	//get the variational exponential family entropy
	

	//get the likelihood cross entropy
	

	//get the prior exponential cross entropy
	
	//get the prior label cross entropy
	
	//get the prior beta cross entropy


	return labelEntropy 
		 + betaEntropy 
		 + expEntropy
		 - likelihoodXEntropy
		 - priorExpXEntropy
		 - priorLabelXEntropy
		 - priorBetaXEntropy;
}

double varLabelEntropy(const double * const zeta, const uint32_t N, const uint32_t K){
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
double cost = varBetaEntropy(a, b, K)
			+ varExpEntropy(logh, eta, nu, dlogh_deta, dlogh_dnu, M, K)
			- likeCrossEntropy(sumzeta, sumzetaT, dlogh_deta, dlogh_dnu, M, K)
			- priorExpCrossEntropy(logh0, eta0, nu0, dlogh_deta, dlogh_dnu, M, K)
			- priorLabelCrossEntropy(a, b, sumzeta, K)
			- priorBetaCrossEntropy(a, b, alpha, K);
    if(zeta != NULL)
      cost += varLabelEntropy(zeta, N, K);

