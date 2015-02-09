#include "dp.hpp"


double boost_lbeta(double a, double b){
	return boost_lgamma(a)+boost_lgamma(b)-boost_lgamma(a+b);
}

void VarDP::run(bool computeTestLL, double tol){

	this->times.clear();
	this->objs.clear();
	this->testlls.clear();
	//TODO this model clear

	double diff = 10.0*tol + 1.0;
	double obj = std::numeric_limits<double>::infinity();
	double prevobj = std::numeric_limits<double>::infinity();
	Timer cpuTime;
	cpuTime.start();
	while(diff > tol){
		this->updateWeightDist();
		this->updateParamDist();
		this->updateLabelDist();
		prevobj = obj;
		this->times.push_back(cpuTime.get());
		obj = this->computeObjective();
		this->objs.push_back(obj);
		diff = (obj - prevobj)/obj;
		if (computeTestLL){
			cpuTime.stop();
			double testll = this->computeTestLL();
			this->testlls.push_back(testll);
			cpuTime.start();
		}
	}
	return cpuTime.stop();

        /*double computeObjective();
        //Update the stick-breaking distribution and psisum
		updateWeightDist(a, b, psisum, sumzeta, alpha, K);
	
		//Update the exponential family parameter distributions
		//Compute the values of logh, derivatives
		for (k=0; k< K; k++){
			updateParamDist(&(eta[k*M]), &(nu[k]), eta0, nu0, sumzeta[k], &(sumzetaT[k*M]), M);
			getLogH(&(logh[k]), &(dlogh_deta[M*k]), &(dlogh_dnu[k]), &(eta[M*k]), nu[k], D, true);
		}

		//Initialize sumzeta to 0
		for(k = 0; k < K; k++){
			sumzeta[k] = 0.0;
			for (j = 0; j < M; j++){
				sumzetaT[k*M+j] = 0.0;
			}
		}

		//Update the label distributions
		for (i=0; i < N; i++){
			//Compute the statistic for datapoint
			getStat(stat, &(T[i*D]), D);
			//Get the new label distribution
			updateLabelDist(&(zeta[i*K]), stat, dlogh_deta, dlogh_dnu, psisum, M,  K);
			//Update sumzeta
			for(k = 0; k < K; k++){
				sumzeta[k] += zeta[i*K+k];
				for (j = 0; j < M; j++){
					sumzetaT[k*M+j] += zeta[i*K+k]*stat[j];
				}
			}
		}*/
}

void VarDP::updateWeightDist(){
	/*Update a, b, and psisum*/
	double psibk = 0.0;
	for (uint32_t k = 0; k < this->K; k++){
		this->a(k) = 1.0+this->sumzeta(k);
		this->b(k) = this->alpha;
		for (uint32_t j = k+1; j < this->K; j++){
			this->b(k) += this->sumzeta(j);
		}
    	double psiak = boost_psi(a(k)) - boost_psi(a(k)+b(k));
    	this->psisum(k) = psiak + psibk;
    	psibk += boost_psi(b(k)) - boost_psi(a(k)+b(k));
	}
	return;
}

void updateParamDist(){
	/*Update the parameters*/
	/*Note that eta[i*M+j] is indexing for a 2D array (K clusters, M components) stored as 1D*/
	for (uint32_t j = 0; j < M; j++){
		this->eta(j) = eta0(j)+sumzetaT(j);
	}
	this->nu = nu0 + sumzeta;
	return;
}

void updateLabelDist(){
	//update the label distribution
	//compute the log of the weights, storing the maximum so far
	double logpmax = -std::numeric_limits<double>::infinity();
	for (uint32_t k = 0; k < this->K; k++){
		this->zeta(k) = this->psisum(k) - this->dlogh_dnu(k);
		for (uint32_t j = 0; j < M; j++){
			this->zeta(k) -= stat(j)*dlogh_deta(k, j);
		}
		logpmax = (zeta(k) > logpmax ? zeta(k) : logpmax);
	}
	//make numerically stable by subtracting max, take exp, sum them up
	double psum = 0.0;
	for (k = 0; k < K; k++){
		zeta(k) -= logpmax;
		zeta(k) = exp(zeta(k));
		psum += zeta(k);
	}
	/*normalize*/
	for (k = 0; k < K; k++){
		zeta(k) /= psum;
	}

	return;
}

void VarDP::getResults(std::vector<double>& times, std::vector<double>& objs, std::vector<double>& testlls){
	//TODO output model
	times = this->times;
	objs = this->objs;
	testlls = this->testlls;
	return;
}


double VarDP::computeObjective(){

	//get the label entropy
	MXd mzero = MatrixXd::Zero(zeta.rows(), zeta.cols());
	MXd zlogz = zeta.array()*zeta.array().log();
	double labelEntropy = ((zeta.array() > 1.0e-16).select(zlogz, mzero)).sum();

	//get the variational beta entropy
	double betaEntropy = 0.0;
	for (uint32_t k = 0; k < this->K; k++){
        betaEntropy += -boost_lbeta(a(k), b(k)) + (a(k)-1.0)*boost_psi(a(k)) +(b(k)-1.0)*boost_psi(b(k))-(a(k)+b(k)-2.0)*boost_psi(a(k)+b(k));
	}

	//get the variational exponential family entropy
	double expEntropy = 0.0;
	for (uint32_t k = 0; k < K; k++){
		expEntropy += logh(k) - nu(k)*dlogh_dnu(k);
		for (uint32_t j = 0; j < M; j++){
			expEntropy -= eta(k, j)*dlogh_deta(k, j);
		}
	}

	//get the likelihood cross entropy
	double likelihoodXEntropy = 0.0;
	for (uint32_t k = 0; k < K; k++){
		likelihoodXEntropy -= sumzeta(k)*dlogh_dnu(k);
		for (j = 0; j < M; j++){
			likelihoodXEntropy -= sumzetaT(k, j)*dlogh_deta(k, j);
		}
	}

	//get the prior exponential cross entropy
    double priorExpXEntropy = K*logh0;
	for (uint32_t k = 0; k < K; k++){
		priorExpXEntropy -= nu0*dlogh_dnu(k);
	    for (uint32_t j=0; j < M; j++){
	    	priorExpXEntropy -= eta0(j)*dlogh_deta(k, j);
	    }
	}

	//get the prior label cross entropy
	double priorLabelXEntropy = 0.0;
	double psibk = 0.0;
	for (uint32_t k = 0; k < K; k++){
		double psiak = boost_psi(a(k)) - boost_psi(a(k)+b(k));
		ent += sumzeta(k)*(psiak + psibk);
		psibk += boost_psi(b(k)) - boost_psi(a(k)+b(k));
	}
	
	//get the prior beta cross entropy
	double ent = -K*boost_lbeta(1.0, alpha);
	for (uint32_t k = 0; k < K; k++){
		ent += (alpha-1.0)*(boost_psi(b(k)) - boost_psi(a(k)+b(k)));
	}

	return labelEntropy 
		 + betaEntropy 
		 + expEntropy
		 - likelihoodXEntropy
		 - priorExpXEntropy
		 - priorLabelXEntropy
		 - priorBetaXEntropy;
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

