#ifndef __DP_IMPL_HPP
double boost_lbeta(double a, double b){
	return boost_lgamma(a)+boost_lgamma(b)-boost_lgamma(a+b);
}

VarDP::VarDP(const std::vector<VXd>& train_data, const std::vector<VXd>& test_data, const Model& model, uint32_t K){
	//copy in the model
	this->model = model;
	this->K = K;
	this->M = model.getStatDimension();
	this->N = train_data.size();
	this->Nt = test_data.size();

	//compute exponential family statistics once
	this->train_stats = MXd::Zero(this->N, this->M);
	this->test_stats = MXd::Zero(this->Nt, this->M);
	for (uint32_t i = 0; i < this->N; i++){
		this->train_stats.row(i) = model.getStat(train_data[i]).transpose();
	}
	for (uint32_t i = 0; i < this->Nt; i++){
		this->test_stats.row(i) = model.getStat(test_data[i]).transpose();
	}

	//initialize the random device
	std::random_device rd;
	if (rd.entropy() == 0){
		std::cout << "WARNING: ENTROPY 0, NOT SEEDING THE RANDOM GEN PROPERLY" << std::endl;
	}
	this->rng.seed(rd());

	//initialize memory
	this->a = this->b = this->psisum = this->nu = this->sumzeta = this->dlogh_dnu = this->logh = VXd::Zeros(this->K);
	this->zeta = MXd::Zeros(this->N, this->K);
	this->sumzetaT = this->dlogh_deta = this->eta = MXd::Zeros(this->K, this->M);
}

void VarDP::run(bool computeTestLL, double tol){
	//clear any previously stored results
	this->times.clear();
	this->objs.clear();
	this->testlls.clear();

	//create objective tracking vars
	double diff = 10.0*tol + 1.0;
	double obj = std::numeric_limits<double>::infinity();
	double prevobj = std::numeric_limits<double>::infinity();

	//start the timer
	Timer cpuTime;
	cpuTime.start();

	//initialize the variables
	this->initWeightsParams();
	this->updateLabelDist();

	//loop on variational updates
	while(diff > tol){
		this->updateWeightDist();
		this->updateParamDist();
		this->updateLabelDist();
		prevobj = obj;
		//store the current time
		this->times.push_back(cpuTime.get());
		//compute the objective
		obj = this->computeObjective();
		//save the objective
		this->objs.push_back(obj);
		//compute the obj diff
		diff = (obj - prevobj)/obj;
		//if test likelihoods were requested, compute those (but pause the timer first)
		if (computeTestLL){
			cpuTime.stop();
			double testll = this->computeTestLL();
			this->testlls.push_back(testll);
			cpuTime.start();
		}
	}
	//done!
	return;
}

void VarDP::initWeightsParams(){
	//create random statistics from data collection
	std::uniform_int_distribution unii(0, this->N);
	MXd random_sumzeta = 5.0*(1.0+MXd::Random(1, this->K))/2.0;
	MXd random_sumzetaT = MXd::Zero(this->K, this->M);
	for (uint32_t k = 0; k < this->K; k++){
		random_sumzetaT.row(k) = this->train_stats.row(unii(this->rng))*random_sumzeta(k);
	}

	//get psi and eta from that
	double psibk = 0.0;
	for (uint32_t k = 0; k < this->K; k++){
		//update weights
		this->a(k) = 1.0+random_sumzeta(k);
		this->b(k) = model.getAlpha();
		for (uint32_t j = k+1; j < this->K; j++){
			this->b(k) += random_sumzeta(k);
		}
    	double psiak = boost_psi(a(k)) - boost_psi(a(k)+b(k));
    	this->psisum(k) = psiak + psibk;
    	psibk += boost_psi(b(k)) - boost_psi(a(k)+b(k));

	    //Update the parameters
	    for (uint32_t j = 0; j < this->M; j++){
	    	this->eta(k, j) = model.getEta0()(j)+random_sumzetaT(k, j);
	    }
		this->nu(k) = model.getNu0() + random_sumzeta(k);
	}
	//update logh/etc
	this->model.getLogH(this->eta, this->nu, this->logh, this->dlogh_deta, this->dlogh_dnu);
	return;

}

void VarDP::updateWeightDist(){
	//Update a, b, and psisum
	double psibk = 0.0;
	for (uint32_t k = 0; k < this->K; k++){
		this->a(k) = 1.0+this->sumzeta(k);
		this->b(k) = model.getAlpha();
		for (uint32_t j = k+1; j < this->K; j++){
			this->b(k) += this->sumzeta(j);
		}
    	double psiak = boost_psi(a(k)) - boost_psi(a(k)+b(k));
    	this->psisum(k) = psiak + psibk;
    	psibk += boost_psi(b(k)) - boost_psi(a(k)+b(k));
	}
	return;
}

void VarDP::updateParamDist(){
	//Update the parameters
	for (uint32_t k = 0; k < this->K; k++){
	    for (uint32_t j = 0; j < this->M; j++){
	    	this->eta(k, j) = model.getEta0()(j)+sumzetaT(k, j);
	    }
	    this->nu(k) = model.getNu0() + sumzeta(k);
	}
	//update logh/etc
	this->model.getLogH(this->eta, this->nu, this->logh, this->dlogh_deta, this->dlogh_dnu);
	return;
}

void VarDP::updateLabelDist(){
	//update the label distribution
	for (uint32_t i = 0; i < this->N; i++){
		//compute the log of the weights, storing the maximum so far
		double logpmax = -std::numeric_limits<double>::infinity();
		for (uint32_t k = 0; k < this->K; k++){
			this->zeta(i, k) = this->psisum(k) - this->dlogh_dnu(k);
			for (uint32_t j = 0; j < this->M; j++){
				this->zeta(i, k) -= this->train_stats(i, j)*dlogh_deta(k, j);
			}
			logpmax = (zeta(i, k) > logpmax ? zeta(i, k) : logpmax);
		}
		//make numerically stable by subtracting max, take exp, sum them up
		double psum = 0.0;
		for (uint32_t k = 0; k < this->K; k++){
			zeta(i, k) -= logpmax;
			zeta(i, k) = exp(zeta(i, k));
			psum += zeta(i, k);
		}
		/*normalize*/
		for (uint32_t k = 0; k < this->K; k++){
			zeta(i, k) /= psum;
		}
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
    double priorExpXEntropy = K*model.getLogH0();
	for (uint32_t k = 0; k < K; k++){
		priorExpXEntropy -= model.getNu0()*dlogh_dnu(k);
	    for (uint32_t j=0; j < M; j++){
	    	priorExpXEntropy -= model.getEta0()(j)*dlogh_deta(k, j);
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
	double priorBetaXEntropy = -K*boost_lbeta(1.0, model.getAlpha());
	for (uint32_t k = 0; k < K; k++){
		priorBetaXEntropy += (model.getAlpha()-1.0)*(boost_psi(b(k)) - boost_psi(a(k)+b(k)));
	}

	return labelEntropy 
		 + betaEntropy 
		 + expEntropy
		 - likelihoodXEntropy
		 - priorExpXEntropy
		 - priorLabelXEntropy
		 - priorBetaXEntropy;
}

#define __DP_IMPL_HPP
#endif /* __DP_HPP */
