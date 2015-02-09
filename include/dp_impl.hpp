#include "dp.hpp"


double boost_lbeta(double a, double b){
	return boost_lgamma(a)+boost_lgamma(b)-boost_lgamma(a+b);
}

VarDP::VarDP(const std::vector<VXd>& train_data, const std::vector<VXd>& test_data, const Model& model, uint32_t K, std::string results_folder){

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
	this->initWeightsParams();
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
}

void VarDP::initWeightsParams(){
	//create random statistics from data collection
	std::uniform_real_distribution unir;
	MXd random_sumzeta = 5.0*(1.0+MXd::Random(1, this->K))/2.0;
	MXd random_sumzetaT = MXd::Zero(this->K, this->M);
	for (uint32_t k = 0; k < this->K; k++){
		random_sumzetaT.row(k) = some_stat*random_sumzeta(k);
	}

	//get psi and eta from that
	double psibk = 0.0;
	for (uint32_t k = 0; k < this->K; k++){
		//update
		this->a(k) = 1.0+random_sumzeta(k);
		this->b(k) = model.getAlpha();
		for (uint32_t j = k+1; j < this->K; j++){
			this->b(k) += random_sumzeta(k);
		}
    	double psiak = boost_psi(a(k)) - boost_psi(a(k)+b(k));
    	this->psisum(k) = psiak + psibk;
    	psibk += boost_psi(b(k)) - boost_psi(a(k)+b(k));
	}
	//Update the parameters
	for (uint32_t j = 0; j < M; j++){
		this->eta(j) = model.getEta0()(j)+random_sumzetaT(j);
	}
	this->nu = model.getNu0() + sumzeta;
	//update logh/etc
	this->model.getLogH(this->eta, this->nu, this->logh, this->dlogh_deta, this->dlogh_dnu);
	return;

}

void VarDP::updateWeightDist(){
	/*Update a, b, and psisum*/
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
	for (uint32_t j = 0; j < M; j++){
		this->eta(j) = model.getEta0()(j)+sumzetaT(j);
	}
	this->nu = model.getNu0() + sumzeta;
	//update logh/etc
	this->model.getLogH(this->eta, this->nu, this->logh, this->dlogh_deta, this->dlogh_dnu);
	return;
}

void VarDP::updateLabelDist(){
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
