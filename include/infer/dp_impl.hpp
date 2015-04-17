#ifndef __DP_IMPL_HPP

template<class Model>
VarDP<Model>::VarDP(const std::vector<VXd>& train_data, const std::vector<VXd>& test_data, const Model& model, double alpha, uint32_t K) : model(model), test_data(test_data), alpha(alpha), K(K){
	M = this->model.getStatDimension();
	N = train_data.size();
	Nt = test_data.size();
	K0 = 0; // no nonstandard prior components


	//compute exponential family statistics once
	train_stats = MXd::Zero(N, M);
	for (uint32_t i = 0; i < N; i++){
		train_stats.row(i) = this->model.getStat(train_data[i]).transpose();
	}

	//initialize the random device
	std::random_device rd;
	rng.seed(rd());

	//initialize memory
	a = b = psisum = nu = sumzeta = dlogh_dnu = logh = VXd::Zero(K);
	zeta = MXd::Zero(N, K);
	sumzetaT = dlogh_deta = eta = MXd::Zero(K, M);
}

template<class Model>
VarDP<Model>::VarDP(const std::vector<VXd>& train_data, const std::vector<VXd>& test_data, const Distribution& prior, const Model& model, 
		double alpha, uint32_t K) : model(model), test_data(test_data), alpha(alpha), K(K){
	M = this->model.getStatDimension();
	N = train_data.size();
	Nt = test_data.size();
	K0 = prior.a.size();


	//compute exponential family statistics once
	train_stats = MXd::Zero(N, M);
	for (uint32_t i = 0; i < N; i++){
		train_stats.row(i) = this->model.getStat(train_data[i]).transpose();
	}

	//initialize the random device
	std::random_device rd;
	rng.seed(rd());

	//initialize memory
	a = b = psisum = nu = sumzeta = dlogh_dnu = logh = VXd::Zero(K);
	zeta = MXd::Zero(N, K);
	sumzetaT = dlogh_deta = eta = MXd::Zero(K, M);
	a0 = prior.a;
	b0 = prior.b;
	nu0 = prior.nu;
	eta0 = prior.eta;
	MXd tmp1 = MXd::Zero(eta0.rows(), eta0.cols());
	VXd tmp2 = VXd::Zero(nu0.size());
	this->model.getLogH(eta0, nu0, logh0, tmp1, tmp2);
}

template<class Model>
void VarDP<Model>::run(bool computeTestLL, double tol){
	//clear any previously stored results
	times.clear();
	objs.clear();
	testlls.clear();

	//create objective tracking vars
	double diff = 10.0*tol + 1.0;
	double obj = std::numeric_limits<double>::infinity();
	double prevobj = std::numeric_limits<double>::infinity();

	//start the timer
	Timer cpuTime, wallTime;
	cpuTime.start();
	wallTime.start();

	//initialize the variables
	init();

	
	//loop on variational updates
	while(diff > tol){
		updateWeightDist();
		updateParamDist();
		updateLabelDist();

		prevobj = obj;
		//store the current time
		times.push_back(cpuTime.get());
		//compute the objective
		obj = computeObjective();
		//save the objective
		objs.push_back(obj);
		//compute the obj diff
		diff = fabs((obj - prevobj)/obj);
		//if test likelihoods were requested, compute those (but pause the timer first)
		if (computeTestLL){
			cpuTime.stop();
			double testll = computeTestLogLikelihood();
			testlls.push_back(testll);
			cpuTime.start();
			std::cout << "obj: " << obj << " testll: " << testll << std::endl;
		} else {
			std::cout << "obj: " << obj << std::endl;
		}
	}
	//done!
	return;
}



template<class Model>
void VarDP<Model>::init(){
	//use kmeans++ to break symmetry in the intiialization
	std::vector<uint32_t> idces = kmeanspp(train_stats, [this](VXd& x, VXd& y){ return model.naturalParameterDistSquared(x, y); }, K, rng);
	for (uint32_t k = 0; k < K; k++){
		//Update the parameters 
		if (k < K0){
			for (uint32_t j = 0; j < M; j++){
	    		eta(k, j) = eta0(k, j)+train_stats(idces[k], j);
	    	}
			nu(k) = nu0(k) + 1.0;
		} else {
			for (uint32_t j = 0; j < M; j++){
	    		eta(k, j) = model.getEta0()(j)+train_stats(idces[k], j);
	    	}
			nu(k) = model.getNu0() + 1.0;
		}
	}

	//initialize a/b to the prior
	double psibk = 0.0;
	for (uint32_t k = 0; k < K; k++){
		//update weights
		if (k < K0){
			a(k) = a0(k);
			b(k) = b0(k);
		} else {
			a(k) = 1.0;
			b(k) = alpha;
		}
    	double psiak = digamma(a(k)) - digamma(a(k)+b(k));
    	psisum(k) = psiak + psibk;
    	psibk += digamma(b(k)) - digamma(a(k)+b(k));
	}

	//update logh/etc
	model.getLogH(eta, nu, logh, dlogh_deta, dlogh_dnu);

	updateLabelDist(); //finally make sure labels are updated


	////BACKUP OLD WAY OF DOING IT
	////create random statistics from data collection
	//std::uniform_int_distribution<> unii(0, N-1);
	//std::uniform_real_distribution<> unir;
	//MXd random_sumzeta = MXd::Zero(1, K);
	//for (uint32_t k = 0; k < K; k++){
	//	random_sumzeta(k) = 10.0*unir(rng);
	//}
	//MXd random_sumzetaT = MXd::Zero(K, M);
	//for (uint32_t k = 0; k < K; k++){
	//	random_sumzetaT.row(k) = train_stats.row(unii(rng))*random_sumzeta(k);
	//}

	////get psi and eta from that
	//double psibk = 0.0;
	//for (uint32_t k = 0; k < K; k++){
	//	//update weights
	//	a(k) = 1.0+random_sumzeta(k);
	//	b(k) = alpha;
	//	for (uint32_t j = k+1; j < K; j++){
	//		b(k) += random_sumzeta(k);
	//	}
    //	double psiak = digamma(a(k)) - digamma(a(k)+b(k));
    //	psisum(k) = psiak + psibk;
    //	psibk += digamma(b(k)) - digamma(a(k)+b(k));

	//    //Update the parameters
	//    for (uint32_t j = 0; j < M; j++){
	//    	eta(k, j) = model.getEta0()(j)+random_sumzetaT(k, j);
	//    }
	//	nu(k) = model.getNu0() + random_sumzeta(k);
	//}
	////update logh/etc
	//model.getLogH(eta, nu, logh, dlogh_deta, dlogh_dnu);

	//updateLabelDist(); //finally make sure labels are updated



	//std::cout << "INIT" << std::endl;
	//std::cout << "Eta: " << std::endl << eta << std::endl;
	//std::cout << "nu: " << std::endl << nu.transpose() << std::endl;
	//std::cout << "logh: " << std::endl << logh << std::endl;
	//std::cout << "dlogh_deta: " << std::endl << dlogh_deta << std::endl;
	//std::cout << "dlogh_dnu: " << std::endl << dlogh_dnu.transpose() << std::endl;
	//std::cout << "psisum: " << std::endl << psisum.transpose() << std::endl;
	//std::cout << "a: " << std::endl << a.transpose() << std::endl;
	//std::cout << "b: " << std::endl << b.transpose() << std::endl;
	return;

}

template<class Model>
void VarDP<Model>::updateWeightDist(){
	//Update a, b, and psisum
	double psibk = 0.0;
	for (uint32_t k = 0; k < K; k++){
		if (k < K0){
			a(k) = a0(k) + sumzeta(k);
			b(k) = b0(k);
		} else {
			a(k) = 1.0 + sumzeta(k);
			b(k) = alpha;
		}
		for (uint32_t j = k+1; j < K; j++){
			b(k) += sumzeta(j);
		}
    	double psiak = digamma(a(k)) - digamma(a(k)+b(k));
    	psisum(k) = psiak + psibk;
    	psibk += digamma(b(k)) - digamma(a(k)+b(k));
	}
	return;
}

template<class Model>
void VarDP<Model>::updateParamDist(){
	//Update the parameters
	for (uint32_t k = 0; k < K; k++){
		if (k < K0){
	    	for (uint32_t j = 0; j < M; j++){
	    		eta(k, j) = eta0(k, j)+sumzetaT(k, j);
	    	}
	    	nu(k) = nu0(k) + sumzeta(k);
		} else {
			for (uint32_t j = 0; j < M; j++){
	    		eta(k, j) = model.getEta0()(j)+sumzetaT(k, j);
	    	}
	    	nu(k) = model.getNu0() + sumzeta(k);
		}
	}
	//update logh/etc
	model.getLogH(eta, nu, logh, dlogh_deta, dlogh_dnu);
	return;
}

template<class Model>
void VarDP<Model>::updateLabelDist(){
	//update the label distribution
	sumzeta = VXd::Zero(K);
	sumzetaT = MXd::Zero(K, M);
	for (uint32_t i = 0; i < N; i++){
		//compute the log of the weights, storing the maximum so far
		double logpmax = -std::numeric_limits<double>::infinity();
		for (uint32_t k = 0; k < K; k++){
			zeta(i, k) = psisum(k) - dlogh_dnu(k);
			for (uint32_t j = 0; j < M; j++){
				zeta(i, k) -= train_stats(i, j)*dlogh_deta(k, j);
			}
			logpmax = (zeta(i, k) > logpmax ? zeta(i, k) : logpmax);
		}
		//make numerically stable by subtracting max, take exp, sum them up
		double psum = 0.0;
		for (uint32_t k = 0; k < K; k++){
			zeta(i, k) -= logpmax;
			zeta(i, k) = exp(zeta(i, k));
			psum += zeta(i, k);
		}
		//normalize
		for (uint32_t k = 0; k < K; k++){
			zeta(i, k) /= psum;
		}
		//update the sumzeta stats
		sumzeta += zeta.row(i).transpose();
		for(uint32_t k = 0; k < K; k++){
			sumzetaT.row(k) += zeta(i, k)*train_stats.row(i);
		}
	}
	return;
}

template<class Model>
typename VarDP<Model>::Distribution VarDP<Model>::getDistribution(){
	VarDP<Model>::Distribution d;
	d.zeta = this->zeta;
	d.a = this->a;
	d.b = this->b;
	d.eta = this->eta;
	d.nu = this->nu;
	return d;
}

template<class Model>
Trace VarDP<Model>::getTrace(){
	Trace tr;
	tr.times = this->times;
	tr.objs = this->objs;
	tr.testlls = this->testlls;
	return tr;
}


template<class Model>
double VarDP<Model>::computeObjective(){

	//get the label entropy
	MXd mzero = MXd::Zero(zeta.rows(), zeta.cols());
	MXd zlogz = zeta.array()*zeta.array().log();
	double labelEntropy = ((zeta.array() > 1.0e-16).select(zlogz, mzero)).sum();
	
	//get the variational beta entropy
	double betaEntropy = 0.0;
	for (uint32_t k = 0; k < K; k++){
        betaEntropy += -boost_lbeta(a(k), b(k)) + (a(k)-1.0)*digamma(a(k)) +(b(k)-1.0)*digamma(b(k))-(a(k)+b(k)-2.0)*digamma(a(k)+b(k));
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
		for (uint32_t j = 0; j < M; j++){
			likelihoodXEntropy -= sumzetaT(k, j)*dlogh_deta(k, j);
		}
	}

	//get the prior exponential cross entropy
    double priorExpXEntropy = 0; 
	for (uint32_t k = 0; k < K; k++){
		if (k < K0){
			priorExpXEntropy += logh0(k) - nu0(k)*dlogh_dnu(k);
			for (uint32_t j=0; j < M; j++){
	    		priorExpXEntropy -= eta0(k, j)*dlogh_deta(k, j);
	    	}
		} else{
			priorExpXEntropy +=  model.getLogH0() - model.getNu0()*dlogh_dnu(k);
			for (uint32_t j=0; j < M; j++){
	    		priorExpXEntropy -= model.getEta0()(j)*dlogh_deta(k, j);
	    	}
		}
	    
	}

	//get the prior label cross entropy
	double priorLabelXEntropy = 0.0;
	double psibk = 0.0;
	for (uint32_t k = 0; k < K; k++){
		double psiak = digamma(a(k)) - digamma(a(k)+b(k));
		priorLabelXEntropy += sumzeta(k)*(psiak + psibk);
		psibk += digamma(b(k)) - digamma(a(k)+b(k));
	}
	
	//get the prior beta cross entropy
	double priorBetaXEntropy = 0; 
	for (uint32_t k = 0; k < K; k++){
		if (k < K0){
			priorBetaXEntropy += (a0(k)-1.0)*(digamma(a(k)) - digamma(a(k)+b(k))) + (b0(k)-1.0)*(digamma(b(k)) - digamma(a(k)+b(k))) - boost_lbeta(a0(k), b0(k));
		} else {
			priorBetaXEntropy += (alpha-1.0)*(digamma(b(k)) - digamma(a(k)+b(k))) - boost_lbeta(1.0, alpha);
		}
	}

	return labelEntropy 
		 + betaEntropy 
		 + expEntropy
		 - likelihoodXEntropy
		 - priorExpXEntropy
		 - priorLabelXEntropy
		 - priorBetaXEntropy;
}


template<class Model>
double VarDP<Model>::computeTestLogLikelihood(){

	if (Nt == 0){
		std::cout << "WARNING: Test Log Likelihood = NaN since Nt = 0" << std::endl;
	}
	//first get average weights
	double stick = 1.0;
	VXd weights = VXd::Zero(K);
	for(uint32_t k = 0; k < K-1; k++){
		weights(k) = stick*a(k)/(a(k)+b(k));
		stick *= b(k)/(a(k)+b(k));
	}
	weights(K-1) = stick;

	//now loop over all test data and get weighted avg likelihood
	double loglike = 0.0;
	for(uint32_t i = 0; i < Nt; i++){
		std::vector<double> loglikes;
		for (uint32_t k = 0; k < K; k++){
			loglikes.push_back(log(weights(k)) + model.getLogPosteriorPredictive(test_data[i], eta.row(k), nu(k)));
		}
		//numerically stable sum
		//first sort in increasing order
		std::sort(loglikes.begin(), loglikes.end());
		//then sum in increasing order
		double like = 0.0;
		for (uint32_t k = 0; k < K; k++){
			//subtract off the max first
			like += exp(loglikes[k] - loglikes.back());
		}
		//now multiply by exp(max), take the log, and add to running loglike total
		loglike += loglikes.back() + log(like);
	}
	return loglike/Nt; //should return NaN if Nt == 0
}


double boost_lbeta(double a, double b){
	return lgamma(a)+lgamma(b)-lgamma(a+b);
}

template<class Model>
void VarDP<Model>::Distribution::save(std::string name){
	std::ofstream out_z(name+"-zeta.log", std::ios_base::trunc);
	out_z << zeta;
	out_z.close();

	std::ofstream out_e(name+"-eta.log", std::ios_base::trunc);
	out_e << eta;
	out_e.close();

	std::ofstream out_n(name+"-nu.log", std::ios_base::trunc);
	out_n << nu.transpose();
	out_n.close();


	std::ofstream out_ab(name+"-ab.log", std::ios_base::trunc);
	out_ab << a.transpose() << std::endl << b.transpose();
	out_ab.close();
}

#define __DP_IMPL_HPP
#endif /* __DP_HPP */
