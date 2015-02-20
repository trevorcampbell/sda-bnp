#ifndef __HDP_IMPL_HPP

template<class Model>
VarHDP<Model>::VarHDP(const std::vector< std::vector<VXd> >& train_data, const std::vector< std::vector<VXd> >& test_data, const Model& model, double gam, double alpha, double eta, uint32_t T, uint32_t K) : model(model), test_data(test_data), gam(gam), alpha(alpha), eta(eta), T(T), K(K){
	this->M = model.getStatDimension();
	this->N = train_data.size();
	this->Nt = test_data.size();

	for (uint32_t i = 0; i < N; i++){
		this->Nl.push_back(train_data[i].size());
		train_stats.push_back(MXd::Zero(Nl.back(), M));
		for (uint32_t j = 0; j < Nl.back(); j++){
			train_stats.back().row(j) = this->model.getStat(train_data[i][j]).transpose();
		}
	}
	for (uint32_t i = 0; i < Nt; i++){
		this->Ntl.push_back(test_data[i].size());
	}

	//seed random gen
	std::random_device rd;
	rng.seed(rd());

	//initialize memory
	nu = logh = dlogh_dnu = VXd::Zero(T);
	eta = dlogh_deta = MXd::Zero(T, M);
	u = v = VXd::Zeros(T-1); 
	for (uint32_t i = 0; i < N; i++){
		a.push_back(VXd::Zeros(K-1));
		b.push_back(VXd::Zeros(K-1));
		zeta.push_back(MXd::Zero(Nl[i], K));
		phi.push_back(MXd::Zero(K, T));
	}
}

template<class Model>
void VarHDP<Model>::init(){
	//initialize topic word distribution and dlogh/d___
	std::uniform_int_distribution<> uniint(0, N);
	for(uint32_t t = 0; t < T; t++){
		uint32_t idx = uniint(rng);
		std::uniform_int_distribution<> uniintl(0, Nl[idx]);
		uint32_t idxl = uniintl(rng);
		for(uint32_t j =0; j < M; j++){
			eta(t, j) = (model.getEta0()(j) + 0.05*train_stats[idx](idxl, j))/1.05;
		}
		nu(j) = model.getNu0();
	}
	model.getLogH(eta, nu, logh, dlogh_deta, dlogh_dnu);

	//initialize the global topic weights
	u = VXd::Ones(T-1);
	v = gam*VXd::Ones(T-1);

	//initial local params
	for(uint32_t i =0; i < N; i++){
		//local weights
		a[i] = VXd::Ones(K-1);
		b[i] = alpha*VXd::Ones(K-1);
		//local psiabsum
		psiabsum[i] = VXd::Zero(K);
		double psibk = 0.0;
		for (uint32_t k = 0; k < K-1; k++){
    		double psiak = digamma(a[i](k)) - digamma(a[i](k)+b[i](k));
    		psiabsum[i](k) = psiak + psibk;
    		psibk += digamma(b[i](k)) - digamma(a[i](k)+b[i](k));
		}
		psiabsum[i](K-1) = psibk;

		//local correspondences
		phi[i] = 1.0/T*MXd::Ones(K, T);
		for(uint32_t k = 0; k < K; k++){
			for(uint32_t t = 0; t < T; t++){
				phiNsum[idx](k) += phi[idx](k, t)*dlogh_dnu(t);
				phiEsum[idx].row(k) += phi[idx](k, t)*dlogh_deta.row(t);
			}
		}

		//everything needed for the first label update is ready
	}
}


template<class Model>
void VarHDP<Model>::run(bool computeTestLL = false, double tol = 1e-6){
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
		//update the local distributions
		updateLocalDists(tol);
		//update the global distribution
		updateGlobalDist();

		prevobj = obj;
		//store the current time
		times.push_back(cpuTime.get());
		//compute the objective
		obj = computeFullObjective();
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
VarHDPResults VarHDP<Model>::getResults(){
	VarHDPResults hdpr;
	//TODO: fill in
	hdpr.times = this->times;
	hdpr.objs = this->objs;
	hdpr.testlls = this->testlls;
	return hdpr;
}

template<class Model>
void VarHDP<Model>::updateLocalDists(double tol){
	//zero out global stats
	phizetasum = MXd::Zero(T);
	phisum = MXd::Zero(T);
	phizetaTsum = MXd::Zero(T, M);
	//loop over all local obs collections
	for (uint32_t i = 0; i < N; i++){
		//create objective tracking vars
		double diff = 10.0*tol + 1.0;
		double obj = std::numeric_limits<double>::infinity();
		double prevobj = std::numeric_limits<double>::infinity();

		//run variational updates on the local params
		while(diff > tol){
			updateLocalLabelDist(idx);
			updateLocalWeightDist(idx);
			updateLocalCorrespondenceDist(idx);
			prevobj = obj;
			obj = computeLocalObjective(idx);
			//compute the obj diff
			diff = fabs((obj - prevobj)/obj);
		}
		//add phi/zeta to global stats
		for(uint32_t t = 0; t < T; t++){
			for(uint32_t k = 0; k < K; k++){
				phizetasum(t) += phi[i](k, t)*zetasum[i](k);
				phisum(t) += phi[i](k, t);
				for (uint32_t j = 0; j < M; j++){
					phizetaTsum(t, j) += phi[i](k, t)*zetaTsum[i](k, j);
				}
			}
		}
	}
}

template<class Model>
void VarHDP<Model>::updateLocalWeightDist(uint32_t idx){
	//Update a, b, and psisum
	psiabsum[idx] = VXd::Zero(K);
	double psibk = 0.0;
	for (uint32_t k = 0; k < K-1; k++){
		a[idx](k) = 1.0+zetasum[idx](k);
		b[idx](k) = alpha;
		for (uint32_t j = k+1; j < K; j++){
			b[idx](k) += zetasum[idx](j);
		}
    	double psiak = digamma(a[idx](k)) - digamma(a[idx](k)+b[idx](k));
    	psiabsum[idx](k) = psiak + psibk;
    	psibk += digamma(b[idx](k)) - digamma(a[idx](k)+b[idx](k));
	}
	psiabsum[idx](K-1) = psibk;
}

template<class Model>
void VarHDP<Model>::updateLocalLabelDist(uint32_t idx){
	//update the label distribution
	zetasum[idx] = VXd::Zero(K);
	zetaTsum[idx] = MXd::Zero(K, M);
	for (uint32_t i = 0; i < Nl[idx]; i++){
		//compute the log of the weights, storing the maximum so far
		double logpmax = -std::numeric_limits<double>::infinity();
		for (uint32_t k = 0; k < K; k++){
			zeta[idx](i, k) = psiabsum[idx](k) - phiNsum[idx](k);
			for (uint32_t j = 0; j < M; j++){
				zeta[idx](i, k) -= train_stats[idx](i, j)*phiEsum[idx](k, j);
			}
			logpmax = (zeta[idx](i, k) > logpmax ? zeta[idx](i, k) : logpmax);
		}
		//make numerically stable by subtracting max, take exp, sum them up
		double psum = 0.0;
		for (uint32_t k = 0; k < K; k++){
			zeta[idx](i, k) -= logpmax;
			zeta[idx](i, k) = exp(zeta[idx](i, k));
			psum += zeta[idx](i, k);
		}
		//normalize
		for (uint32_t k = 0; k < K; k++){
			zeta[idx](i, k) /= psum;
		}
		//update the zetasum stats
		zetasum[idx] += zeta[idx].row(i).transpose();
		for(uint32_t k = 0; k < K; k++){
			zetaTsum[idx].row(k) += zeta[idx](i, k)*train_stats[idx].row(i);
		}
	}
}

template<class Model>
void VarHDP<Model>::updateLocalCorrespondenceDist(uint32_t idx){
	//update the correspondence distribution
	phiNsum[idx] = VXd::Zero(K);
	phiEsum[idx] = MXd::Zero(K, M);
	
	for (uint32_t k = 0; k < K; k++){
		//compute the log of the weights, storing the maximum so far
		double logpmax = -std::numeric_limits<double>::infinity();
		for (uint32_t t = 0; t < T; t++){
			phi[idx](k, t) = psiuvsum[idx](t) - zetasum[idx](k)*dlogh_dnu(t);
			for (uint32_t j = 0; j < M; j++){
				phi[idx](k, t) -= zetaTsum[idx](k, j)*dlogh_deta(t, j);
			}
			logpmax = (phi[idx](k, t) > logpmax ? phi[idx](k, t) : logpmax);
		}
		//make numerically stable by subtracting max, take exp, sum them up
		double psum = 0.0;
		for (uint32_t t = 0; t < T; t++){
			phi[idx](k, t) -= logpmax;
			phi[idx](k, t) = exp(phi[idx](k, t));
			psum += phi[idx](k, t);
		}
		//normalize
		for (uint32_t t = 0; t < T; t++){
			phi[idx](k, t) /= psum;
		}
		//update the phisum stats
		for(uint32_t t = 0; t < T; t++){
			phiNsum[idx](k) += phi[idx](k, t)*dlogh_dnu(t);
			phiEsum[idx].row(k) += phi[idx](k, t)*dlogh_deta.row(t);
		}
	}
}



template<class Model>
void VarHDP<Model>::updateGlobalDist(){
	updateGlobalWeightDist();
	updateGlobalParamDist();
}

template<class Model>
void VarHDP<Model>::updateGlobalWeightDist(){
	//Update u, v, and psisum
	psiuvsum[idx] = VXd::Zero(T);
	double psivk = 0.0;
	for (uint32_t t = 0; t < T-1; t++){
		u[idx](t) = 1.0+phisum(t);
		v[idx](t) = gam;
		for (uint32_t j = t+1; j < T; j++){
			v[idx](t) += phisum(j);
		}
    	double psiuk = digamma(u(k)) - digamma(u(k)+v(k));
    	psiabsum[idx](k) = psiuk + psivk;
    	psivk += digamma(v(k)) - digamma(u(k)+v(k));
	}
	psiuvsum[idx](T-1) = psivk;
}

template<class Model>
void VarHDP<Model>::updateGlobalParamDist(){
	for (uint32_t t = 0; t < T; t++){
		nu(t) = model.getNu0() + phizetasum(t);
		for (uint32_t j = 0; j < M; j++){
			eta(t, j) = model.getEta0()(j) + phizetaTsum.row(t, j);
		}
	}
	model.getLogH(eta, nu, logh, dlogh_deta, dlogh_dnu);
}

template<class Model>
double VarHDP<Model>::computeFullObjective(){
	//reuse the local code for computing each local obj
	double obj = 0;
	for (uint32_t i =0 ; i < N; i++){
		obj += computeLocalObjective(i);
	}

	//get the variational beta entropy
	double betaEntropy = 0.0;
	for (uint32_t k = 0; t < T-1; k++){
        betaEntropy += -boost_lbeta(u(t), v(t)) + (u(t)-1.0)*digamma(u(t)) +(v(t)-1.0)*digamma(v(t))-(u(t)+v(t)-2.0)*digamma(u(t)+v(t));
	}

	//get the variational exponential family entropy
	double expEntropy = 0.0;
	for (uint32_t t = 0; t < T; t++){
		expEntropy += logh(t) - nu(t)*dlogh_dnu(t);
		for (uint32_t j = 0; j < M; j++){
			expEntropy -= eta(t, j)*dlogh_deta(t, j);
		}
	}

	//prior exp cross entropy
    double priorExpXEntropy = T*model.getLogH0();
	for (uint32_t t = 0; t < T; t++){
		priorExpXEntropy -= model.getNu0()*dlogh_dnu(t);
	    for (uint32_t j=0; j < M; j++){
	    	priorExpXEntropy -= model.getEta0()(j)*dlogh_deta(t, j);
	    }
	}

	//get the prior beta cross entropy
	double priorBetaXEntropy = -T*boost_lbeta(1.0, alpha);
	for (uint32_t t = 0; t < T-1; t++){
		priorBetaXEntropy += (alpha-1.0)*(digamma(v(t)) - digamma(u(t)+v(t)));
	}

	//output
	return obj
		+ betaEntropy
		+ expEntropy
		- priorExpXEntropy
		- priorBetaXentropy;
}

template<class Model>
double VarHDP<Model>::computeLocalObjective(uint32_t idx){
	//get the label entropy
	MXd mzero = MXd::Zero(zeta[idx].rows(), zeta[idx].cols());
	MXd zlogz = zeta[idx].array()*zeta[idx].array().log();
	double labelEntropy = ((zeta[idx].array() > 1.0e-16).select(zlogz, mzero)).sum();

	//get the correspondence entropy
	MXd pzero = MXd::Zero(phi[idx].rows(), phi[idx].cols());
	MXd plogp = phi[idx].array()*phi[idx].array().log();
	double corrEntropy = ((phi[idx].array() > 1.0e-16).select(plogp, pzero)).sum();

	//get the variational beta entropy
	double betaEntropy = 0.0;
	for (uint32_t k = 0; k < K-1; k++){
        betaEntropy += -boost_lbeta(a[idx](k), b[idx](k)) + (a[idx](k)-1.0)*digamma(a[idx](k)) +(b[idx](k)-1.0)*digamma(b[idx](k))-(a[idx](k)+b[idx](k)-2.0)*digamma(a[idx](k)+b[idx](k));
	}

	//get the likelihood cross entropy
	double likelihoodXEntropy = 0.0;
	for (uint32_t k = 0; k < K; k++){
		likelihoodXEntropy -= zetasum[idx](k)*phiNsum[idx](k);
		for (uint32_t j = 0; j < M; j++){
			likelihoodXEntropy -= zetaTsum[idx](k, j)*phiEsum[idx](k, j);
		}
	}

	//get the prior label cross entropy
	double priorLabelXEntropy = 0.0;
	double psibk = 0.0;
	for (uint32_t k = 0; k < K-1; k++){
		double psiak = digamma(a[idx](k)) - digamma(a[idx](k)+b[idx](k));
		priorLabelXEntropy += zetasum[idx](k)*(psiak + psibk);
		psibk += digamma(b[idx](k)) - digamma(a[idx](k)+b[idx](k));
	}
	priorLabelXEntropy += zetasum[idx](K-1)*psibk;

	//get the prior correspondence cross entropy
	double priorCorrXEntropy = 0.0;
	double psivt = 0.0;
	for (uint32_t t = 0; t < T-1; t++){
		double psiut = digamma(u(t)) - digamma(u(t)+v(t));
		for (uint32_t k = 0; k < K; k++){
			priorCorrXEntropy += phisum[idx](k, t)*(psiut + psivt);
		}
		psivt += digamma(v(t)) - digamma(u(t)+v(t));
	}
	priorCorrXEntropy += phisum[idx](T-1)*psivk;

	//get the prior beta cross entropy
	double priorBetaXEntropy = -K*boost_lbeta(1.0, alpha);
	for (uint32_t k = 0; k < K-1; k++){
		priorBetaXEntropy += (alpha-1.0)*(digamma(b[idx](k)) - digamma(a[idx](k)+b[idx](k)));
	}

	return labelEntropy 
		 + corrEntropy
		 + betaEntropy 
		 - likelihoodXEntropy
		 - priorCorrXEntropy
		 - priorLabelXEntropy
		 - priorBetaXEntropy;
}

template<class Model>
double VarHDP<Model>::computeTestLogLikelihood(){

	//TODO: fill in
	//run local variational inference on some % of each test collection
	//compute logposteriorpredictive on the other % using the local variational params

	return 0.0;

}

double boost_lbeta(double a, double b){
	return lgamma(a)+lgamma(b)-lgamma(a+b);
}

#define __HDP_IMPL_HPP
#endif /* __HDP_HPP */
