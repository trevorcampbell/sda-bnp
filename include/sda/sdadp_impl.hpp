#ifndef __SDADP_IMPL_HPP

template<class Model>
SDADP<Model>::SDADP(const std::vector<VXd>& test_data, const Model& model, double alpha, uint32_t Knew, uint32_t nthr):
test_data(test_data), model(model), alpha(alpha), Knew(Knew), pool(nthr){
	timer.start(); //start the clock -- used for tracking performance
}

template<class Model>
void SDADP<Model>::submitMinibatch(const std::vector<VXd>& train_data){
	pool.submit(std::bind(&SDADP<Model>::varDPJob, this, train_data));
}

template<class Model>
VarDP<Model>::Distribution SDADP<Model>::getDistribution(){
	//have to lock/store since the worker pool might be doing stuff with it
	VarDP<Model>::Distribution out;
	{
		std::lock_guard<std::mutex> lock(distmut);
		out = dist;
	}
	return out;
}

template<class Model>
MultiTrace SDADP<Model>::getTrace(){
	//TODO be careful about locking here???
	return mtrace;
}

template<class Model>
double SDADP<Model>::computeTestLogLikelihood(){
	VarDP<Model>::Distribution dist0;
	{
		std::lock_guard<std::mutex> lock(distmut);
		dist0 = dist;
	} //release the lock

	if (test_data.size() == 0){
 		std::cout << "WARNING: Test Log Likelihood = NaN since Nt = 0" << std::endl;
		return 0;
	}
	//first get average weights
	double stick = 1.0;
	VXd weights = VXd::Zero(K);
	for(uint32_t k = 0; k < K-1; k++){
		weights(k) = stick*dist0.a(k)/(dist0.a(k)+dist0.b(k));
		stick *= dist0.b(k)/(dist0.a(k)+dist0.b(k));
	}
	weights(K-1) = stick;

	//now loop over all test data and get weighted avg likelihood
	double loglike = 0.0;
	for(uint32_t i = 0; i < test_data.size(); i++){
		std::vector<double> loglikes;
		for (uint32_t k = 0; k < dist0.eta.rows(); k++){
			loglikes.push_back(log(weights(k)) + model.getLogPosteriorPredictive(test_data[i], dist0.eta.row(k), dist0.nu(k)));
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
	return loglike/Nt;
}

template<class Model>
void SDADP<Model>::varDPJob(const std::vector<VXd>& train_data){

	//lock the mutex, get the distribution, unlock
	VarDP<Model>::Distribution dist0;
	{
		std::lock_guard<std::mutex> lock(distmut);
		dist0 = dist;
	} //release the lock

	//do minibatch inference
	VarDP<Model>::Distribution dist1;
	Trace tr;
	double t0 = timer.get();
	if(dist0.a.size() == 0){ //if there is no prior to work off of
		VarDP<Model> vdp(train_data, test_data, model, alpha, Knew);
		vdp.run(true);
	 	dist1 = vdp.getDistribution();
	 	tr = vdp.getTrace();
	} else { //if there is a prior
		VarDP<Model> vdp(train_data, test_data, dist0, model, alpha, dist0.a.size()+Knew);
		vdp.run(true);
	 	dist1 = vdp.getDistribution();
	 	tr = vdp.getTrace();
	}

	//lock mutex, store the local trace, get the current distribution, unlock
	VarDP<Model>::Distribution dist2;
	{
		std::lock_guard<std::mutex> lock(distmut);
		//add the result of the job to the multitrace
		mtrace.localstarttimes.push_back(t0);
		mtrace.localtimes.push_back(tr.times);
		mtrace.localobjs.push_back(tr.objs);
		mtrace.localtestlls.push_back(tr.testlls);
		//get the distribution at the current point
		dist2 = dist;
	} //release the lock


	//solve matching between dist1 nd dist2 with prior dist0

	//update the global distribution
	{
		std::lock_guard<std::mutex> lock(distmut);
		dist = ;
	} //release the lock

	//compute the test log likelihood of the global model
	t0 = timer.get();
	double testll = computeTestLogLikelihood();
	//lock mutex, update  with matching, unlock
	{
		std::lock_guard<std::mutex> lock(distmut);
		mtrace.globaltimes.push_back(t0);
		mtrace.globaltestlls.push_back(testll);
	} //release the lock
}


void buildMatchingCost(int* const costs,
					   const double* const eta1,   const double* const eta2, const double* const eta0,
				       const double* const nu1,    const double* const nu2, const double nu0,
				       const double* const logp01, const double* const logp02,
				       const double* const Enk1,   const double* const Enk2,
				       const double alpha, const double logh0, void (*getLogH)(double*, double* const, double* const, const double * const, const double, const uint32_t, bool),
				       const uint32_t K1,    const uint32_t K2, const uint32_t M, const uint32_t D){

	uint32_t k1, k2, j;
	double *etam = (double*) malloc(sizeof(double)*M);
	double* costs12 = (double*) malloc(sizeof(double)*K1*K2);	
	double* costs10 = (double*) malloc(sizeof(double)*K1);
	double* costs20 = (double*) malloc(sizeof(double)*K2);
	double cost0 = logh0;
	double mincost = cost0, maxcost = cost0;
	for (k1 = 0; k1 < K1; k1++){
		for (k2 = 0; k2 < K2; k2++){
			//merge the etas into a working var
			for(j=0; j < M; j++){
				etam[j] = eta1[k1*M+j] + eta2[k2*M+j] - eta0[j];
			}
			//compute the cost/row/col in the matching matrix
			//note that amps DP obj assumes maximization, so take the negative
			const uint32_t i = k1*K2 + k2;
			costs12[i] = -ampsDPObj_k(etam, nu1[k1]+nu2[k2]-nu0, exp(logp01[k1]+logp02[k2]), Enk1[k1]+Enk2[k2], alpha, getLogH, D);
			if(costs12[i] < mincost){ mincost = costs12[i];}
			if(costs12[i] > maxcost){ maxcost = costs12[i];}
		}
		//compute cost for matching k1 to nothing in agent 2
		costs10[k1] = -ampsDPObj_k(&(eta1[k1*M]), nu1[k1], exp(logp01[k1]), Enk1[k1], alpha, getLogH, D);
		if(costs10[k1] < mincost){ mincost = costs10[k1];}
		if(costs10[k1] > maxcost){ maxcost = costs10[k1];}
	}
	//compute score for matching k2 to nothing in K1
	for(k2 = 0; k2 < K2; k2++){
		//compute cost for matching k2 to nothing in agent 1
		costs20[k2] = -ampsDPObj_k(&(eta2[k2*M]), nu2[k2], exp(logp02[k2]), Enk2[k2], alpha, getLogH, D);
		if(costs20[k2] < mincost){ mincost = costs20[k2];}
		if(costs20[k2] > maxcost){ maxcost = costs20[k2];}
	}
	//now all costs have been computed, and max/min are known
	//subtract off the minimum from everything and remap to integers between 0 and INT_MAX/1000 
	maxcost -= mincost;
	double fctr = ((double)INT_MAX/1000.0)/maxcost;
	for(k1 = 0; k1 < K1; k1++){
		for(k2 = 0; k2 < K2; k2++){
			costs[k1*(K1+K2)+k2] = (int)(fctr*(costs12[k1*K2+k2] - mincost));
		}
		for(k2 = K2; k2 < K1+K2; k2++){
			costs[k1*(K1+K2)+k2] = (int)(fctr*(costs10[k1]-mincost));
		}
	}
	for(k1 = K1; k1 < K1+K2; k1++){
		for(k2 = 0; k2 < K2; k2++){
			costs[k1*(K1+K2)+k2] = (int)(fctr*(costs20[k2]-mincost));
		}
		for(k2 = K2; k2 < K1+K2; k2++){
			costs[k1*(K1+K2)+k2] = (int)(fctr*(cost0 - mincost));
		}
	} 
	free(costs12); free(costs10); free(costs20);	
	free(etam);
	return;
}

double ampsDPObj_k(const double * const eta, 
		const double nu, 
		const double logp0, 
		const double Enk, 
		const double alpha, 
		void (*getLogH)(double*, double* const, double* const, const double * const, const double, const uint32_t, bool),
		const uint32_t D){

	double dpreg = log(alpha)*(1.0-exp(logp0)) + gsl_sf_lngamma(Enk) ;
	double logh;
    getLogH(&logh, NULL, NULL, eta, nu, D, false);
    return dpreg - logh;
}
void computeLogP0EnkSingle(double* logp0, double* Enk,
				const double * const zetas, 
				const uint32_t N,
				const uint32_t K){
	uint32_t k, j;
		for(k = 0; k < K; k++){
			logp0[k] = 0.0;
			Enk[k] = 0.0;
			for (j = 0; j < N; j++){
				logp0[k] += log(1.0-zetas[j*K+k]);
				Enk[k] += zetas[j*K+k];
			}
			/*there is a possibility that logp0[k] = -inf, if the zetas are deterministic*/
			/*this is fine theoretically, but computationally when doing amps one needs to add/subtract statistics*/
			/*if you leave these as inf, you end up adding/subtracting inf from inf, which results in nans*/
			/*therefore, just replace these with large numbers -- double precision sets exp(-800) to 0, so:*/
			if(logp0[k] < -800.0){logp0[k] = -800.0;}
		}
	return;
}

#define __SDADP_IMPL_HPP
#endif /* __SDADP_IMPL_HPP */
