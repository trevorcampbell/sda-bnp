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


#define __SDADP_IMPL_HPP
#endif /* __SDADP_IMPL_HPP */
