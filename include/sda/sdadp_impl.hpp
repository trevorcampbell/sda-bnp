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
void SDADP<Model>::waitUntilDone(){
	pool.wait();
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
	MultiTrace mt;
	{
		std::lock_guard<std::mutex> lock(distmut);
		mt = mtrace;
	}
	return mt;
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
	VarDP<Model>::Distribution distm;
	distm = mergeDistributions(dist2, dist1, dist0);

	//update the global distribution
	{
		std::lock_guard<std::mutex> lock(distmut);
		dist = distm;
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

	//done!
	return;
}


VarDP<Model>::Distribution mergeDistributions(VarDP<Model>::Distribution src, VarDP<Model>::Distribution dest, VarDP<Model>::Distribution prior){
	uint32_t Kp = prior.a.size();
	uint32_t Ks = src.a.size();
	uint32_t Kd = dest.a.size();

	MXd costs = MXd::Zero(Ks+Kd, Ks+Kd);
	MXi costsi = MXi::Zero(Ks+Kd, Ks+Kd);

	//compute logp0 and Enk for d1 and d2
	VXd logp0s = VXd::Zero(Ks);
	VXd logp0d = VXd::Zero(Kd);
	VXd Enks = VXd::Zero(Ks);
	VXd Enkd = VXd::Zero(Kd);

	for (uint32_t k = 0; k < Ks; k++){
		Enks(k) = src.zeta.col(k).sum();
		for (uint32_t j = 0; j < src.zeta.rows(); j++){
			logp0s(k) += log(1.0-src.zeta(j, k));
		}
		if (logp0s(k) < -800.0){
			logp0s(k) = -800.0;
		}
	}
	for (uint32_t k = 0; k < Kd; k++){
		Enkd(k) = dest.zeta.col(k).sum();
		for (uint32_t j = 0; j < dest.zeta.rows(); j++){
			logp0d(k) += log(1.0-dest.zeta(j, k));
		}
		if (logp0d(k) < -800.0){
			logp0d(k) = -800.0;
		}
	}

	//compute costs
	VXd etam = VXd::Zeros(model.getEta0().size())
	VXd num = VXd::Zeros(1);
	VXd loghm = VXd::Zeros(1);
	VXd dlogh_dnum = VXd::Zeros(1);
	MXd dlogh_detam = MXd::Zeros(1, etam.size());
	for (uint32_t i = 0; i < Ks; i++){
		//compute costs in the 1-2 block and fill in the 1-0 block
		for (uint32_t j = 0; j < Kd; j++){
			etam = src.eta.row(i) + dest.eta.row(j);
			num(0) = src.nu(i) + dest.nu(j);
			if (j < Kp){
				etam -= prior.eta.row(j);
				num(0) -= prior.nu(j);
			} else {
				etam -= model.getEta0();
				num(0) -= model.getNu0();
			}
			model.getLogH(etam, num, loghm, dlogh_detam, dlogh_dnum);
			costs(i, j) = loghm(0) - log(alpha)*(1.0-exp(logp0s(i)+logp0d(j))) - gsl_sf_lngamma(Enks(i)+Enkd(j));
		}
		//compute costs in the 1-0 block
		model.getLogH(src.eta.row(i), src.nu(i), loghm, dlogh_detam, dlogh_dnum);
		double c10 = loghm(0) - log(alpha)*(1.0-exp(logp0s(i))) - gsl_sf_lngamma(Enks(i));
		for (uint32_t j = Kd; j < Ks+Kd; j++){
			costs(i, j) = c10;
		}
	}

	//compute costs in the 2-0 block
	for (uint32_t j = 0; j < K2; j++){
		model.getLogH(dest.eta.row(j), dest.nu(j), loghm, dlogh_detam, dlogh_dnum);
		double c20 = loghm(0) - log(alpha)*(1.0-exp(logp0d(i))) - gsl_sf_lngamma(Enkd(i));
		for (uint32_t i = Ks; i < Ks+Kd; i++){
			costs(i, j) = c20;
		}
	}

	//the 0-0 block is a constant
	for (uint32_t i = Ks; i < Ks+Kd; i++){
		for (uint32_t j = Kd; j < Ks+Kd; j++){
			costs(i, j) = model.getLogH0();
		}
	}

	//now all costs have been computed, and max/min are known
	//subtract off the minimum from everything and remap to integers between 0 and INT_MAX/1000 
	double mincost = costs.minCoeff();
	double maxcost = costs.minCoeff();
	maxcost -= mincost;
	double fctr = ((double)INT_MAX/1000.0)/maxcost;
	for (uint32_t i = 0; i < Ks+Kd; i++){
		for (uint32_t j = 0; j < Ks+Kd; j++){
			costsi(i, j) = (int)(fctr*(costs(i, j) - mincost));
		}
	}

	std::vector<int> matchings;
	int cost = hungarian(costsi, matchings);

	VarDP<Model>::Distribution out = dest;

	//match the first Ks elements (one for each src component) to the dest
	out.zeta.conservativeResize(out.zeta.rows()+src.zeta.rows(), Eigen::NoChange_t);
	for (uint32_t i = 0; i < Ks; i++){
		if (matchings[i] < Kd){
			out.eta.row(matchings[i]) += src.eta.row(i);
			out.nu(matchings[i]) += src.nu(i);
			if (matchings[i] < Kp){
				out.eta.row(matchings[i]) -= prior.eta.row(matchings[i]);
				out.nu(matchings[i]) -= prior.nu(matchings[i]);
			} else {
				out.eta.row(matchings[i]) -= model.getEta0();
				out.nu(matchings[i]) -= model.getNu0();
			}
			out.zeta.block(out.zeta.rows()-src.zeta.rows(), matchings[i], src.zeta.rows(), 1) = src.zeta.block(0, i, src.zeta.rows(), 1);
		} else {
			out.eta.conservativeResize(out.eta.rows()+1, Eigen::NoChange_t);
			out.eta.row(out.eta.rows()-1) = src.eta.row(i);
			out.nu.conservativeResize(out.nu.size()+1);
			out.nu(out.nu.size()-1) = src.nu(i);
			out.zeta.conservativeResize(Eigen::NoChange_t, out.zeta.cols()+1);
			out.zeta.block(out.zeta.rows()-src.zeta.rows(), out.zeta.cols()-1, src.zeta.rows(), 1) = src.zeta.block(0, i, src.zeta.rows(), 1);
		}
	}
	out.a.resize(out.eta.rows());
	out.b.resize(out.eta.rows());
	VXd sumz = VXd::Zeros(out.eta.rows());
	for (uint32_t k = 0; k < out.eta.rows(); k++){
		sumz(k) = out.zeta.col(k).sum();
	}
	for (uint32_t k = 0; k < out.eta.rows(); k++){
		out.a(k) = 1.0 + sumz(k);
		out.b(k) = alpha;
		for (uint32_t j = k+1; j < out.eta.rows(); j++){
			out.b(k) += sumz(j);
		}
	}

	return out;
}

#define __SDADP_IMPL_HPP
#endif /* __SDADP_IMPL_HPP */
