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
typename VarDP<Model>::Distribution SDADP<Model>::getDistribution(){
	//have to lock/store since the worker pool might be doing stuff with it
	typename VarDP<Model>::Distribution out;
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
	typename VarDP<Model>::Distribution dist0;
	{
		std::lock_guard<std::mutex> lock(distmut);
		dist0 = dist;
	} //release the lock

	uint32_t K = dist0.eta.rows();
	uint32_t Nt = test_data.size();

	if (Nt == 0){
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
	for(uint32_t i = 0; i < Nt; i++){
		std::vector<double> loglikes;
		for (uint32_t k = 0; k < K; k++){
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
	typename VarDP<Model>::Distribution dist0;
	{
		std::lock_guard<std::mutex> lock(distmut);
		dist0 = dist;
	} //release the lock

	//do minibatch inference
	typename VarDP<Model>::Distribution dist1;
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
	typename VarDP<Model>::Distribution dist2;
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
	typename VarDP<Model>::Distribution distm;
	t0 = timer.get();
	distm = mergeDistributions(dist1, dist2, dist0);
	double mergeTime = timer.get()-t0;

	//update the global distribution
	{
		std::lock_guard<std::mutex> lock(distmut);
		mtrace.localmergetimes.push_back(mergetime);
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
		mtrace.globalclusters.push_back(dist.eta.rows());
		uint32_t nm = mtrace.globalmatchings.back();
		if (dist1.eta.rows() > dist0.eta.rows() && dist2.eta.rows() > dist0.eta.rows()){
			mtrace.globalmatchings.push_back(nm+1);
		} else {
			mtrace.globalmatchings.push_back(nm);
		}
	} //release the lock

	//done!
	return;
}


template<class Model>
typename VarDP<Model>::Distribution SDADP<Model>::mergeDistributions(typename VarDP<Model>::Distribution src, typename VarDP<Model>::Distribution dest, typename VarDP<Model>::Distribution prior){
	uint32_t Kp = prior.a.size();
	uint32_t Ks = src.a.size();
	uint32_t Kd = dest.a.size();
	typename VarDP<Model>::Distribution out;
	assert(Kd >= Kp && Ks >= Kp);
	if (Ks == Kp){
		//no new components created; just do the merge directly
		//match the first Ks elements (one for each src component) to the dest
		out = dest;
		out.eta.block(0, 0, Ks, out.eta.cols()) += src.eta - prior.eta;
		out.nu.head(Ks) += src.nu - prior.nu;
		out.zeta.conservativeResize(out.zeta.rows()+src.zeta.rows(), Eigen::NoChange);
		out.zeta.block(out.zeta.rows()-src.zeta.rows(), 0, src.zeta.rows(), out.zeta.cols()) = Eigen::MatrixXd::Zero(src.zeta.rows(), out.zeta.cols());
		out.zeta.block(out.zeta.rows()-src.zeta.rows(), 0, src.zeta.rows(), src.zeta.cols()) = src.zeta;
		VXd sumz = VXd::Zero(out.eta.rows());
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
	} else if (Kd == Kp) {
		//new components were created in src, but dest is still the same size as prior
		//just do the merge directly
		out = src;
		out.eta.block(0, 0, Kd, out.eta.cols()) += dest.eta - prior.eta;
		out.nu.head(Kd) += dest.nu - prior.nu;
		out.zeta.conservativeResize(out.zeta.rows()+dest.zeta.rows(), Eigen::NoChange);
		out.zeta.block(out.zeta.rows()-dest.zeta.rows(), 0, dest.zeta.rows(), out.zeta.cols()) = Eigen::MatrixXd::Zero(dest.zeta.rows(), out.zeta.cols());
		out.zeta.block(out.zeta.rows()-dest.zeta.rows(), 0, dest.zeta.rows(), dest.zeta.cols()) = dest.zeta;
		VXd sumz = VXd::Zero(out.eta.rows());
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
	} else {
		uint32_t Ksp = Ks-Kp;
		uint32_t Kdp = Kd-Kp;
		//new components were created in both dest and src -- need to solve a matching
		MXd costs = MXd::Zero(Ksp+Kdp, Ksp+Kdp);
		MXi costsi = MXi::Zero(Ksp+Kdp, Ksp+Kdp);

		//compute logp0 and Enk for d1 and d2
		VXd logp0s = VXd::Zero(Ksp);
		VXd logp0d = VXd::Zero(Kdp);
		VXd Enks = VXd::Zero(Ksp);
		VXd Enkd = VXd::Zero(Kdp);

		for (uint32_t k = 0; k < Ksp; k++){
			Enks(k) = src.zeta.col(Kp+k).sum();
			for (uint32_t j = 0; j < src.zeta.rows(); j++){
				logp0s(k) += log(1.0-src.zeta(j, Kp+k));
			}
			if (logp0s(k) < -800.0){
				logp0s(k) = -800.0;
			}
		}
		for (uint32_t k = 0; k < Kdp; k++){
			Enkd(k) = dest.zeta.col(Kp+k).sum();
			for (uint32_t j = 0; j < dest.zeta.rows(); j++){
				logp0d(k) += log(1.0-dest.zeta(j, Kp+k));
			}
			if (logp0d(k) < -800.0){
				logp0d(k) = -800.0;
			}
		}

		//compute costs
		MXd etam = MXd::Zero(1, model.getEta0().size());
		VXd num = VXd::Zero(1);
		VXd loghm = num;
		VXd dlogh_dnum = num;
		MXd dlogh_detam = etam;
		for (uint32_t i = 0; i < Ksp; i++){
			//compute costs in the 1-2 block and fill in the 1-0 block
			for (uint32_t j = 0; j < Kdp; j++){
				etam = src.eta.row(Kp+i) + dest.eta.row(Kp+j) - model.getEta0().transpose();
				num(0) = src.nu(Kp+i) + dest.nu(Kp+j) - model.getNu0();
				model.getLogH(etam, num, loghm, dlogh_detam, dlogh_dnum);
				costs(i, j) = loghm(0) - log(alpha)*(1.0-exp(logp0s(i)+logp0d(j))) - lgamma(Enks(i)+Enkd(j));
			}
			//compute costs in the 1-0 block
			etam = src.eta.row(Kp+i);
			num(0) = src.nu(Kp+i);
			model.getLogH(etam, num, loghm, dlogh_detam, dlogh_dnum);
			double c10 = loghm(0) - log(alpha)*(1.0-exp(logp0s(i))) - lgamma(Enks(i));
			for (uint32_t j = Kdp; j < Ksp+Kdp; j++){
				costs(i, j) = c10;
			}
		}

		//compute costs in the 2-0 block
		for (uint32_t j = 0; j < Kdp; j++){
			etam = dest.eta.row(Kp+j);
			num(0) = dest.nu(Kp+j);
			model.getLogH(etam, num, loghm, dlogh_detam, dlogh_dnum);
			double c20 = loghm(0) - log(alpha)*(1.0-exp(logp0d(j))) - lgamma(Enkd(j));
			for (uint32_t i = Ksp; i < Ksp+Kdp; i++){
				costs(i, j) = c20;
			}
		}

		//the 0-0 block is a constant
		for (uint32_t i = Ksp; i < Ksp+Kdp; i++){
			for (uint32_t j = Kdp; j < Ksp+Kdp; j++){
				costs(i, j) = model.getLogH0();
			}
		}

		//now all costs have been computed, and max/min are known
		//subtract off the minimum from everything and remap to integers between 0 and INT_MAX/1000 
		double mincost = costs.minCoeff();
		double maxcost = costs.minCoeff();
		maxcost -= mincost;
		double fctr = ((double)INT_MAX/1000.0)/maxcost;
		for (uint32_t i = 0; i < Ksp+Kdp; i++){
			for (uint32_t j = 0; j < Ksp+Kdp; j++){
				costsi(i, j) = (int)(fctr*(costs(i, j) - mincost));
			}
		}

		std::vector<int> matchings;
		int cost = hungarian(costsi, matchings);

		out = dest;
		//merge the first Kp elements directly (no matchings)
		out.eta.block(0, 0, Kp, out.eta.cols()) += src.eta.block(0, 0, Kp, src.eta.cols()) - prior.eta;
		out.nu.head(Ks) += src.nu.head(Kp) - prior.nu;
		out.zeta.conservativeResize(out.zeta.rows()+src.zeta.rows(), Eigen::NoChange);
		out.zeta.block(out.zeta.rows()-src.zeta.rows(), 0, src.zeta.rows(), out.zeta.cols()) = MXd::Zero(src.zeta.rows(), out.zeta.cols());
		out.zeta.block(out.zeta.rows()-src.zeta.rows(), 0, src.zeta.rows(), Kp) = src.zeta.block(0, 0, src.zeta.rows(), Kp);

		//merge the last Ksp elements using the matchings
		for (uint32_t i = Kp; i < Ks; i++){
			uint32_t toIdx = Kp+matchings[i-Kp];
			if (toIdx < Kd){
				out.eta.row(toIdx) += src.eta.row(i) - model.getEta0().transpose();
				out.nu(toIdx) += src.nu(i) - model.getNu0();
				out.zeta.block(out.zeta.rows()-src.zeta.rows(), toIdx, src.zeta.rows(), 1) = src.zeta.block(0, i, src.zeta.rows(), 1);
			} else {
				out.eta.conservativeResize(out.eta.rows()+1, Eigen::NoChange);
				out.eta.row(out.eta.rows()-1) = src.eta.row(i);
				out.nu.conservativeResize(out.nu.size()+1);
				out.nu(out.nu.size()-1) = src.nu(i);
				out.zeta.conservativeResize(Eigen::NoChange, out.zeta.cols()+1);
				out.zeta.block(0, out.zeta.cols()-1, out.zeta.rows(), 1) = MXd::Zero(out.zeta.rows(), 1);
				out.zeta.block(out.zeta.rows()-src.zeta.rows(), out.zeta.cols()-1, src.zeta.rows(), 1) = src.zeta.block(0, i, src.zeta.rows(), 1);
			}
		}
		out.a.resize(out.eta.rows());
		out.b.resize(out.eta.rows());
		VXd sumz = VXd::Zero(out.eta.rows());
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
	}
	return out;
}

#define __SDADP_IMPL_HPP
#endif /* __SDADP_IMPL_HPP */
