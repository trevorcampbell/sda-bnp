#ifndef __SDADP_IMPL_HPP

template<class Model>
SDADP<Model>::SDADP(const std::vector<VXd>& test_data, const Model& model, double alpha, uint32_t Knew, uint32_t nthr):
test_data(test_data), model(model), alpha(alpha), Knew(Knew), pool(nthr){
	timer.start(); //start the clock -- used for tracking performance
	dist.K = 0;
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

	uint32_t K = dist0.K;
	uint32_t Nt = test_data.size();

	if (Nt == 0){
 		std::cout << "WARNING: Test Log Likelihood = NaN since Nt = 0" << std::endl;
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

	if (train_data.size() == 0){
		return;
	}

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
	if(dist0.K == 0){ //if there is no prior to work off of
		VarDP<Model> vdp(train_data, test_data, model, alpha, Knew);
		vdp.run(true);
	 	dist1 = vdp.getDistribution();
	 	tr = vdp.getTrace();
	} else { //if there is a prior
		VarDP<Model> vdp(train_data, test_data, dist0, model, alpha, dist0.K+Knew);
		vdp.run(true);
	 	dist1 = vdp.getDistribution();
	 	tr = vdp.getTrace();
	}
	//remove empty clusters
	for (uint32_t k = dist0.K; k < dist1.K; k++){
		if (dist1.sumz(k) < 1.0 && k < dist1.K-1){
			dist1.sumz.segment(k, dist1.K-(k+1)) = (dist1.sumz.segment(k+1, dist1.K-(k+1))).eval(); //eval stops aliasing
			dist1.sumz.conservativeResize(dist1.K-1);
			dist1.logp0.segment(k, dist1.K-(k+1)) = (dist1.logp0.segment(k+1, dist1.K-(k+1))).eval(); //eval stops aliasing
			dist1.logp0.conservativeResize(dist1.K-1);
			dist1.nu.segment(k, dist1.K-(k+1)) = (dist1.nu.segment(k+1, dist1.K-(k+1))).eval(); //eval stops aliasing
			dist1.nu.conservativeResize(dist1.K-1);
			dist1.a.segment(k, dist1.K-(k+1)) = (dist1.a.segment(k+1, dist1.K-(k+1))).eval(); //eval stops aliasing
			dist1.a.conservativeResize(dist1.K-1);
			dist1.b.segment(k, dist1.K-(k+1)) = (dist1.b.segment(k+1, dist1.K-(k+1))).eval(); //eval stops aliasing
			dist1.b.conservativeResize(dist1.K-1);
			dist1.eta.block(k, 0, dist1.K-(k+1), dist1.eta.cols()) = (dist1.eta.block(k+1, 0, dist1.K-(k+1), dist1.eta.cols())).eval();
			dist1.eta.conservativeResize(dist1.K-1, Eigen::NoChange);
			dist1.K--;
			k--;
		} else if (dist1.sumz(k) < 1.0){ //just knock off the end
			dist1.sumz.conservativeResize(dist1.K-1);
			dist1.logp0.conservativeResize(dist1.K-1);
			dist1.nu.conservativeResize(dist1.K-1);
			dist1.a.conservativeResize(dist1.K-1);
			dist1.b.conservativeResize(dist1.K-1);
			dist1.eta.conservativeResize(dist1.K-1, Eigen::NoChange);
			dist1.K--;
			k--;
		}
	}
	if(dist1.K == 0){//if removing empty clusters destroyed all of them, just quit
		return;
	}

	//lock mutex, store the local trace, merge the minibatch distribution, unlock
	typename VarDP<Model>::Distribution dist2; //dist2 is used to check if a matching was solved later
	{
		std::lock_guard<std::mutex> lock(distmut);
		dist2 = dist;
		//add the result of the job to the multitrace
		mtrace.localstarttimes.push_back(t0);
		mtrace.localtimes.push_back(tr.times);
		mtrace.localobjs.push_back(tr.objs);
		mtrace.localtestlls.push_back(tr.testlls);
		//merge
		t0 = timer.get(); //reuse t0 -- already stored it above
		dist = mergeDistributions(dist1, dist, dist0);
		mtrace.localmergetimes.push_back(timer.get()-t0);
	} //release the lock

	//compute the test log likelihood of the global model
	t0 = timer.get();
	double testll = computeTestLogLikelihood(); //this function locks dist internally where required
	//lock mutex, update  with matching, unlock
	{
		std::lock_guard<std::mutex> lock(distmut);
		mtrace.globaltimes.push_back(t0);
		mtrace.globaltestlls.push_back(testll);
		mtrace.globalclusters.push_back(dist.eta.rows());
		if (mtrace.globalmatchings.size() == 0){
			mtrace.globalmatchings.push_back(0); // the first merge never needs to do a matching since all components are new
		} else {
			uint32_t nm = mtrace.globalmatchings.back();
			if (dist1.K > dist0.K && dist2.K > dist0.K){
				mtrace.globalmatchings.push_back(nm+1);
			} else {
				mtrace.globalmatchings.push_back(nm);
			}
		}
	} //release the lock

	//done!
	return;
}


template<class Model>
typename VarDP<Model>::Distribution SDADP<Model>::mergeDistributions(typename VarDP<Model>::Distribution src, typename VarDP<Model>::Distribution dest, typename VarDP<Model>::Distribution prior){
	uint32_t Kp = prior.K;
	uint32_t Ks = src.K;
	uint32_t Kd = dest.K;
	uint32_t M = dest.eta.cols();
	typename VarDP<Model>::Distribution out;
	assert(Kd >= Kp && Ks >= Kp);
	if (Ks == Kp){
		//no new components created; just do the merge directly
		//match the first Ks elements (one for each src component) to the dest
		out = dest;
		out.eta.block(0, 0, Ks, M) += src.eta - prior.eta;
		out.nu.head(Ks) += src.nu - prior.nu;
		out.sumz.head(Ks) += src.sumz;
		out.logp0.head(Ks) += src.logp0;
		for (uint32_t k = 0; k < Kd; k++){
			out.a(k) = 1.0 + out.sumz(k);
			out.b(k) = alpha;
			for (uint32_t j = k+1; j < Kd; j++){
				out.b(k) += out.sumz(j);
			}
		}
	} else if (Kd == Kp) {
		//new components were created in src, but dest is still the same size as prior
		//just do the merge directly from dest into src
		out = src;
		if (Kp > 0){
			out.eta.block(0, 0, Kd, M) += dest.eta - prior.eta;
			out.nu.head(Kd) += dest.nu - prior.nu;
			out.sumz.head(Kd) += dest.sumz-prior.sumz;
			out.logp0.head(Kd) += dest.logp0-prior.sumz;
		}
		for (uint32_t k = 0; k < Ks; k++){
			out.a(k) = 1.0 + out.sumz(k);
			out.b(k) = alpha;
			for (uint32_t j = k+1; j < Ks; j++){
				out.b(k) += out.sumz(j);
			}
		}
	} else {
		uint32_t Ksp = Ks-Kp;
		uint32_t Kdp = Kd-Kp;
		//new components were created in both dest and src -- need to solve a matching
		MXd costs = MXd::Zero(Ksp+Kdp, Ksp+Kdp);
		MXi costsi = MXi::Zero(Ksp+Kdp, Ksp+Kdp);

		//get logp0 and Enk for d1 and d2
		VXd logp0s = src.logp0.tail(Ksp);
		VXd logp0d = dest.logp0.tail(Kdp);
		VXd Enks = src.sumz.tail(Ksp);
		VXd Enkd = dest.sumz.tail(Kdp);

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
		if (Kp > 0){ //if dest + src both have elements, but their common prior is empty this can happen
			out.eta.block(0, 0, Kp, M) += src.eta.block(0, 0, Kp, M) - prior.eta;
			out.nu.head(Kp) += src.nu.head(Kp) - prior.nu;
			out.sumz.head(Kp) += src.sumz.head(Kp)-prior.sumz;
			out.logp0.head(Kp) += src.logp0.head(Kp)-prior.logp0;
		}

		//merge the last Ksp elements using the matchings
		for (uint32_t i = Kp; i < Ks; i++){
			uint32_t toIdx = Kp+matchings[i-Kp];
			if (toIdx < Kd){
				out.eta.row(toIdx) += src.eta.row(i) - model.getEta0().transpose();
				out.nu(toIdx) += src.nu(i) - model.getNu0();
				out.sumz(toIdx) += src.sumz(i);
				out.logp0(toIdx) += src.logp0(i);
			} else {
				out.eta.conservativeResize(out.K+1, Eigen::NoChange);
				out.nu.conservativeResize(out.K+1);
				out.sumz.conservativeResize(out.K+1);
				out.logp0.conservativeResize(out.K+1);
				out.K++;
				out.eta.row(out.K-1) = src.eta.row(i);
				out.nu(out.K-1) = src.nu(i);
				out.sumz(out.K-1) = src.sumz(i);
				out.logp0(out.K-1) = src.logp0(i);
			}
		}
		out.a.resize(out.K);
		out.b.resize(out.K);
		for (uint32_t k = 0; k < out.K; k++){
			out.a(k) = 1.0 + out.sumz(k);
			out.b(k) = alpha;
			for (uint32_t j = k+1; j < out.K; j++){
				out.b(k) += out.sumz(j);
			}
		}
	}
	return out;
}

#define __SDADP_IMPL_HPP
#endif /* __SDADP_IMPL_HPP */
