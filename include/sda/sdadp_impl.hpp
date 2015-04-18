#ifndef __SDADP_IMPL_HPP

//void submit(Job job);

template<class Model>
SDADP<Model>::SDADP(const std::vector<VXd>& test_data, const Model& model, double alpha, uint32_t Knew, uint32_t nthr):
test_data(test_data), model(model), alpha(alpha), Knew(Knew), pool(nthr){

}

template<class Model>
void SDADP<Model>::start(const std::vector<std::vector<VXd> >& miniBatches, bool computeTestLL = false, double tol = 1e-6){

}

template<class Model>
void SDADP<Model>::addMinibatch(const std::vector<VXd>& train_data){
	pool.submit(std::bind(&SDADP<Model>::varDPJob, this, train_data));
}

template<class Model>
VarDP<Model>::Distribution SDADP<Model>::getDistribution(){
	return dist;
}

template<class Model>
Trace SDADP<Model>::getTrace(){
	Trace tr;
	tr.times = this->times;
	tr.objs = this->objs;
	tr.testlls = this->testlls;
	return tr;
}

template<class Model>
double SDADP<Model>::computeObjective(){

}

template<class Model>
double SDADP<Model>::computeTestLogLikelihood(){

}

template<class Model>
void SDADP<Model>::varDPJob(const std::vector<VXd>& train_data){

	//lock the mutex, get the distribution, unlock
	VarDP<Model>::Distribution dist0;
	{
		std::lock_guard<std::mutex> lock(distmut);
		dist0 = dist;
	} //release the lock

	//do inference
	VarDP<Model> vdp();
	VarDP<Model>::Distribution dist1 = vdp.getDistribution();
	VarDP<Model>::Distribution dist2;
	//lock mutex, update  with matching, unlock
	{
		std::lock_guard<std::mutex> lock(distmut);
		dist2 = dist;
	} //release the lock


	//solve matching between dist1 nd dist2 with prior dist0
	


	//update the global distribution
	{
		std::lock_guard<std::mutex> lock(distmut);
		dist2 = dist;
	} //release the lock
}


#define __SDADP_IMPL_HPP
#endif /* __SDADP_IMPL_HPP */
