#ifndef __SDA_IMPL_HPP

template<typename Alg>
SDA<Alg>::SDA(){
}

template<typename Alg>
SDA<Alg>::~SDA(){
}

template<typename Alg>
SDA<Alg>::run(uint32_t nThr){
	stop = false;
	for(uint32_t t = 0; t < nThr; t++){
		workers.push_back(std::thread(&SDA<Alg>::worker, this));
	}
}


template<typename Alg>
void SDA<Alg>::worker(){
	while(true){
		{
			std::lock_guard<std::mutex> lock(queue_mutex);
			while(!stop && jobs.empty()){
				queue_cond.wait(lock);
			}
			if (stop){
				lock.unlock();
				return;
			}

			job = jobs.front();
			jobs.pop_front();
		}//release the lock_guard

		//do the job
	}
}
#define __SDA_IMPL_HPP
#endif /* __SDA_IMPL_HPP */
