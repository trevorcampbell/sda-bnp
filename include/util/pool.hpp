#ifndef __POOL_HPP

#include<vector>
#include<thread>
#include<mutex>
#include<condition_variable>

template<typename Alg>
class SDA{
	public:
		SDA(uint32_t nThr);
		~SDA();
		void run();
	private:
		std::vector< std::thread > workers;
		void worker();
		bool stop;

		std::deque< job > jobs;
		std::mutex model_mutex, queue_mutex;
		std::condition_variable queue_cond;
};



template<typename Alg>
SDA<Alg>::SDA(uint32_t nThr){
	stop = false;
	for(uint32_t t = 0; t < nThr; t++){
		workers.push_back(std::thread(&SDA<Alg>::worker, this));
	}
}

template<typename Alg>
SDA<Alg>::~SDA(){
	stop = true;
	queue_cond.notify_all();
	for(uint32_t i = 0; i < workers.size(); i++){
		workers[i].join();
	}
}

template<typename Alg>
SDA<Alg>::run(){

	//TODO: add jobs to the queue
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

		//TODO: do the job
	}
}
#define __POOL_HPP
#endif /* __POOL_HPP */
