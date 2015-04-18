#ifndef __POOL_HPP

#include<vector>
#include<thread>
#include<mutex>
#include<condition_variable>

template<typename Job>
class Pool{
	public:
		Pool(uint32_t nThr);
		~Pool();
		void submit(Job job);
	private:
		std::vector< std::thread > workers;
		void worker();
		bool stop;

		std::deque<Job> jobs;
		std::mutex queue_mutex;
		std::condition_variable queue_cond;
};


template<typename Job>
Pool<Job>::Pool(uint32_t nThr){
	stop = false;
	for(uint32_t t = 0; t < nThr; t++){
		workers.push_back(std::thread(&Pool<Job>::worker, this));
	}
}

template<typename Job>
Pool<Job>::~Pool(){
	stop = true;
	queue_cond.notify_all();
	for(uint32_t i = 0; i < workers.size(); i++){
		workers[i].join();
	}
}

template<typename Job>
void Pool<Job>::submit(Job job){
	{
		std::lock_guard<std::mutex> lock(queue_mutex);
		jobs.push_back(job);
	} // release the lock_guard
	queue_cond.notify_one();
}


template<typename Job>
void Pool<Job>::worker(){
	while(true){
		{
			std::lock_guard<std::mutex> lock(queue_mutex);
			while(!stop && jobs.empty()){
				queue_cond.wait(lock);
			}
			if (stop){
				//lock.unlock();
				return; //releases the lock_guard automatically upon destruction
			}

			job = jobs.front();
			jobs.pop_front();
		}//release the lock_guard

		job();//do the job
	}
}
#define __POOL_HPP
#endif /* __POOL_HPP */
