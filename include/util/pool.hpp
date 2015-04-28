#ifndef __POOL_HPP

#include<vector>
#include<queue>
#include<thread>
#include<mutex>
#include<condition_variable>

template<typename Job>
class Pool{
	public:
		Pool(uint32_t nThr);
		~Pool();
		void submit(Job job);
		void wait();
	private:
		std::vector< std::thread > workers;
		std::vector<bool> busy;
		void worker(uint32_t id);
		bool stop;

		std::queue<Job> jobs;
		std::mutex queue_mutex;
		std::condition_variable queue_cond, wait_cond;
};



template<typename Job>
Pool<Job>::Pool(uint32_t nThr){
	stop = false;
	for(uint32_t t = 0; t < nThr; t++){
		busy.push_back(false);
		workers.push_back(std::thread(&Pool<Job>::worker, this, t));
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
		std::lock_guard<std::mutex> lk(queue_mutex);
		jobs.push(job);
	} // release the lock_guard
	queue_cond.notify_one();
}

template<typename Job>
void Pool<Job>::wait(){
	std::unique_lock<std::mutex> lk(queue_mutex);
	bool allFree = true;
	bool queueEmpty = jobs.empty();
	for (uint32_t i = 0; i < workers.size(); i++){
		allFree &= !busy[i];
	}
	while( (!allFree || !queueEmpty) && !stop){
		wait_cond.wait(lk);
		allFree = true;
		queueEmpty = jobs.empty();
		for (uint32_t i = 0; i < workers.size(); i++){
			allFree &= !busy[i];
		}
	}
	lk.unlock();
}

template<typename Job>
void Pool<Job>::worker(uint32_t id){
	while(true){
		std::unique_lock<std::mutex> lk(queue_mutex);
		busy[id] = false;
		while(!stop && jobs.empty()){
			wait_cond.notify_one(); //if the main thread is waiting for us to be done,
									//wake it up to check since I'm now going to sleep
			queue_cond.wait(lk);
		}
		if (stop){
			lk.unlock();
			wait_cond.notify_one();
			return; 
		}

		Job job = jobs.front();
		jobs.pop();
		busy[id] = true;

		job();//do the job
	}
}
#define __POOL_HPP
#endif /* __POOL_HPP */
