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
		//std::cout << "Starting worker " << t << std::endl;
	}
}

template<typename Job>
Pool<Job>::~Pool(){
	stop = true;
	//std::cout << "Stopped, notifying all" << std::endl;
	queue_cond.notify_all();
	for(uint32_t i = 0; i < workers.size(); i++){
		//std::cout << "Stopping worker " << i << std::endl;
		workers[i].join();
	}
}

template<typename Job>
void Pool<Job>::submit(Job job){
	{
		std::lock_guard<std::mutex> lk(queue_mutex);
		//std::cout << "Submitting job and notifying one" << std::endl;
		jobs.push(job);
	} // release the lock_guard
	queue_cond.notify_one();
}

template<typename Job>
void Pool<Job>::wait(){
	std::unique_lock<std::mutex> lk(queue_mutex);
	//std::cout << "Waiting on pool..." << std::endl;
	bool allFree = true;
	bool queueEmpty = jobs.empty();
	for (uint32_t i = 0; i < workers.size(); i++){
		allFree &= !busy[i];
	}
	while( (!allFree || !queueEmpty) && !stop){
		//std::cout << "allFree: " << allFree << " queueEmpty: " << queueEmpty << std::endl;
		//std::cout << "Waiter sleeping..." << std::endl;
		wait_cond.wait(lk);
		//std::cout << "Waiter woken up!" << std::endl;
		allFree = true;
		queueEmpty = jobs.empty();
		for (uint32_t i = 0; i < workers.size(); i++){
			allFree &= !busy[i];
		}
	}
	//std::cout << "Waiter done waiting!" << std::endl;
	lk.unlock();
}

template<typename Job>
void Pool<Job>::worker(uint32_t id){
	while(true){
		//std::cout << "Worker " << id << " trying to lock top of the loop" << std::endl;
		std::unique_lock<std::mutex> lk(queue_mutex);
		busy[id] = false;
		//std::cout << "Worker " << id << " at the top of the loop" << std::endl;
		//std::cout << "Worker " << id << ": stop: " << stop << " jobs.empty: " << jobs.empty() << std::endl;
		while(!stop && jobs.empty()){
			//std::cout << "Worker " << id << " notifying waiters that it's waiting" << std::endl;
			wait_cond.notify_all(); //if any thread is waiting for us to be done,
									//wake it up to check since I'm now going to sleep
			//std::cout << "Worker " << id << " going to sleep" << std::endl;
			queue_cond.wait(lk);
			//std::cout << "Worker " << id << " woken up!" << std::endl;
			//std::cout << "Worker " << id << ": stop: " << stop << " jobs.empty: " << jobs.empty() << std::endl;
		}
		if (stop){
			//std::cout << "Worker " << id << " was stopped! Notifying any waiters" << std::endl;
			lk.unlock();
			wait_cond.notify_all();
			return; 
		}

		//std::cout << "Worker " << id << " has a job!" << std::endl;
		Job job = jobs.front();
		jobs.pop();
		busy[id] = true;

		lk.unlock();

		job();//do the job
	}
}
#define __POOL_HPP
#endif /* __POOL_HPP */
