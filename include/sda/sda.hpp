#ifndef __SDA_HPP
#include<vector>
#include<thread>
#include<mutex>
#include<condition_variable>

template<typename Alg>
class SDA{
	public:
		SDA();
		~SDA();
		void run(uint32_t nThr);
	private:
		std::vector< std::thread > workers;
		void worker();
		bool stop;

		std::deque< job > jobs;
		std::mutex model_mutex, queue_mutex;
		std::condition_variable queue_cond;
};

#define __SDA_HPP
#endif /* __SDA_HPP */
