#ifndef __SDA_HPP
#include<vector>
#include<sdabnp/util/pool.hpp>

template<typename Alg>
class SDA{
	public:
		SDA(uint32_t nThr);
		~SDA();
		void run();
	private:
		Pool tpool;
		std::mutex model_mutex;
		void minibatch_inf();
};

#define __SDA_HPP
#endif /* __SDA_HPP */
