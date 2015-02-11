#ifndef __TIMER_HPP
#include <chrono>

class Timer{
	public:
		Timer();
		void start();
		double stop();
		double get();
	private:
		std::chrono::time_point<std::chrono::high_resolution_clock> t0;
		std::chrono::duration<double> elapsed;
};

#include "timer_impl.hpp"
#define __TIMER_HPP
#endif /* __TIMER_HPP */
