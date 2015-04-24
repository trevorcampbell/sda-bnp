#ifndef __TIMER_HPP
#include <chrono>
#include <exception>

class Timer{
	public:
		Timer();
		void start();
		double stop();
		double get();
	private:
		std::chrono::time_point<std::chrono::high_resolution_clock> t0;
		std::chrono::duration<double> elapsed;
		bool running;
};

Timer::Timer(){
	this->running = false;
	this->elapsed = std::chrono::duration<double>(0);
}

void Timer::start(){
	if (!this->running){
		this->running = true;
		t0 = std::chrono::high_resolution_clock::now();
	}
}

double Timer::stop(){
	if (this->running){
		this->running = false;
		this->elapsed += (std::chrono::high_resolution_clock::now() - this->t0);
		return std::chrono::duration_cast<std::chrono::duration<double> >(this->elapsed).count();
	} else {
		return std::chrono::duration_cast<std::chrono::duration<double> >(this->elapsed).count();
	}
}

double Timer::get(){
	if (this->running){
		return std::chrono::duration_cast<std::chrono::duration<double> >(this->elapsed + (std::chrono::high_resolution_clock::now() - this->t0)).count();
	} else {
		return std::chrono::duration_cast<std::chrono::duration<double> >(this->elapsed).count();
	}
}

#define __TIMER_HPP
#endif /* __TIMER_HPP */
