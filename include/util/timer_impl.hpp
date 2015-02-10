
void Timer::start(){
	this->t0 = std::chrono::high_resolution_clock::now();
}

double Timer::stop(){
	this->elapsed_s += (std::chrono::high_resolution_clock::now() - this->t0);
	return this->elapsed_s.count();
}

double Timer::get(){
	return (this->elapsed_s + std::chrono::high_resolution_clock::now() - this->t0).count();
}

void Timer::reset(){
	this->elapsed_s = std::chrono::duration<double>();
}
