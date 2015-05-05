#include <iostream>
#include <random>
#include <sdabnp/util/pool.hpp>


class Foo{
	public:
		Foo() : pool(8){
			a = 0;
		}
		void incrementBy(int n){
			std::unique_lock<std::mutex> lock(m);
			lock.unlock();
			for (uint32_t i = 0; i < n; i++){
				lock.lock();
				a += 1;
				std::cout << "Incremented a: " << i+1 << "/" << n << std::endl;
				lock.unlock();
				std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(1000));
			}
		}
		void submit(int n){
			pool.submit(std::bind(&Foo::incrementBy, this, n));
		}
		void waitUntilDone(){
			pool.wait();
		}
		int get(){
			std::lock_guard<std::mutex> lock(m);
			return a;
		}
	private:
		Pool<std::function<void ()> > pool;
		std::mutex m;
		int a;
};

int main(int argc, char** argv){

	Foo f;
	int Njobs = 10;
	std::vector<int> jobs;
	std::random_device rd;
	std::mt19937 rng;
	rng.seed(rd());
	std::uniform_int_distribution<int> uintgen(0, 10);

	int sum = 0;
	for (uint32_t i = 0; i < Njobs; i++){
		int r = uintgen(rng);
		sum += r;
		f.submit(r);
	}

	f.waitUntilDone();

	std::cout << "f.get() = " << f.get() << std::endl;
	std::cout << "sum = " << sum << std::endl;

	return 0;
}
