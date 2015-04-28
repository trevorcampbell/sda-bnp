
#include <sdabnp/util/pool.hpp>
#include <Eigen/Dense>
#include <random>

typedef Eigen::MatrixXd MXd;
typedef Eigen::VectorXd VXd;

class Foo{
	public:
		Foo() : pool(8){
			a = 0;
		}
		void incrementBy(int n){
			std::lock_guard<std::mutex> lock(m);
			//really dumb way to simulate a thread doing work
			for (uint32_t i = 0; i < n; i++){
				a += 1;
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
		Pool pool;
		std::mutex m;
		int a;
};

int main(int argc, char** argv){

	Foo f;
	int Njobs = 1000;
	std::vector<int> jobs;
	std::random_device rd;
	std::mt19937 rng.seed(rd());
	std::uniform_int_distribution<int> uintgen(0, 50000);

	int sum = 0;
	for (uint32_t i = 0; i < Njobs; i++){
		int r = uintgen(rng);
		sum += r;
		f.submit(r);
	}

	f.wait();

	std::cout << "f.get() = " << f.get() << std::endl;
	std::cout << "sum = " << sum << std::endl;

	return 0;
}
