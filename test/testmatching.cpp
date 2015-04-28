
#include <sdabnp/util/matching.hpp>
#include <Eigen/Dense>
#include <random>

typedef Eigen::MatrixXi MXi;

int main(int argc, char** argv){
	int N = 1000;
	std::random_device rd;
	std::mt19937 rng.seed(rd());
	std::uniform_int_distribution<int> uintgen(0, 1000);

	MXi costs(5, 5);
	costs(0, 0) = 1; costs(0, 1) = 9; costs(0, 2) = 3; costs(0, 3) = 1; costs(0, 4) = 2;
	costs(1, 0) = 3; costs(1, 1) = 4; costs(1, 2) = 1; costs(1, 3) = 2; costs(1, 4) = 6;
	costs(2, 0) = 3; costs(2, 1) = 0; costs(2, 2) = 1; costs(2, 3) = 6; costs(2, 4) = 7;
	costs(3, 0) = 7; costs(3, 1) = 5; costs(3, 2) = 4; costs(3, 3) = 6; costs(3, 4) = 4;
	costs(4, 0) = 5; costs(4, 1) = 2; costs(4, 2) = 5; costs(4, 3) = 1; costs(4, 4) = 1;

	std::cout << "costs: " << std::endl << costs << std::endl;
	
	std::vector<int> matchings;
	int cost =  hungarian(costs, matchings);

	std::cout << "matching cost: " <<  cost << std::endl << "matching: " << std::endl;
	for (uint32_t i = 0; i < matchings.size(); i++){
		std::cout << i << " -> " << matchings[i] << std::endl;
	}
	//MXi costs(N, N);
	//for (uint32_t i = 0; i < N; i++){
	//	for (uint32_t j = 0; j < N; j++){
	//		costs(i, j) = uintgen(rng);
	//	}
	//}

	return 0;
}
