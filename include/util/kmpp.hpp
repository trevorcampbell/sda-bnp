#ifndef __KMPP_HPP
#include <Eigen/Dense>
#include <vector>
#include <random>
typedef Eigen::VectorXd VXd;
typedef Eigen::MatrixXd MXd;

std::vector<uint32_t> kmeanspp(MXd x, uint32_t K, std::mt19937& rng){
	std::uniform_int_distribution<> uniint(0, x.rows()-1);
	std::vector<uint32_t> res;
	res.push_back(uniint(rng));
	std::vector<double> mindists(std::numeric_limits<double>::infinity(), x.rows());
	for (uint32_t i =1; i < K; i++){
		for (uint32_t j = 0; j < x.rows(); j++){
			double newdist = (x.row(j)-x.row(res.back())).squaredNorm();
			mindists[j] = mindists[j] < newdist ? mindists[j] : newdist;
		}
		std::discrete_distribution<> disc(mindists.begin(), mindists.end());
		res.push_back(disc(rng));
	}
	return res;
}

#define __KMPP_HPP
#endif /* __KMPP_HPP */
