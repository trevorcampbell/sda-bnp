#ifndef __KMPP_HPP
#include <Eigen/Dense>
#include <vector>
#include <random>
#include <limits>
#include <cassert>
typedef Eigen::VectorXd VXd;
typedef Eigen::MatrixXd MXd;

//Dsq is a function that takes two eigen vectors as outputs their "distance" squared
//"distance" could be a bregman divergence sq, or euclidean dist squared, etc
template<typename Func>
std::vector<uint32_t> kmeanspp(MXd x, Func Dsq, uint32_t K, MXd c0, uint32_t K0, std::mt19937& rng){
	assert (K>= K0);
	std::vector<uint32_t> res;
	std::vector<double> mindists(x.rows(), std::numeric_limits<double>::infinity());
	std::vector<double> sampledDists(K);

	//if K = K0, just return the empty set since we have everything already initialized
	if (K == K0){
		return res;
	}
	if (K0 == 0){
		//K != K0 and K0 = 0 -- just initialize the first center to uniform random
		std::uniform_int_distribution<> uniint(0, x.rows()-1);
		res.push_back(uniint(rng));
		//update the min dists
		for (uint32_t j = 0; j < x.rows(); j++){
			VXd x1 = x.row(j).transpose();
			VXd x2 = x.row(res.back()).transpose();
			double newdist = Dsq(x1, x2);
			mindists[j] = mindists[j] < newdist ? mindists[j] : newdist;
		}
	} else {
		//Otherwise K0 != 0 -- just compute the distances to all the K0 old centers first
		for (uint32_t i = 0; i < K0; i++){
			VXd x2 = c0.row(i).transpose();
			for (uint32_t j = 0; j < x.rows(); j++){
				VXd x1 = x.row(j).transpose();
				double newdist = Dsq(x1, x2);
				mindists[j] = mindists[j] < newdist ? mindists[j] : newdist;
			}
		}
	}
	//for each new cluster, sample a center and update min dists
	for (uint32_t i =K0; i < K; i++){
		std::discrete_distribution<> disc(mindists.begin(), mindists.end());
		res.push_back(disc(rng));
		for (uint32_t j = 0; j < x.rows(); j++){
			VXd x1 = x.row(j).transpose();
			VXd x2 = x.row(res.back()).transpose();
			double newdist = Dsq(x1, x2);
			mindists[j] = mindists[j] < newdist ? mindists[j] : newdist;
		}
	}
	return res;
}

#define __KMPP_HPP
#endif /* __KMPP_HPP */
