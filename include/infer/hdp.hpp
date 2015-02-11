#ifndef __HDP_HPP
#include <vector>
#include <algorithm>
#include <string>
#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <boost/filesystem.hpp>
#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <random>
#include <sdabnp/util/timer.hpp>

typedef Eigen::VectorXd VXd;
typedef Eigen::MatrixXd MXd;
double boost_lbeta(double a, double b);
using boost::math::digamma;
using boost::math::lgamma;

class VarHDPResults{
	public:
		//todo fill in
		std::vector<double> times, objs, testlls;
		void save(std::string filename);
};

template<class Model>
class VarHDP{
	public:
		VarHDP();
		void run(bool computeTestLL = false, double tol = 1e-6);
		VarHDPResults getResults();

	private:
		void initWeightsParams();
		void updateWeightDist();
		void updateLabelDist();
		void updateParamDist();
		double computeObjective();
		double computeTestLogLikelihood();




		std::mt19937 rng;
		double gam, alpha, eta; //gamma = global concentration, alpha = local concentration, eta = prior dirichlet topic
		uint32_t N, T, K, W; // N is nubmer of docs, T is global truncation, K is local truncation, W is vocabulary size
		MXd beta;//dirichlet variational parameters for topics
		VXd u, v;//beta variational parameters for global sticks


		self.m_var_sticks_ss = np.zeros(T)
        self.m_var_logp0_ss = np.zeros(T)
        self.m_var_beta_ss = np.zeros((T, size_vocab))


};


#include "hdp_impl.hpp"
#define __HDP_HPP
#endif /* __HDP_HPP */
