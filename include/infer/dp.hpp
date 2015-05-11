#ifndef __DP_HPP
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
#include <sdabnp/util/kmpp.hpp>
#include <sdabnp/util/trace.hpp>

typedef Eigen::VectorXd VXd;
typedef Eigen::MatrixXd MXd;
double boost_lbeta(double a, double b);
using boost::math::digamma;
using boost::math::lgamma;

template<class Model>
class VarDP{
	public:
		class Distribution{
			public:
				uint32_t K;
				MXd eta;
				VXd sumz, logp0;
				VXd a, b, nu;
				void save(std::string name);
		};
		VarDP(const std::vector<VXd>& train_data, const std::vector<VXd>& test_data, const Model& model, double alpha, uint32_t K);
		VarDP(const std::vector<VXd>& train_data, const std::vector<VXd>& test_data, const Distribution& prior, const Model& model, double alpha, uint32_t K);
		void run(bool computeTestLL, double tol = 1e-6, uint32_t nItr = 0);
		Distribution getDistribution();
		Trace getTrace();

	private:
		void init();
		void updateWeightDist();
		void updateLabelDist();
		void updateParamDist();
		double computeObjective();
		double computeTestLogLikelihood();

		std::mt19937 rng;

		double alpha;
		uint32_t K0, K, M, N, Nt; //K0 is number of nonstandard prior components, K is the # components in the model, M is the dimension of the statistic
		Model model;
		MXd zeta, sumzetaT, dlogh_deta, eta, train_stats;
		MXd test_mxd;
		VXd a, b, psisum, nu, logh, dlogh_dnu, sumzeta;
		VXd a0, b0, nu0, logh0;
		MXd eta0;
		Trace trace;
};

#include "dp_impl.hpp"

#define __DP_HPP
#endif /* __DP_HPP */
