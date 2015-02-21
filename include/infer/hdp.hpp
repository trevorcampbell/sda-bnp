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
		VarHDP(const std::vector< std::vector<VXd> >& train_data, const std::vector< std::vector<VXd> >& test_data, const Model& model, double gam, double alpha, uint32_t T, uint32_t K);
		void run(bool computeTestLL = false, double tol = 1e-6);
		VarHDPResults getResults();

	private:
		void init();

		//Local variational updates
		void updateLocalDists(double tol);
		void updateLocalWeightDist(uint32_t idx);
		void updateLocalLabelDist(uint32_t idx);
		void updateLocalCorrespondenceDist(uint32_t idx);

		//global variational updates
		void updateGlobalDist();
		void updateGlobalWeightDist();
		void updateGlobalParamDist();

		double computeFullObjective();
		double computeLocalObjective(uint32_t idx);
		double computeTestLogLikelihood();

		std::mt19937 rng;
		double gam, alpha; //gamma = global concentration, alpha = local concentration, eta = prior dirichlet topic
		uint32_t N, Nt, T, K, M; // N is number of observation collections, T is global truncation, K is local truncation, M is stat dimension
		std::vector<uint32_t> Nl, Ntl; //local number of observations in each collection

		Model model;

		MXd eta, dlogh_deta;//dirichlet variational parameters for topics
		VXd u, v, nu, logh, dlogh_dnu;//eeta variational parameters for global sticks
		MXd phizetaTsum;
		VXd phizetasum, phisum, psiuvsum;
		std::vector<VXd> a, b, psiabsum, zetasum, phiNsum;
		std::vector<MXd> phi, zeta, train_stats, zetaTsum, phiEsum; 
		std::vector<double> times, objs, testlls;
		std::vector<std::vector<VXd> > test_data;

};


#include "hdp_impl.hpp"
#define __HDP_HPP
#endif /* __HDP_HPP */
