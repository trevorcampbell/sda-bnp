#ifndef __SDADP_HPP
#include <vector>
#include <algorithm>
#include <string>
#include <iostream>
#include <fstream>
#include <Eigen/Dense>
//#include <boost/filesystem.hpp>
//#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/gamma.hpp>
//#include <random>
#include<thread>
#include<mutex>
#include <sdabnp/infer/dp.hpp>
#include <sdabnp/util/timer.hpp>
#include <sdabnp/util/matching.hpp>
#include <sdabnp/util/trace.hpp>
#include <sdabnp/util/pool.hpp>
#include <cassert>

typedef Eigen::VectorXd VXd;
typedef Eigen::MatrixXd MXd;
typedef Eigen::MatrixXi MXi;
using boost::math::lgamma;

template<class Model>
class SDADP{
	public:
		SDADP(const std::vector<VXd>& test_data, const Model& model, double alpha, uint32_t Knew, uint32_t nthr);
		void submitMinibatch(const std::vector<VXd>& train_data); 
		void waitUntilDone();
		typename VarDP<Model>::Distribution getDistribution();
		MultiTrace getTrace();
	private:
		double computeTestLogLikelihood(typename VarDP<Model>::Distribution dist0);
		typename VarDP<Model>::Distribution mergeDistributions(typename VarDP<Model>::Distribution d1, typename VarDP<Model>::Distribution d2, typename VarDP<Model>::Distribution d0);

		Timer timer;
		double alpha;
		uint32_t Knew;
		Model model;
		typename VarDP<Model>::Distribution dist;
		std::mutex distmut;
		MultiTrace mtrace;
		MXd test_mxd;
		std::vector<VXd> test_data;

		void varDPJob(const std::vector<VXd>& train_data);
		Pool<std::function<void()> > pool;
};

#include "sdadp_impl.hpp"

#define __SDADP_HPP
#endif /* __SDADP_HPP */
