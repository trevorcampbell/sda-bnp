#ifndef __SDADP_HPP
#include <vector>
#include <algorithm>
#include <string>
#include <iostream>
#include <fstream>
#include <Eigen/Dense>
//#include <boost/filesystem.hpp>
#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/gamma.hpp>
//#include <random>
#include<thread>
#include<mutex>
#include <sdabnp/util/timer.hpp>
#include <sdabnp/util/trace.hpp>
#include <sdabnp/util/pool.hpp>

typedef Eigen::VectorXd VXd;
typedef Eigen::MatrixXd MXd;
using boost::math::lgamma;

template<class Model>
class SDADP{
	public:
		SDADP(const std::vector<VXd>& test_data, const Model& model, double alpha, uint32_t Knew);
		void start(const std::vector<std::vector<VXd> >& miniBatches, bool computeTestLL = false, double tol = 1e-6);
		void addMinibatch(const std::vector<VXd>& train_data); 
		VarDP<Model>::Distribution getDistribution();
		Trace getTrace();
	private:
		double computeObjective();
		double computeTestLogLikelihood();

		double alpha;
		uint32_t Knew;
		std::vector<VXd> test_data;
		Model model;
		VarDP<Model>::Distribution dist;
		std::mutex distmut;
		std::vector<double> times, objs, testlls;

		void varDPJob(const std::vector<VXd>& train_data);
		Pool<std::function<void()> > pool;
};


#define __SDADP_HPP
#endif /* __SDADP_HPP */
