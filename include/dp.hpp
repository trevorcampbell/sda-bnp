#ifndef __DP_HPP
#include <vector>
#include <string>
#include <Eigen/Dense>
#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <random>

typedef Eigen::VectorXd VXd;
typedef Eigen::MatrixXd MXd;
typedef boost::math::digamma boost_psi;
typedef boost::math::lgamma boost_lgamma;
double boost_lbeta(double a, double b);

template<class Model>
class VarDP{
	public:
		VarDP(const std::vector<VXd>& train_data, const std::vector<VXd>& test_data, const Model& model, uint32_t K, std::string results_folder);
		void run(bool computeTestLL = false, double tol = 1e-6);
		void getResults(std::vector<double>& times, std::vector<double>& objs, std::vector<double>& testlls);
	private:
		void initWeightsParams();
		void updateWeightDist();
		void updateLabelDist();
		void updateParamDist();
		double computeObjective();
		double computeTestLL();

		std::mt19937 rng;

		uint32_t K;
		Model model;
		MXd zeta, sumzeta, sumzetaT, train_stats, test_stats;
		MXd a, b, psisum, dlogh_deta, dlogh_dnu, nu, logh, eta;
		std::string results_folder;
		std::vector<double> times, objs, testlls;
};

#define __DP_HPP
#endif /* __DP_HPP */
