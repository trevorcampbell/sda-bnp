#ifndef __DP_HPP
#include <vector>
#include <string>
#include <Eigen/Dense>

typedef Eigen::VectorXd VXd;
typedef Eigen::MatrixXd MXd;

template<class Model>
class VarDP{
	public:
		VarDP(const std::vector<VXd>& train_data, const std::vector<VXd>& test_data, const Model& prior, uint32_t K, std::string results_folder);
		void run(bool computeTestLL = false, double tol = 1e-6);
		void getResults(std::vector<double>& times, std::vector<double>& objs, std::vector<double>& testlls);
	private:
		void updateWeightDist();
		void updateLabelDist();
		void updateParamDist();
		double computeObjective();
		double computeTestLL();

		MXd zeta, sumzetaT, train_stats, test_stats;
		MXd a, b;
		Model prior;
		std::string results_folder;
		std::vector<double> times, objs, testlls;
};

#define __DP_HPP
#endif /* __DP_HPP */
