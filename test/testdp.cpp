#include <sdabnp/dp_mixture>
#include <sdabnp/model/normal_inverse_wishart>
#include <Eigen/Dense>
#include <random>

typedef Eigen::MatrixXd MXd;
typedef Eigen::VectorXd VXd;

int main(int argc, char** argv){
	//constants
	uint32_t K = 3;
	uint32_t N = 100;
	uint32_t Nt = 100;
	uint32_t D = 3;

	std::mt19937 rng;
	std::random_device rd;
	if (rd.entropy() == 0){
		std::cout << "WARNING: USING 0 ENTROPY RANDOM DEVICE FOR SEED..." << std::endl;
	}
	rng.seed(rd());


	//setup the generating model
	std::vector<VXd> mus;
	std::vector<MXd> sigs;
	std::vector<MXd> sigsqrts;
	std::vector<double> pis;
	VXd pisv = VXd::Random(K);
	double sumpisv = 0;
	for (uint32_t k = 0; k < K; k++){
		mus.push_back(10.0*VXd::Random(D));
		MXd m = MXd::Random(D, D);
		sigs.push_back(m.transpose()*m);
		sigsqrts.push_back(Eigen::LLT<MXd, Eigen::Upper>(sigs.back()).matrixL());
		pisv(k) += 1.0;
		sumpisv += pisv(k);
	}
	for (uint32_t k = 0; k < K; k++){
		pis.push_back(pisv(k)/sumpisv);
	}


	//sample from the model
	std::vector<VXd> train_data, test_data;
	std::normal_distribution<> nrm;
	std::discrete_distribution<> disc(pis.begin(), pis.end());
	for (uint32_t i = 0; i < N; i++){
		VXd x = VXd::Zero(D);
		for (uint32_t j = 0; j < D; j++){
			x(j) = nrm(rng);
		}
		uint32_t k = disc(rng);
		train_data.push_back(mus[k] + sigsqrts[k]*x);
	}
	for (uint32_t i = 0; i < Nt; i++){
		VXd x = VXd::Zero(D);
		for (uint32_t j = 0; j < D; j++){
			x(j) = nrm(rng);
		}
		uint32_t k = disc(rng);
		test_data.push_back(mus[k] + sigsqrts[k]*x);
	}


	VXd mu0 = VXd::Zero(D);
	MXd psi0 = MXd::Identity(D, D);
	double kappa0 = 1.0;
	double xi0 = D+2;
	NIWModel niw(mu0, kappa0, psi0, xi0);

	VarDP<NIWModel> dp(train_data, test_data, niw, 1.0, K);
	dp.run(true);
	VarDPResults res = dp.getResults();

	return 0;
}
