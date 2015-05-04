#include <sdabnp/sda_dp_mixture>
#include <sdabnp/model/normal_inverse_wishart>
#include <Eigen/Dense>
#include <random>
#include <iostream>
#include <sstream>

typedef Eigen::MatrixXd MXd;
typedef Eigen::VectorXd VXd;

//This code does a comprehensive evaluation of SDA BNP on the DP Gaussian mixture

int main(int argc, char** argv){
	//constants
	uint32_t K = 3;
	uint32_t N = 1000;
	uint32_t Nmini = 100;
	uint32_t Nt = 100;
	uint32_t D = 2;
	double alpha = 1.0;
	uint32_t Knew = 3;
	std::vector<uint32_t> Nthr;
	Nthr.push_back(1);
	Nthr.push_back(2);
	Nthr.push_back(4);
	Nthr.push_back(8);
	Nthr.push_back(16);
	Nthr.push_back(32);

	double minMu = -50.0, maxMu = 50.0;
	double sigMagnitude =5.0;
	double pi0 = 0.0;

	std::mt19937 rng;
	std::random_device rd;
	rng.seed(rd());
	std::uniform_real_distribution<> unir;


	//setup the generating model
	std::vector<VXd> mus;
	std::vector<MXd> sigs;
	std::vector<MXd> sigsqrts;
	std::vector<double> pis;
	double sumpis = 0.0;
	std::cout << "Creating generative model..." << std::endl;
	for (uint32_t k = 0; k < K; k++){
		mus.push_back(VXd::Zero(D));
		sigs.push_back(MXd::Zero(D, D));
		for(uint32_t d = 0; d < D; d++){
			mus.back()(d) = (maxMu-minMu)*unir(rng) + minMu;
			for(uint32_t f = 0; f < D; f++){
				sigs.back()(d, f) = sigMagnitude*unir(rng);
			}
		}
		sigs.back() = (sigs.back().transpose()*sigs.back()).eval();//eval to stop aliasing
		sigsqrts.push_back(Eigen::LLT<MXd, Eigen::Upper>(sigs.back()).matrixL());
		pis.push_back(pi0+unir(rng));
		sumpis += pis.back();
		//std::cout << "Mu: " << mus.back().transpose() << std::endl << "Sig: " << sigs.back() << std::endl << "Wt: " << pis.back() << std::endl;
	}
	for (uint32_t k = 0; k < K; k++){
		pis[k] /= sumpis;
	}

	//output the generating model
	std::ofstream mout("model.log");
	for (uint32_t k = 0; k < K; k++){
		mout << mus[k].transpose() << " ";
		for (uint32_t j = 0; j < D; j++){
			mout << sigs[k].row(j) << " ";
		}
		mout << pis[k] << std::endl;
	}
	mout.close();


	//sample from the model
	std::vector<VXd> train_data, test_data;
	std::normal_distribution<> nrm;
	std::discrete_distribution<> disc(pis.begin(), pis.end());
	std::ofstream trout("train.log");
	std::ofstream teout("test.log");
	std::cout << "Sampling training/test data" << std::endl;
	for (uint32_t i = 0; i < N; i++){
		VXd x = VXd::Zero(D);
		for (uint32_t j = 0; j < D; j++){
			x(j) = nrm(rng);
		}
		uint32_t k = disc(rng);
		train_data.push_back(mus[k] + sigsqrts[k]*x);
		trout << train_data.back().transpose() << std::endl;
		//std::cout << train_data.back().transpose() << std::endl;
	}
	for (uint32_t i = 0; i < Nt; i++){
		VXd x = VXd::Zero(D);
		for (uint32_t j = 0; j < D; j++){
			x(j) = nrm(rng);
		}
		uint32_t k = disc(rng);
		test_data.push_back(mus[k] + sigsqrts[k]*x);
		teout << train_data.back().transpose() << std::endl;
		//std::cout << test_data.back().transpose() << std::endl;
	}
	trout.close();
	teout.close();


	VXd mu0 = VXd::Zero(D);
	MXd psi0 = MXd::Identity(D, D);
	double kappa0 = 1e-6;
	double xi0 = D+2;
	NIWModel niw(mu0, kappa0, psi0, xi0);
	for (uint32_t i = 0; i < Nthr.size(); i++){
		std::cout << "Running VarDP with " << Nthr[i] < " threads..." << std::endl;
		SDADP<NIWModel> sdadp(test_data, niw, alpha, Knew, Nthr[i]);
		uint32_t Nctr = 0;
		while(Nctr < N){
			std::vector<VXd> minibatch;
			minibatch.insert(minibatch.begin(), train_data.begin()+Nctr, train_data.begin()+Nctr+Nmini);
			sdadp.submitMinibatch(minibatch);
			Nctr += Nmini;
		}
		sdadp.waitUntilDone();
		VarDP<NIWModel>::Distribution res = sdadp.getDistribution();
		std::ostringstream oss;
		oss << "sdadpmix" << Nthr[i];
		res.save(oss.str().c_str());
	}
	return 0;
}
