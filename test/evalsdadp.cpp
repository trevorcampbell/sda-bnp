#include <sdabnp/sda_dp_mixture>
#include <sdabnp/dp_mixture>
#include <sdabnp/model/normal_inverse_wishart>
#include <Eigen/Dense>
#include <random>
#include <iomanip>
#include <iostream>
#include <sstream>

//extern "C"{
#include "include/vb.h"
#include "include/vbfuncs_gaussian.h"
#include "include/costfcn.h"
#include "include/updates.h"
//}

typedef Eigen::MatrixXd MXd;
typedef Eigen::VectorXd VXd;

//This code does a comprehensive evaluation of SDA BNP on the DP Gaussian mixture

int main(int argc, char** argv){
	//constants
	uint32_t K = 40;
	uint32_t Knew = 40;
	uint32_t N = 10000;
	uint32_t Nmini = 100;
	uint32_t Nt = 1000;
	uint32_t D = 2;
	double alpha = 1.0;
	uint32_t monteCarloTrials = 20;
	std::vector<uint32_t> Nthr;
	Nthr.push_back(1);
	Nthr.push_back(2);
	Nthr.push_back(4);
	Nthr.push_back(8);
	//Nthr.push_back(16);
	//Nthr.push_back(32);
	VXd mu0 = VXd::Zero(D);
	MXd psi0 = MXd::Identity(D, D);
	double kappa0 = 1e-6;
	double xi0 = D+2;

	double minMu = -100.0, maxMu = 100.0;
	double sigMagnitude = 2.0;
	double pi0 = 0.0;

	std::mt19937 rng;
	std::random_device rd;
	rng.seed(rd());
	std::uniform_real_distribution<> unir;


	for(uint32_t nMC = 0; nMC < monteCarloTrials; nMC++){
		std::cout << "Run " << nMC+1 << "/" << monteCarloTrials << std::endl;
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
		std::ostringstream oss1;
		oss1 << "model-" << std::setw(3) << std::setfill('0')<< nMC << ".log";
		std::ofstream mout(oss1.str().c_str());
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
		std::ostringstream oss2, oss3;
		oss2  << "train-" << std::setfill('0') << std::setw(3) << nMC << ".log";
		oss3  << "test-"  << std::setfill('0') << std::setw(3) << nMC << ".log";
		std::ofstream trout(oss2.str().c_str());
		std::ofstream teout(oss3.str().c_str());
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


		//SDA DP Test:
		NIWModel niw(mu0, kappa0, psi0, xi0);
		for (uint32_t i = 0; i < Nthr.size(); i++){
			std::cout << "Running VarDP with " << Nthr[i] << " threads..." << std::endl;
			SDADP<NIWModel> sdadp(test_data, niw, alpha, Knew, Nthr[i]);
			uint32_t Nctr = 0;
			while(Nctr < N){
				std::vector<VXd> minibatch;
				minibatch.insert(minibatch.begin(), train_data.begin()+Nctr, train_data.begin()+Nctr+Nmini);
				sdadp.submitMinibatch(minibatch);
				Nctr += Nmini;
			}
			sdadp.waitUntilDone();
			std::cout << "Saving output..." << std::endl;
			std::ostringstream oss;
			oss  << "sdadpmix-nThr_" << std::setfill('0') << std::setw(3) << Nthr[i] << "-" << std::setfill('0') << std::setw(3) << nMC;
			sdadp.getDistribution().save(oss.str().c_str());
			sdadp.getTrace().save(oss.str().c_str());
		}

		//BATCH DP (new) TEST:
		std::cout << "Running Batch VarDP ..." << std::endl;
		VarDP<NIWModel> vardp(train_data, test_data, niw, alpha, K);
		vardp.run(true);
		std::cout << "Saving output..." << std::endl;
		std::ostringstream oss4;
		oss4  << "vardpmix-" << std::setfill('0') << std::setw(3) << nMC;
		vardp.getDistribution().save(oss4.str().c_str());
		vardp.getTrace().save(oss4.str().c_str());


		////Convert the parameters/data/etc to the old c code format 
		//MXd x(D, N), xt(D, Nt);
		//for (uint32_t i = 0; i < N; i++){
		//	x.col(i) = train_data[i];
		//}
		//for (uint32_t i = 0; i < Nt; i++){
		//	xt.col(i) = test_data[i];
		//}
		////get the prior in the required format
		//uint32_t M = D*D+D+1;
		//VXd eta0 = VXd::Zero(M);
		//for (uint32_t i = 0; i < D; i++){
		//	for (uint32_t j = 0; j < D; j++){
		//		eta0(i*D+j) = psi0(i, j) + kappa0*mu0(i)*mu0(j);
		//	}
		//}
		//for (uint32_t i = 0; i < D; i++){
		//	eta0(D*D+i) = kappa0*mu0(i);
		//}
		//eta0(D*D+D) = xi0+D+2;
		//double nu0 = kappa0;
		//uint32_t Kf, Ntll;
		//double *zeta, *eta, *nu, *a, *b, *times, *testlls;

		////BATCH DP (old) TEST
		//std::cout << "Running Old Batch VarDP ..." << std::endl;
		//varDP(&zeta, &eta, &nu, &a, &b, &Kf, &times, &testlls, &Ntll,
		//    x.data(), xt.data(), alpha, eta0.data(), nu0, &getLogHGaussian,
		//    &getStatGaussian, &getLogPostPredGaussian, N, Nt, M, D, K); 
		////output results
		//std::ostringstream oss5;
		//oss5 << "vardpmixold-" << std::setfill('0') << std::setw(3) << nMC << "-trace.log";
		//std::ofstream fout1(oss5.str().c_str());
		//for (uint32_t i = 0; i < Ntll; i++){
		//	fout1 << times[i] << " " << testlls[i] << std::endl;
		//}
		//fout1.close();
		//free(eta); free(nu); free(a); free(b); free(zeta); free(times); free(testlls);

		////SVI DP TEST
		//std::cout << "Running SVI ..." << std::endl;
    	//soVBDP(&zeta, &eta, &nu, &a, &b, &Kf,  &times, &testlls, &Ntll,
    	//    x.data(), xt.data(), alpha, eta0.data(), nu0, &getLogHGaussian,
    	//    &getStatGaussian,&getLogPostPredGaussian, N, Nt, M, D, K, Nmini); 
		////output results
		//std::ostringstream oss6;
		//oss6 << "svidpmix-" << std::setfill('0') << std::setw(3) << nMC << "-trace.log";
		//std::ofstream fout2(oss6.str().c_str());
		//for (uint32_t i = 0; i < Ntll; i++){
		//	fout2 << times[i] << " " << testlls[i] << std::endl;
		//}
		//fout2.close();
		//free(eta); free(nu); free(a); free(b); free(zeta); free(times); free(testlls);

		////moVB DP TEST
		//std::cout << "Running moVB ..." << std::endl;
    	//moVBDP(&zeta, &eta, &nu, &a, &b, &Kf, &times, &testlls, &Ntll,
    	//    x.data(), xt.data(), alpha, eta0.data(), nu0, &getLogHGaussian,
    	//    &getStatGaussian,&getLogPostPredGaussian, N, Nt, M, D, K, Nmini); 
		////output results
		//std::ostringstream oss7;
		//oss7 << "movbdpmix-" << std::setfill('0') << std::setw(3) << nMC << "-trace.log";
		//std::ofstream fout3(oss7.str().c_str());
		//for (uint32_t i = 0; i < Ntll; i++){
		//	fout3 << times[i] << " " << testlls[i] << std::endl;
		//}
		//fout3.close();
		//free(eta); free(nu); free(a); free(b); free(zeta); free(times); free(testlls);

		////SVA DP TEST
		//std::cout << "Running SVA ..." << std::endl;
		//svaDP(&zeta, &eta, &nu, &a, &b, &Kf, &times, &testlls, &Ntll,
		//    x.data(), xt.data(), alpha, 1.0e-1, 1.0e-2, eta0.data(), nu0, &getLogHGaussian,
		//    &getStatGaussian, &getLogPostPredGaussian, N, Nt, M, D, K); 
		////output results
		//std::ostringstream oss8;
		//oss8 << "svadpmix-" << std::setfill('0') << std::setw(3) << nMC << "-trace.log";
		//std::ofstream fout4(oss8.str().c_str());
		//for (uint32_t i = 0; i < Ntll; i++){
		//	fout4 << times[i] << " " << testlls[i] << std::endl;
		//}
		//fout4.close();
		//free(eta); free(nu); free(a); free(b); free(zeta); free(times); free(testlls);
	}
	return 0;
}
