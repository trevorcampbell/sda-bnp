#include <sdabnp/sda_dp_mixture>
#include <sdabnp/dp_mixture>
#include <sdabnp/model/normal_inverse_wishart>
#include <Eigen/Dense>
#include <random>
#include <string>
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

std::vector<VXd> readVectorRows(const char* filename){
	std::ifstream infile;
	infile.open(filename);
	uint32_t vecSize = 0;
	std::vector<VXd> res;
	while(! infile.eof()){
		std::string line;
		std::getline(infile, line);
		std::stringstream stream(line);
		std::vector<VXd> vvec;
		while(!stream.eof()){
			double dd;
			stream >> dd;
			vvec.push_back(dd);
		}
		if (vecSize == 0){
			vecSize = vvec.size();
		}
		if (vecSize != vvec.size()){
			std::cout << "ERROR LOADING FILE " << filename << ": columns are not uniform length" << std::endl;
		}
		VXd vec = VXd::Zero(vecSize);
		for (uint32_t j = 0; j < vecSize; j++){
			vec(j) = vvec[j];
		}
		res.push_back(vec);
	}
	infile.close();
	return res;
}


int main(int argc, char** argv){
	//load the train/test data
	std::vector<VXd> train_data = readVectorRows("mnistTrainData-20D.csv");
	std::vector<VXd> test_data = readVectorRows("mnistTestData-20D.csv");
	std::cout << "Loaded " << train_data.size() << " training vecs and " << test_data.size() << " test vecs." << std::endl;

	//constants
	uint32_t Knew = 40;
	uint32_t Nmini = 100;
	double alpha = 1.0;
	std::vector<uint32_t> Nthr;
	Nthr.push_back(1);
	Nthr.push_back(2);
	Nthr.push_back(4);
	Nthr.push_back(8);
	Nthr.push_back(16);
	Nthr.push_back(24);
	Nthr.push_back(32);
	VXd mu0 = VXd::Zero(D);
	MXd psi0 = MXd::Identity(D, D);
	double kappa0 = 1e-6;
	double xi0 = D+2;

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
		oss  << "sdadpmix-nThr_" << std::setfill('0') << std::setw(3) << Nthr[i];
		sdadp.getDistribution().save(oss.str().c_str());
		sdadp.getTrace().save(oss.str().c_str());
	}

	//Convert the parameters/data/etc to the old c code format 
	MXd x(D, N), xt(D, Nt);
	for (uint32_t i = 0; i < N; i++){
		x.col(i) = train_data[i];
	}
	for (uint32_t i = 0; i < Nt; i++){
		xt.col(i) = test_data[i];
	}
	//get the prior in the required format
	uint32_t M = D*D+D+1;
	VXd eta0 = VXd::Zero(M);
	for (uint32_t i = 0; i < D; i++){
		for (uint32_t j = 0; j < D; j++){
			eta0(i*D+j) = psi0(i, j) + kappa0*mu0(i)*mu0(j);
		}
	}
	for (uint32_t i = 0; i < D; i++){
		eta0(D*D+i) = kappa0*mu0(i);
	}
	eta0(D*D+D) = xi0+D+2;
	double nu0 = kappa0;
	uint32_t Kf, Ntll;
	double *zeta, *eta, *nu, *a, *b, *times, *testlls;

	//BATCH DP (old) TEST
	std::cout << "Running Old Batch VarDP ..." << std::endl;
	varDP(&zeta, &eta, &nu, &a, &b, &Kf, &times, &testlls, &Ntll,
	    x.data(), xt.data(), alpha, eta0.data(), nu0, &getLogHGaussian,
	    &getStatGaussian, &getLogPostPredGaussian, N, Nt, M, D, K); 
	//output results
	std::ostringstream oss5;
	oss5 << "vardpmixold-trace.log";
	std::ofstream fout1(oss5.str().c_str());
	for (uint32_t i = 0; i < Ntll; i++){
		fout1 << times[i] << " " << testlls[i] << std::endl;
	}
	fout1.close();
	free(eta); free(nu); free(a); free(b); free(zeta); free(times); free(testlls);

	//SVI DP TEST
	std::cout << "Running SVI ..." << std::endl;
    soVBDP(&zeta, &eta, &nu, &a, &b, &Kf,  &times, &testlls, &Ntll,
        x.data(), xt.data(), alpha, eta0.data(), nu0, &getLogHGaussian,
        &getStatGaussian,&getLogPostPredGaussian, N, Nt, M, D, K, Nmini); 
	//output results
	std::ostringstream oss6;
	oss6 << "svidpmix-trace.log";
	std::ofstream fout2(oss6.str().c_str());
	for (uint32_t i = 0; i < Ntll; i++){
		fout2 << times[i] << " " << testlls[i] << std::endl;
	}
	fout2.close();
	free(eta); free(nu); free(a); free(b); free(zeta); free(times); free(testlls);

	//moVB DP TEST
	std::cout << "Running moVB ..." << std::endl;
    moVBDP(&zeta, &eta, &nu, &a, &b, &Kf, &times, &testlls, &Ntll,
        x.data(), xt.data(), alpha, eta0.data(), nu0, &getLogHGaussian,
        &getStatGaussian,&getLogPostPredGaussian, N, Nt, M, D, K, Nmini); 
	//output results
	std::ostringstream oss7;
	oss7 << "movbdpmix-trace.log";
	std::ofstream fout3(oss7.str().c_str());
	for (uint32_t i = 0; i < Ntll; i++){
		fout3 << times[i] << " " << testlls[i] << std::endl;
	}
	fout3.close();
	free(eta); free(nu); free(a); free(b); free(zeta); free(times); free(testlls);

	//SVA DP TEST
	std::cout << "Running SVA ..." << std::endl;
	svaDP(&zeta, &eta, &nu, &a, &b, &Kf, &times, &testlls, &Ntll,
	    x.data(), xt.data(), alpha, 1.0e-3, 1.0e-3, eta0.data(), nu0, &getLogHGaussian,
	    &getStatGaussian, &getLogPostPredGaussian, N, Nt, M, D, K); 
	//output results
	std::ostringstream oss8;
	oss8 << "svadpmix-trace.log";
	std::ofstream fout4(oss8.str().c_str());
	for (uint32_t i = 0; i < Ntll; i++){
		fout4 << times[i] << " " << testlls[i] << std::endl;
	}
	fout4.close();
	free(eta); free(nu); free(a); free(b); free(zeta); free(times); free(testlls);

	return 0;
}
