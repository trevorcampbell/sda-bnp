#ifndef __DIR_MODEL_HPP
#include <Eigen/Dense>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/special_functions/digamma.hpp>

typedef Eigen::VectorXd VXd;
typedef Eigen::MatrixXd MXd;
using boost::math::digamma;
using boost::math::lgamma;


class DirModel{
	public:
		DirModel(VXd alpha0); 
		VXd getEta0();
		double getNu0();
		uint32_t getStatDimension();
		VXd getStat(VXd data);
		double getLogH0();
		void getLogH(MXd eta, VXd nu, VXd& logh, MXd& dlogh_deta, VXd& dlogh_dnu);
		double getLogPosteriorPredictive(VXd stat, VXd etak, double nuk);
	private:
		uint32_t D;
		VXd eta0;
		double nu0;
		double logh0;
};

DirModel::DirModel(VXd alpha0){
	this->D = alpha0.rows();
	this->eta0 = alpha0 - VXd::Ones(D);
	this->nu0 = 0; //nu isn't needed in the dirichlet model
	this->logh0 = lgamma(alpha0.sum());
	for (uint32_t i = 0; i < D; i++){
		this->logh0 -= lgamma(alpha0(i));
	}
}

uint32_t DirModel::getStatDimension(){
	return D;
}

double DirModel::getNu0(){
	return nu0;
}

VXd DirModel::getEta0(){
	return eta0;
}

VXd DirModel::getStat(VXd data){
	VXd stat = VXd::Zero(D);
	stat(data(0)) = data(1);
	return stat;
}

double DirModel::getLogH0(){
	return logh0;
}

void DirModel::getLogH(MXd eta, VXd nu, VXd& logh, MXd& dlogh_deta, VXd& dlogh_dnu){
	uint32_t K = eta.rows();
	logh = dlogh_dnu = VXd::Zero(K);
	dlogh_deta = MXd::Zero(K, D);
	for (uint32_t k = 0; k < K; k++){
		dlogh_deta.row(k) = digamma(D + eta.row(k).sum());
		logh(k) = lgamma(D + eta.row(k).sum());
		for(uint32_t i = 0; i < D; i++){
			dlogh_deta(k, i) -= digamma(eta(k, i)+1.0);
			logh(k) -= lgamma(eta(k, i)+1.0);
		}
		dlogh_dnu(k) = 0.0;
	}
	//std::cout << "Eta: " << std::endl << eta << std::endl;
	//std::cout << "Nu: " << std::endl << nu << std::endl;
	//std::cout << "logh: " << std::endl << logh << std::endl;
	//std::cout << "dlogh_deta: " << std::endl << dlogh_deta << std::endl;
	//std::cout << "dlogh_dnu: " << std::endl << dlogh_dnu << std::endl;

}

double DirModel::getLogPosteriorPredictive(VXd x, VXd etak, double nuk){
	double alphai = etak(x(0))+1.0;
	double sumalpha = D + etak.sum();
	return log(alphai/sumalpha);
}

double DirModel::naturalParameterDistSquared(VXd& stat1, VXd& stat2){
	//TODO: FIX THIS
	assert(false);
	return 0;
}


#define __DIR_MODEL_HPP
#endif /* __DIR_MODEL_HPP */
