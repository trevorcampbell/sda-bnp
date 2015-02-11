#ifndef __NIW_MODEL_HPP
#include <Eigen/Dense>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/special_functions/digamma.hpp>

typedef Eigen::VectorXd VXd;
typedef Eigen::MatrixXd MXd;
using boost::math::digamma;
using boost::math::lgamma;

double multivariateLnGamma(double x, uint32_t p){
	double ret = p*(p-1)/4.0*log(M_PI);
	uint32_t i = 0;
	for (i = 0; i < p; i++){
	    ret += lgamma(x - i/2.0);
	}
	return ret;
}

double multivariatePsi(double x, uint32_t p){
	double ret = 0;
	uint32_t i = 0;
	for (i = 0; i < p; i++){
	    ret += digamma( (x-p-i-2.0)/2.0 );
	}
	return ret;
}

double multivariateTLogLike(VXd x, VXd mu, MXd cov, double dof){
	uint32_t D = cov.rows();
	Eigen::LDLT<MXd, Eigen::Upper> ldlt(cov);
	VXd diag = ldlt.vectorD();
	double ldet = 0.0;
	for (uint32_t i = 0; i < D; i++){
		ldet += log(diag(i));
	}
	return lgamma( (dof+D)/2.0 ) - lgamma( dof/2.0 ) - D/2.0*log(dof) - D/2*log(M_PI) - 0.5*ldet - (dof+D)/2.0*(1.0+1.0/dof*(x-mu).transpose()*ldlt.solve(x-mu));
}

class NIWModel{
	public:
		NIWModel(VXd mu0, double kappa0, MXd psi, double xi0); 
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

NIWModel::NIWModel(VXd mu0, double kappa0, MXd psi, double xi0){
	this->D = mu0.rows();
	this->eta0 = VXd::Zero(D*D+D+1);
	//first component corresponds to psi
	for(uint32_t i = 0; i < D; i++){
		for (uint32_t j = 0; j < D; j++){
			this->eta0(i*D+j) = psi(i, j);
		}
	}
	//second to mu0
	for(uint32_t i =0; i < D; i++){
		this->eta0(D*D+i) = kappa0*mu0(i);
	}
	//third to xi
	this->eta0(D+D*D) = xi0+D+2;
	//fourth to kappa
	this->nu0 = kappa0;

	//compute logh0 based on the prior
	const double eta3frc = (this->eta0(D*D+D)-D-2.0)/2.0;
	MXd n1n2n2T = MXd::Zero(D, D);
    for (uint32_t i = 0; i < D; i++){
      for(uint32_t j=i; j < D; j++){
        n1n2n2T(i, j) = this->eta0(i*D+j) - 1.0/this->nu0*this->eta0(D*D+i)*this->eta0(D*D+j);
      }
    }

	Eigen::LDLT<MXd, Eigen::Upper> ldlt(n1n2n2T);
	VXd diag = ldlt.vectorD();
	double ldet = 0;
	for(uint32_t i =0; i < D; i++){
		ldet += log(diag(i));
	}

	//compute logh
	logh0 = -1.0*D/2.0*log(2.0*M_PI/this->nu0) +eta3frc*(ldet- 1.0*D*log(2.0))-multivariateLnGamma(eta3frc, D);
}

uint32_t NIWModel::getStatDimension(){
	return D*D+D+1;
}

double NIWModel::getNu0(){
	return nu0;
}

VXd NIWModel::getEta0(){
	return eta0;
}

VXd NIWModel::getStat(VXd data){
	VXd stat = VXd::Zero(D*D+D+1);
	//first to cov suff stat
	for(uint32_t i = 0; i < D; i++){
		for (uint32_t j = 0; j < D; j++){
			stat(i*D+j) = data(i)*data(j);
		}
	}
	//second to mean suff stat
	for(uint32_t i =0; i < D; i++){
		stat(D*D+i) = data(i);
	}
	//third is just 1
	stat(D+D*D) = 1.0;

	return stat;
}

double NIWModel::getLogH0(){
	return logh0;
}

void NIWModel::getLogH(MXd eta, VXd nu, VXd& logh, MXd& dlogh_deta, VXd& dlogh_dnu){
	uint32_t K = eta.rows();
	logh = dlogh_dnu = VXd::Zero(K);
	dlogh_deta = MXd::Zero(K, D*D+D+1);
	for (uint32_t k = 0; k < K; k++){
		const double eta3frc = (eta(k, D*D+D)-D-2.0)/2.0;
		MXd n1n2n2T = MXd::Zero(D, D);
        for (uint32_t i = 0; i < D; i++){
          //for(uint32_t j=i; j < D; j++){
          for(uint32_t j=0; j < D; j++){
            n1n2n2T(i, j) = eta(k, i*D+j) - 1.0/nu(k)*eta(k, D*D+i)*eta(k, D*D+j);
          }
        }

		Eigen::LDLT<MXd, Eigen::Upper> ldlt(n1n2n2T);
		VXd diag = ldlt.vectorD();
		double ldet = 0;
		for(uint32_t i =0; i < D; i++){
			ldet += log(diag(i));
		}
		MXd n1n2n2TInv = ldlt.solve(MXd::Identity(D, D));


		//compute logh
		logh(k) = -1.0*D/2.0*log(2.0*M_PI/nu(k)) +eta3frc*(ldet- 1.0*D*log(2.0))-multivariateLnGamma(eta3frc, D);

		/*compute dlogh_deta1*/
		for (uint32_t i = 0; i < D; i++){
			for (uint32_t j = 0; j < D; j++){
		    dlogh_deta(k, i*D+j) = eta3frc*n1n2n2TInv(i, j);
			}
   		}

		/*compute dlogh_deta2*/
		for (uint32_t i = 0; i < D; i++){
		    dlogh_deta(k, D*D+i) = 0;
		    for (uint32_t j = 0; j < D; j++){ 
		        dlogh_deta(k, D*D+i) += -2.0/nu(k)*dlogh_deta(k, i*D+j)*eta(k, D*D+j);
		    }
		}

		/*compute dlogh_deta3*/
		dlogh_deta(k, D*D+D) = 0.5*(ldet-D*log(2.0) - multivariatePsi(eta(k, D*D+D), D));	

		/*compute dlogh_dnu*/
		dlogh_dnu(k) = 0.5*D/nu(k);
 		for(uint32_t i = 0; i < D; i++){
		    dlogh_dnu(k) -= 1.0/(2.0*nu(k))*eta(k, D*D+i)*dlogh_deta(k, D*D+i);
		}
	}
}

double NIWModel::getLogPosteriorPredictive(VXd x, VXd etak, double nuk){
	//convert etak to regular parameters of NIW
	MXd psi_post = MXd::Zero(D, D);
	VXd mu_post = VXd::Zero(D);
	double k_post = 0;
	double xi_post = 0;

	for(uint32_t i = 0; i < D; i++){
		for(uint32_t j = 0; j < D; j++){
			psi_post(i, j) = etak(i*D+j) - etak(D*D+i)*etak(D*D+j)/nuk;
		}
		mu_post(i) = etak(D*D+i)/nuk;
	}
	xi_post = etak(D*D+D)-D-2.0;
	k_post = nuk;

	//get multivariate t parameters
	double dof = xi_post - D + 1.0;
	MXd scale = psi_post*(k_post+1)/(k_post*dof);
	return multivariateTLogLike(x, mu_post, scale, dof);
}

#define __NIW_MODEL_HPP
#endif /* __NIW_MODEL_HPP */
