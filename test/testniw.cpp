#include <sdabnp/model/normal_inverse_wishart>
#include <fstream>

int main(int argc, char** argv){
	VXd mu = VXd::Zero(2);
	MXd cov = MXd::Identity(2, 2);
	MXd pdf = MXd::Zero(200, 200);
	double dof = 2000.0;
	for (uint32_t i = 0; i < 200; i++){
		for (uint32_t j = 0; j < 200; j++){
			VXd v = VXd::Zero(2);
			v(0) = -5 + 10*((double)j)/((double)200.0);
			v(1) = -5 + 10*((double)i)/((double)200.0);
			pdf(i, j) = exp(multivariateTLogLike(v, mu, cov, dof));
		}
	}

	std::ofstream fout("niwtest.log");
	fout << pdf;
	fout.close();

	return 0;
}
