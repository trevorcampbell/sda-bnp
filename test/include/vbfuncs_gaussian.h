#include <stdint.h>
#include <stdbool.h>

void getStatGaussian(double* stat, const double* const y, const uint32_t D);

void getLogHGaussian(double* logh, double* const dlogh_deta, double* const dlogh_dnu,
	const double * const eta, 
	const double nu, 
	const uint32_t D, bool doDeriv);


double* getLogPostPredGaussian(const double* const x, const double* const eta, const double* const nu, const uint32_t K, const uint32_t D, const uint32_t N);
