#ifndef __UPDATES_H
#include <stdint.h>
#include <stdbool.h>

void updateWeightDist(double* a, double* b, double* psisum,
		const double* const sumzeta, 
		const double alpha, 
		const uint32_t K);

void updateWeightDist(double* a, double* b, double* psisum,
		const double* const sumzeta, 
		const double alpha, 
		const uint32_t K);

void updateParamDist(double* const eta, double* const nu, 
		const double* const eta0, 
		const double nu0, 
		const double sumzeta, 
		const double* const sumzetaT, 
		const uint32_t M);

void updateLabelDist(double* const zeta,
		const double * const stat, 
		const double * const dlogh_deta, 
		const double * const dlogh_dnu, 
		const double * const psisum, 
		const uint32_t M, 
		const uint32_t K);

void computeRho(double* const r, 
		const double* const w,
		const double* const stat, 
		const double* const eta, 
		const double* const nu, 
		const double* const eta0, 
		const double nu0, 
		double * const etatmp,
		void (*getLogH)(double*, double* const, double* const, const double * const, const double, const uint32_t, bool),
		const double alpha, 
		const uint32_t M,
		const uint32_t D,
		const uint32_t K);


#define __UPDATES_H
#endif /* __UPDATES_H */
