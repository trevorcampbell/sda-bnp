#ifndef __COSTFCN_H
#include <stdint.h>
#include <stdbool.h>

double varLabelEntropy(const double * const zeta, const uint32_t N, const uint32_t K);

double varBayesCost(
		const double * const zeta,
		const double * const sumzeta,
		const double * const sumzetaT,
		const double * const a,
		const double * const b,
		const double * const eta,
		const double * const eta0,
		const double * const nu,
		const double nu0,
		const double * const logh,
		const double logh0,
		const double * const dlogh_deta,
		const double * const dlogh_dnu,
		const double alpha,
		const uint32_t N,
		const uint32_t M,
		const uint32_t K);

#define __COSTFCN_H
#endif /* __COSTFCN_H */
