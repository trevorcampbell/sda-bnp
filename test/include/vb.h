#ifndef __VB_H
#include <stdbool.h>
#include <stdint.h>

void varDP_init(double* zeta, double* sumzeta, double* sumzetaT, double* logh0, double *prevobj,
		const double* const T, 
		const double* const eta0, 
		const double nu0,
		void (*getLogH)(double*, double* const, double* const, const double * const, const double, const uint32_t, bool),
		void (*getStat)(double*, const double* const, const uint32_t),
		const uint32_t N, 
		const uint32_t M,
		const uint32_t D,
		uint32_t K, uint32_t id);

double varDP_iteration(double* zeta, double* sumzeta, double* sumzetaT, double * logh0,
    double* eta, double* nu, double* a, double* b, unsigned int* out_K,
		const double* const T, 
    double* prevobj,
		const double alpha, 
		const double* const eta0, 
		const double nu0,
		void (*getLogH)(double*, double* const, double* const, const double * const, const double, const uint32_t, bool),
		void (*getStat)(double*, const double* const, const uint32_t),
		const uint32_t N, 
		const uint32_t M,
		const uint32_t D,
		uint32_t K,uint32_t id);

double varDP_noAllocSumZeta(double* zeta, double* sumzeta, double* sumzetaT,
    double* eta, double* nu, double* a, double* b, unsigned int* out_K,
		const double* const T, 
		const double alpha, 
		const double* const eta0, 
		const double nu0,
		void (*getLogH)(double*, double* const, double* const, const double * const, const double, const uint32_t, bool),
		void (*getStat)(double*, const double* const, const uint32_t),
		const uint32_t N, 
		const uint32_t M,
		const uint32_t D,
		uint32_t K,uint32_t id);

double varDP_noAlloc(double* zeta, double* eta, double* nu, double* a, double* b, uint32_t* out_K,
		const double* const T, 
		const double alpha, 
		const double* const eta0, 
		const double nu0,
		void (*getLogH)(double*, double* const, double* const, const double * const, const double, const uint32_t, bool),
		void (*getStat)(double*, const double* const, const uint32_t),
		const uint32_t N, 
		const uint32_t M,
		const uint32_t D,
		uint32_t K, uint32_t id);

double varDP(double** out_zeta, double** out_eta, double** out_nu, double** out_a, double** out_b, uint32_t* out_K,
		const double* const T, 
		const double alpha, 
		const double* const eta0, 
		const double nu0,
		void (*getLogH)(double*, double* const, double* const, const double * const, const double, const uint32_t, bool),
		void (*getStat)(double*, const double* const, const uint32_t),
		const uint32_t N, 
		const uint32_t M,
		const uint32_t D,
		uint32_t K);

double moVBDP_noAllocSumZeta(double* zeta, double* sumzeta, double* sumzetaT,
    double* eta, double* nu, double* a, double* b, unsigned int* out_K,
		const double* const T, 
		const double alpha, 
		const double* const eta0, 
		const double nu0,
		void (*getLogH)(double*, double* const, double* const, const double * const, const double, const uint32_t, bool),
		void (*getStat)(double*, const double* const, const uint32_t),
		const uint32_t N, 
		const uint32_t M,
		const uint32_t D,
		uint32_t K, uint32_t B, uint32_t id);

double moVBDP_noAlloc(double* zeta, double* eta, double* nu, double* a, double* b, unsigned int* out_K,
		const double* const T, 
		const double alpha, 
		const double* const eta0, 
		const double nu0,
		void (*getLogH)(double*, double* const, double* const, const double * const, const double, const uint32_t, bool),
		void (*getStat)(double*, const double* const, const uint32_t),
		const uint32_t N, 
		const uint32_t M,
		const uint32_t D,
		uint32_t K, uint32_t B, uint32_t id);

double moVBDP(double** out_zeta, double** out_eta, double** out_nu, double** out_a, double** out_b, uint32_t* out_K,
		const double* const T, 
		const double alpha, 
		const double* const eta0, 
		const double nu0,
		void (*getLogH)(double*, double* const, double* const, const double * const, const double, const uint32_t, bool),
		void (*getStat)(double*, const double* const, const uint32_t),
		const uint32_t N, 
		const uint32_t M,
		const uint32_t D,
		uint32_t K, uint32_t B);

double soVBDP_noAllocSumZeta(double* zeta, double* sumzeta, double* sumzetaT,
    double* eta, double* nu, double* a, double* b, unsigned int* out_K,
		const double* const T, 
		const double alpha, 
		const double* const eta0, 
		const double nu0,
		void (*getLogH)(double*, double* const, double* const, const double * const, const double, const uint32_t, bool),
		void (*getStat)(double*, const double* const, const uint32_t),
		const uint32_t N, 
		const uint32_t M,
		const uint32_t D,
		uint32_t K, uint32_t B, uint32_t id);

double soVBDP_noAlloc(double* zeta, double* eta, double* nu, double* a, double* b, unsigned int* out_K,
		const double* const T, 
		const double alpha, 
		const double* const eta0, 
		const double nu0,
		void (*getLogH)(double*, double* const, double* const, const double * const, const double, const uint32_t, bool),
		void (*getStat)(double*, const double* const, const uint32_t),
		const uint32_t N, 
		const uint32_t M,
		const uint32_t D,
		uint32_t K, uint32_t B, uint32_t id);


double soVBDP(double** out_zeta, double** out_eta, double** out_nu, double** out_a, double** out_b, uint32_t* out_K,
		const double* const T, 
		const double alpha, 
		const double* const eta0, 
		const double nu0,
		void (*getLogH)(double*, double* const, double* const, const double * const, const double, const uint32_t, bool),
		void (*getStat)(double*, const double* const, const uint32_t),
		const uint32_t N, 
		const uint32_t M,
		const uint32_t D,
		uint32_t K, uint32_t B);


double svaDP(double** out_zeta, double** out_eta, double** out_nu, double** out_a, double** out_b, uint32_t* out_K,
		const double* const T, 
		const double alpha, 
		const double epsclus,
		const double epspm,
		const double* const eta0, 
		const double nu0,
		void (*getLogH)(double*, double* const, double* const, const double * const, const double, const uint32_t, bool),
		void (*getStat)(double*, const double* const, const uint32_t),
		const uint32_t N, 
		const uint32_t M,
		const uint32_t D,
		uint32_t K);

void removeEmptyClusters(double* zeta, double* eta, double* nu, const double nu0, double* a, double* b, uint32_t* out_K,
		const uint32_t N,  const uint32_t M, uint32_t K, bool compress);

void removeEmptyClustersX(double* zeta, double* sumzeta, double* sumzetaT, double* eta, double* nu, 
			double* logh, double* dlogh_deta, double* dlogh_dnu, const double nu0, double* a, double* b, uint32_t* out_K,
		const uint32_t N,  const uint32_t M, const uint32_t K, bool compress);

#define __VB_H
#endif /* __VB_H */


