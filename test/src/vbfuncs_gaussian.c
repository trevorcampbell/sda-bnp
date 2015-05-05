#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <gsl/gsl_sf_psi.h>
#include <gsl/gsl_sf_gamma.h>
#include <stdint.h>
#include "../include/vbfuncs_gaussian.h"

/*A couple functions that are private to this c file*/
double multivariateLnGamma(double, uint32_t); /*Computes the multivariate Gamma*/
double multivariatePsi(double x, uint32_t p); /*Computes the multivariate Psi*/
uint32_t choleskyLDetAndInversion1D(double *, uint32_t, double *, double*, bool); /*inverts matrix, computes logdet via cholesky*/

void getStatGaussian(double* stat, const double* const y, const uint32_t D){
	uint32_t i, j;
	for(i = 0; i < D; i++){
	for(j = 0; j < D; j++){
		stat[i*D+j] = y[i]*y[j];
	}
	}
	for(i = 0; i < D; i++){
		stat[D*D+i] = y[i];
	}
	stat[D*D+D] = 1.0;
	return;
}

void getLogHGaussian(double* logh, double* const dlogh_deta, double* const dlogh_dnu,
	const double * const eta, 
	const double nu, 
	const uint32_t D, bool doDeriv){

	/*Compute dimension from M = d+d^2+1*/
	/*(M-1) = d(d+1)*/
	/*Note that d^2 < d(d+1) < (d+1)^2*/
	/*so floor(sqrt(M-1)) = d*/
	/*precompute a fraction that gets used a lot*/
	const double eta3frc = (eta[D*D+D]-D-2.0)/2.0;
	/*compute (n1-1/nu*n2*n2^T)^{-1} and its neg logdet*/ 
	double* n1n2n2T = (double*) malloc(sizeof(double)*D*D);
	double* n1n2n2TInv  = NULL;
	if(doDeriv){
		n1n2n2TInv = (double*) malloc(sizeof(double)*D*D);
	}
	double ldet = 0;
	if (n1n2n2T == NULL || (doDeriv && n1n2n2TInv == NULL)){
		printf("Error (getLogHGaussian): Cannot allocate memory\n");
	//	exit(0);
	}
	/*fill in n1 - 1/nu n2n2^T, only actually compute upper triangle and just fill in lower*/
	uint32_t i, j;
  for (i = 0; i < D; i++){
    for(j = 0; j < i; j++){
      n1n2n2T[i*D+j] = n1n2n2T[j*D+i];
    }
    for(j=i; j < D; j++){
      n1n2n2T[i*D+j] = eta[i*D+j] - 1.0/nu*eta[D*D+i]*eta[D*D+j];
    }
  }

//  for (i = 0; i < D; i++){
//    for(j = 0; j < D; j++){
//      printf("%f ",n1n2n2T[i*D+j]);
//    }
//    printf("\n");
//  }
	/*compute inverse and log det*/
	int chret = choleskyLDetAndInversion1D(n1n2n2T, D, &ldet, n1n2n2TInv, doDeriv);
	if (chret==0){
		printf("Error (getLogHGaussian): Non PSD matrix detected!\n");
	//	exit(0);
	}
  // Looks fine
//  printf("ldet=%f\n",ldet);
//  if(doDeriv){
//    for(uint32_t i=0; i<D*D; ++i)
//      printf("%f ",n1n2n2TInv[i]);
//    printf("\n");
//  }

	/*compute logh*/
	//printf("eta3frc: %f D: %d eta3: %f ", eta3frc, D, eta[D*D+D]);
	*logh = -1.0*D/2.0*log(2.0*M_PI/nu) +eta3frc*(ldet- 1.0*D*log(2.0))-multivariateLnGamma(eta3frc, D);

	if (doDeriv){

		/*compute dlogh_deta1*/
		for (i = 0; i < D*D; i++){
		    dlogh_deta[i] = eta3frc*n1n2n2TInv[i];
   		}

		/*compute dlogh_deta2*/
		for (i = D*D; i < D*D+D; i++){
		    dlogh_deta[i] = 0;
		    for (j = 0; j < D; j++){ 
		        dlogh_deta[i] += -2.0/nu*dlogh_deta[(i-D*D)*D+j]*eta[D*D+j];
		    }
		}

		/*compute dlogh_deta3*/
		dlogh_deta[D*D+D] = 0.5*(ldet-D*log(2.0) - multivariatePsi(eta[D*D+D], D));	

		/*compute dlogh_dnu*/
		*dlogh_dnu = 0.5*D/nu;
 		for(i = 0; i < D; i++){
		    *dlogh_dnu -= 1.0/(2.0*nu)*eta[D*D+i]*dlogh_deta[D*D+i];
		}
	/*done! Free up memory*/
		free(n1n2n2TInv); //only free this if we used it to compute derivatives
	}
	free(n1n2n2T);
}


double multivariateLnGamma(double x, uint32_t p){
	double ret = p*(p-1)/4.0*log(M_PI);
	uint32_t i = 0;
	for (i = 0; i < p; i++){
	    ret += gsl_sf_lngamma(x - i/2.0);
	}
	return ret;
}

double multivariatePsi(double x, uint32_t p){
	double ret = 0;
	uint32_t i = 0;
	for (i = 0; i < p; i++){
	    ret += gsl_sf_psi( (x-p-i-2.0)/2.0 );
	}
	return ret;
}

uint32_t choleskyLDetAndInversion1D(const double * const mat, uint32_t n, double* ldet, double * inv, bool doInv)
     /* 
	Do the augmented cholesky decomposition as described in FA Graybill
	(1976) Theory and Application of the Linear Model with 1-dimensional
	flattened matrices. The original matrix
	must be symmetric positive definite. C^-t is found in the process, where C is the
	upper triangular cholesky matrix, ie C^t * C = M and M is the original
	matrix. Then C^-t is multiplied by its transpose, ie C^-1C^-t = M^-1
        which is used to fill in inv. ldet is computed inthe process.
        Returns with a value of 0 if M is a nn-positive definite 
	matrix. Returns with a value of 1 with succesful completion.

	Arguments:

	mat (input) double n x n array. The matrix to take the Cholesky
	      decomposition of.
	n    (input) integer. Number of rows and columns in orig.
	inv (output) double n x n array. Holds the inverse of mat.
	ldet (output) double. The log determinant of mat.
     */
{
  uint32_t i, j, k, l;
  uint32_t retval = 1;
  *ldet = 0.0;

  /*allocate memory to store C^{-t}*/
  double* cnt = NULL;
  if (doInv){
    cnt = (double*) malloc(sizeof(double)*n*n);
  }
  double* chol = (double*) malloc(sizeof(double)*n*n);
  if (chol == NULL || (doInv && cnt == NULL)){
    printf("ERROR (cholesky1DInv): could not allocate memory!");
  }

  for (i=0; i<n; i++) {
    uint32_t rw = i*n;
    chol[rw+i] = mat[rw+i];
    for (k=0; k<i; k++)
      chol[rw+i] -= chol[k*n+i]*chol[k*n+i];
    if (chol[rw+i] < 0) { // TODO or is it <= ?
      //	 fprintf(stderr,"\nERROR: non-positive definite matrix!\n");
      printf("\nproblem from %d %f\n",i,chol[rw+i]);
      retval = 0;
      return retval;
    }
    chol[rw+i] = sqrt(chol[rw+i]);
    (*ldet) += 2.0*log(chol[rw+i]);

    if (doInv){
      /*This portion gets C^-t and stores in cnt */
      for (l=0; l<n; l++) {
        cnt[rw+l] = (i == l ? 1 : 0);
        for (k=0; k<i; k++) {
          cnt[rw+l] -= cnt[k*n+l]*chol[k*n+i];
        }
        cnt[rw+l] /= chol[rw+i];
      }
    }

    for (j=i+1; j<n; j++) {
      chol[rw+j] = mat[rw+j];
      for (k=0; k<i; k++)
        chol[rw+j] -= chol[k*n+i]*chol[k*n+j];
      chol[rw+j] /= chol[rw+i];
    }
  }

  if (doInv){
    /*Now cnt holds C^-T -- we have that C^T C = A,
      so C^{-1}C^{-T} = A^{-1}*/
    for(i=0; i < n; i++){
      for(j=0; j< i; j++){/*by symmetry copy upper to lower*/
        inv[i*n+j] = inv[j*n+i];
      }
      for(j=i; j < n; j++){ /*compute upper*/
        inv[i*n+j] = 0;
        for(k= i; k < n; k++){
          inv[i*n+j] += cnt[k*n+j]*cnt[k*n+i];
        }
      }
    }
  }
  if (doInv){free(cnt);}
  free(chol);

  return retval;
}

double multivariateTLogLike(const double* const x, const double* const mu, const double* const cov, double dof, const uint32_t D){
	double ldet = 0;
	double* cinv = (double*) malloc(sizeof(double)*D*D);
	if (choleskyLDetAndInversion1D(cov, D, &ldet, cinv, true) != 1){
      	exit(0);
	}
	double prod = 0;
	for (uint32_t i = 0; i < D; i++){
		for (uint32_t j = 0; j < D; j++){
			prod += (x[i]-mu[i])*cinv[i*D+j]*(x[j]-mu[j]);
		}
	}
	free(cinv);
	return gsl_sf_lngamma( (dof+D)/2.0 ) - gsl_sf_lngamma( dof/2.0 ) - D/2.0*log(dof) - D/2*log(M_PI) - 0.5*ldet - (dof+D)/2.0*log(1.0+1.0/dof*prod);
}

double getLogPostPredGaussian(const double* const x, const double* const etak, const double nuk, const uint32_t D){
	//convert etak to regular parameters of NIW
	double* psi_post = (double*) malloc(sizeof(double)*D*D);
	double* scale = (double*) malloc(sizeof(double)*D*D);
	double* mu_post = (double*) malloc(sizeof(double)*D);

	for(uint32_t i = 0; i < D; i++){
		for(uint32_t j = 0; j < D; j++){
			psi_post[i*D+j] = etak[i*D+j] - etak[D*D+i]*etak[D*D+j]/nuk;
		}
		mu_post[i] = etak[D*D+i]/nuk;
	}
	double xi_post = etak[D*D+D]-D-2.0;
	double k_post = nuk;

	//get multivariate t parameters
	double dof = xi_post - D + 1.0;
	for(uint32_t i = 0; i < D; i++){
		for(uint32_t j = 0; j < D; j++){
			scale[i*D+j] = (k_post+1.0)/(k_post*dof)*psi_post[i*D+j];
		}
	}
	double mvtll = multivariateTLogLike(x, mu_post, scale, dof, D);
	free(psi_post); free(scale); free(mu_post);
	return mvtll;
}



/* DEBUGGING PRINTOUTS
printf("Mat: \n");
   for (i = 0; i < n; i++){
   for (j = 0; j < n; j++){
   	   printf("%f ", mat[i*n+j]);
   }
   printf("\n");
   }
	printf("Chol: \n");
   for (i = 0; i < n; i++){
   for (j = 0; j < n; j++){
   	   printf("%f ", chol[i*n+j]);
   }
   printf("\n");
   }
   printf("Inv: \n");
   for (i = 0; i < n; i++){
   for (j = 0; j < n; j++){
   	   printf("%f ", inv[i*n+j]);
   }
   printf("\n");
   }
   printf("Ldet %f\n", *ldet);
   */
