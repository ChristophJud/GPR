
// do not replace ifndef with pragma once
#ifndef LAPACK_UTILS_H
#define LAPACK_UTILS_H

#include <exception>
#include "GaussianProcess.h"

namespace gpr {
namespace lapack{

#ifndef NO_LAPACK_FLAG
extern "C" {
    // LU decomoposition of a general matrix
    void dgetrf_(int* M, int *N, double* A, int* lda, int* IPIV, int* INFO);

    // generate inverse of a matrix given its LU decomposition
    void dgetri_(int* N, double* A, int* lda, int* IPIV, double* WORK, int* lwork, int* INFO);

    // cholesky decomposition
    void dpotrf_(char *UPLO, int *N, double *A, int *LDA, int *INFO);

    // inversion of a symmetric positive definit matrix using its cholesky decomposition
    void dpotri_(char *UPLO, int *N, double *A, int *LDA, int *INFO);

    int ilaenv_(int* ispec, char* name, char* opts, int* n1, int* n2, int* n3, int* n4);
}

int get_optimal_block_size(int N){
    int ispec = 1;
    int n = -1;
    char name[] = "DGETRI";
    char space[] = " ";

    return ilaenv_(&ispec, name, space, &N, &n, &n, &n);
}

int lu_inversion(double* A, int N)
{
    int *IPIV = new int[N];
    //int LWORK = N*N;
    int LWORK = N*get_optimal_block_size(N);
    double *WORK = new double[LWORK];
    int INFO=0;

    dgetrf_(&N,&N,A,&N,IPIV,&INFO);
    if(INFO!=0) return INFO;

    dgetri_(&N,A,&N,IPIV,WORK,&LWORK,&INFO);
    if(INFO!=0) return INFO;

    delete[] IPIV;
    delete[] WORK;

    return INFO;
}


void chol_inversion(double* A, int N)
{
    int INFO;
    char UPLO = 'L'; // for Eigen upper triangle of inv(A) is correct

    dpotrf_(&UPLO, &N, A, &N, &INFO);
    dpotri_(&UPLO, &N, A, &N, &INFO);

    // since only upper triangle is correct, mirror matrix on diagonal
    for(unsigned i=0; i<N; i++){
        for(unsigned j=i; j<N; j++){
            A[j*N+i] = A[i*N+j];
        }
    }
}
#endif

struct LAPACKException : public std::exception
{
   std::string s;
   LAPACKException(std::string ss) : s(ss) {}
   ~LAPACKException() throw () {} // Updated
   const char* what() const throw() { return s.c_str(); }
};

// wrapper arround lapack inversion routine (LU)
template<class T>
typename GaussianProcess<T>::MatrixType lu_invert(const typename GaussianProcess<T>::MatrixType& matrix){
    GaussianProcess<double>::MatrixType inv_matrix(matrix.template cast<double>());

#ifndef NO_LAPACK_FLAG
    int info = lapack::lu_inversion(inv_matrix.data(), inv_matrix.cols());
    if(info!=0) throw LAPACKException("lu_invert: error in inversion. Matrix is singular or contains an illegal value.");
#else
    throw LAPACKException("lu_invert: lapack library not linked.");
#endif

    return inv_matrix.cast<T>();
}

// wrapper arround lapack inversion routine (cholesky) (never use float as T)
template<class T>
typename GaussianProcess<T>::MatrixType chol_invert(const typename GaussianProcess<T>::MatrixType& matrix){
    GaussianProcess<double>::MatrixType inv_matrix(matrix.template cast<double>());

#ifndef NO_LAPACK_FLAG
    lapack::chol_inversion(inv_matrix.data(), matrix.cols());
#else
    throw LAPACKException("chol_invert: lapack library not linked.");
#endif

    return inv_matrix.cast<T>();
}

} // lapack namespace
} // gpr namespace

#endif
