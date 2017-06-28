# -*- coding: utf-8 -*-
"""
Created on Tue May 14 16:55:41 2013

@author: antoinebonnefoy
"""
import cython
import numpy as np
cimport numpy as np
import scipy 
#cimport scipy
cimport cython
from libc.math cimport sqrt

DTYPE = np.float
INDTYPE = np.int 
ctypedef np.float64_t DTYPE_t
ctypedef np.int_t INDTYPE_t

cdef extern from "cblas.h":
    enum CBLAS_ORDER:
       CblasRowMajor=101
       CblasColMajor=102
    enum CBLAS_TRANSPOSE:
       CblasNoTrans=111
       CblasTrans=112
       CblasConjTrans=113
       AtlasConj=114 
       
    void axpy "cblas_daxpy"(int N, double alpha, double *X, int incX, double *Y, int incY) 
    
    double blasdot "cblas_ddot"(int N, double *X, int incX, double *Y, int incY) 
     
#    void swap "cblas_dswap"(int N, double *X, int incX, double *Y, int incY ) 
#    
#    void blascopy "cblas_dcopy"(int N,double *X, int incX, double *Y, int incY)
#    
#    void blasMatVectProd "cblas_dgemv"(CBLAS_ORDER Order, CBLAS_TRANSPOSE TransA,
#           int M, int N,
#           double alpha, double *A, int lda,
#           double *X, int incX,
#           double beta, double *Y, int incY) 
    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def BlasCalcDx(np.ndarray[DTYPE_t,ndim=2] D, np.ndarray[DTYPE_t,ndim=2] x, np.ndarray[INDTYPE_t,ndim=1] S):


    cdef Py_ssize_t N = D.shape[0]
    cdef Py_ssize_t K = D.shape[1]
    cdef Py_ssize_t i, j
    cdef DTYPE_t s

    cdef np.ndarray[DTYPE_t, ndim=2] out = np.zeros((N,1),order='F')    
    
    for i in range(K):
        if S[i] and x[i,0]!=0:
            axpy(N, x[i,0],
                <DTYPE_t*>(D.data + i*D.strides[1]), D.strides[0] // sizeof(DTYPE_t),
                <DTYPE_t*>(out.data ), out.strides[0] // sizeof(DTYPE_t))
                
    return out
    
    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def BlasCalcDty(np.ndarray[DTYPE_t,ndim=2] Dt, np.ndarray[DTYPE_t,ndim=2] y, np.ndarray[INDTYPE_t,ndim=1] S):


    cdef Py_ssize_t N = Dt.shape[1]
    cdef Py_ssize_t K = Dt.shape[0]
    cdef Py_ssize_t i, j
    cdef DTYPE_t s

    cdef np.ndarray[DTYPE_t, ndim=2] out = np.zeros((K,1),order='F')    
    
    for i in range(K):
        if S[i] :
            out[i,0] = blasdot(N,
                <DTYPE_t*>(Dt.data + i*Dt.strides[0]), Dt.strides[1] // sizeof(DTYPE_t),
                <DTYPE_t*>(y.data ), y.strides[0] // sizeof(DTYPE_t))

    return out
    
#TODO
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def BlasGrNorm( np.ndarray[DTYPE_t,ndim=2] x, list Gr, np.ndarray[INDTYPE_t,ndim=1] S):
    """
    This function works only when groups are connexes
    """
    cdef Py_ssize_t nbGr = len(Gr)
    cdef Py_ssize_t i, j
    cdef DTYPE_t s
    cdef np.ndarray[DTYPE_t, ndim=1] out = np.zeros(nbGr,order='F')   

    for ind,(g,wg) in enumerate(Gr): 
        if S[g[0]] : 
            out[ind] = sqrt(blasdot(len(g),
                <DTYPE_t*>(x.data + <Py_ssize_t>g[0]*x.strides[0]),  x.strides[0] // sizeof(DTYPE_t),
                <DTYPE_t*>(x.data + <Py_ssize_t>g[0]*x.strides[0]),  x.strides[0] // sizeof(DTYPE_t)))

    return out
    
    
    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def BlasTestVect(np.ndarray[DTYPE_t,ndim=2] D,np.ndarray[DTYPE_t,ndim=2] Dtc, list Gr):
    """
    This function works only when groups are connexes
    """
    cdef Py_ssize_t N = D.shape[0]
    cdef Py_ssize_t K = D.shape[1]
    cdef Py_ssize_t nbGr = len(Gr)
    cdef Py_ssize_t i, j
    cdef DTYPE_t s
    cdef DTYPE_t matnorm
    cdef np.ndarray[DTYPE_t, ndim=2] mat   
    cdef np.ndarray[DTYPE_t, ndim=1] out = np.zeros(nbGr,order='F')    
    
    for ind,(g,wg) in enumerate(Gr):
        mat = D[:,g].T.dot(D[:,g])
        matnorm = sqrt(scipy.linalg.eigh(mat,eigvals_only=1,eigvals = (len(g)-1,len(g)-1)))
        out[ind] = sqrt(Dtc[g].T.dot(Dtc[g]))/(matnorm) - wg/matnorm

    return out
