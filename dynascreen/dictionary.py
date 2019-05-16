# -*- coding: utf-8 -*-
"""
This module defines the Dict class
It is usefull for allowing fast indexed matrix-vector product.

Created on Mon Mar 18 16:29:46 2014

@author: antoine Bonnefoy, cassiofraga

Copyright (C) 2019 Cassio Fraga Dantas

SPDX-License-Identifier: AGPL-3.0-or-later
"""
import numpy as np  
import fast_mat_prod as fprod
import warnings 
from scipy.linalg import blas

class Dict:
    """
    Dictionary class
    includes the data and some basic method needed to compute products
    """
    def __init__(self, data=[], opType="matrix", params=dict(),
                 normalized = False,
                 normcoef = []):
        if data.flags['F_CONTIGUOUS']:
            self.data=data
        else:
            self.data = np.asarray(data, dtype=float, order='F')
            
        self.normalized=normalized
        self.normcoef=normcoef
        self.opType=opType
        
        self.shape = self.data.shape
        if not self.normalized:
            self.normalize()

        if opType=="sukro":
            # Submatrices - provided via input 'params'
            #TODO assert that A and B are provided in 'params'
            self.A = params['A']
            self.B = params['B']
            # Submatrices sizes
            self.N1 = self.A.shape[0]
            self.N2 = self.B.shape[0]
            self.K1 = self.A.shape[1]
            self.K2 = self.B.shape[1]
            self.nkron = self.A.shape[2]
        elif opType=="low-rank":  
            # L and R matrices(such that D = L*R) provided via input 'params'
            #TODO test if L and R are provided. If not, do the SVD.
            self.L = np.asarray(params['L'], dtype=float, order='F') # make sure it is F-contiguous
            self.R = np.asarray(params['R'], dtype=float, order='F')/ self.normcoef[None,:] # Normalize R
#            self.Lc = np.asarray(self.L, dtype=float, order='C') # C-contiguous copy
#            self.Rc = np.asarray(self.R, dtype=float, order='C') # C-contiguous copy
            self.nrank = self.L.shape[1]
        elif opType=="faust":
            #TODO assert that params contains a FAuST object
            self.faust = params # TODO add normcoef inside faust and remove it from the Apply methods
#            self.faust = []
#            for faust_k in params:
#                self.faust.append(faust_k)
#            self.napprox = 0
        elif not opType=="matrix":
            raise ValueError('Not valid dictionary type')

    def Apply(self,vector):
        if self.opType=="matrix":
            if self.data.__class__.__name__ == 'ndarray':
                return self.data.dot(vector)
            elif 'matrix' in self.data.__class__.__name__:
                raise NotImplementedError("The fast indexed matrix product vector\
                    for sparse matrix is not yet implemented")
        elif self.opType=="sukro":
            X = np.reshape(vector/self.normcoef[:,None],[self.K2,self.K1], order='F') # Unvec version of 'vector' (after multiplied by normalization scalars)
            y2 = np.zeros([self.N2,self.N1])
            for r in range(self.nkron):
                y2 = y2 + self.B[:,:,r].dot(X.dot(self.A[:,:,r].T))
            return np.reshape(y2,[self.N2*self.N1,1], order='F')
        elif self.opType=="low-rank":
            return self.L.dot(self.R.dot(vector))
        elif self.opType=="faust":
            # vector_normalized =  vector/self.normcoef[:,None]
            return self.faust*(vector/self.normcoef[:,None])
#            return self.faust*vector # without normalizing
                
    def ApplyScreen(self,vector,screen):
        """
        Apply screened D to vector
        """
        if self.opType=="matrix":
            if self.data.__class__.__name__ == 'ndarray':
                return fprod.BlasCalcDx(self.data,vector,screen)
                #return fprod.BlasCalcDty(self.data,vector,screen) #This is faster if C-contiguous
            elif 'matrix' in self.data.__class__.__name__:
                return self.Apply(vector)
        elif self.opType=="sukro": # Screening is not used
            X = np.reshape(vector/self.normcoef[:,None],[self.K2,self.K1], order='F') # Unvec version of 'vector' (after multiplied by normalization scalars)
            y2 = np.zeros([self.N2,self.N1])
            for r in range(self.nkron):
                y2 = y2 + self.B[:,:,r].dot(X.dot(self.A[:,:,r].T))
            return np.reshape(y2,[self.N2*self.N1,1], order='F')
        elif self.opType=="low-rank":
            return fprod.BlasCalcDx(self.L, fprod.BlasCalcDx(self.R,vector,screen), np.ones_like(screen)) # Using screening
#            return self.L.dot(fprod.BlasCalcDx(self.R,vector,screen)) # Using screening
#            return self.L.dot(self.R.dot(vector)) # without screening
#            return blas.dgemm(alpha=1., a=self.L, b=blas.dgemm(alpha=1., a=self.R, b=vector, trans_a=False), trans_a=False) 
        elif self.opType=="faust":
            # vector_normalized =  vector/self.normcoef[:,None]
            return self.faust*(vector/self.normcoef[:,None])
#            return self.faust*vector # without normalizing
    
    def ApplyTranspose(self,vector):
        if self.opType=="matrix":
            if self.data.__class__.__name__ == 'ndarray':
                return self.data.T.dot(vector)
            elif 'matrix' in self.data.__class__.__name__:
                raise NotImplementedError("The fast indexed matrix product vector\
                    for sparse matrix is not yet implemented")
            else:
                return self.data.T.dot(vector)
        elif self.opType=="sukro": # Screening is not used
            X = np.reshape(vector,[self.N2,self.N1], order='F') # Unvec version of 'vector'
            y2 = np.zeros([self.K2,self.K1])
            for r in range(self.nkron):
                y2 = y2 + self.B[:,:,r].T.dot(X.dot(self.A[:,:,r]))
            return np.reshape(y2,[self.K2*self.K1,1], order='F')/self.normcoef[:,None]
        elif self.opType=="low-rank": # Screening is not used
            return self.R.T.dot(self.L.T.dot(vector))
        elif self.opType=="faust":
            return (self.faust.transpose()*vector)/self.normcoef[:,None]
#            return self.faust.transpose()*vector # without normalizing
            
    def ApplyTransposeScreen(self,vector,screen):
        """
        Apply screened D transpose to dual vector
        """
        if self.opType=="matrix":            
            if self.data.__class__.__name__ == 'ndarray':
                return fprod.BlasCalcDty(self.data.T,vector,screen)
                #return fprod.BlasCalcDx(self.data.T,vector,screen) #This is faster if C-contiguous
            elif 'matrix' in self.data.__class__.__name__:
                return self.ApplyTranspose(vector)
        elif self.opType=="sukro": # Screening is not used
            X = np.reshape(vector,[self.N2,self.N1], order='F') # Unvec version of 'vector'
            y2 = np.zeros([self.K2,self.K1])
            for r in range(self.nkron):
                y2 = y2 + self.B[:,:,r].T.dot(X.dot(self.A[:,:,r]))
            return np.reshape(y2,[self.K2*self.K1,1], order='F')/self.normcoef[:,None]
        elif self.opType=="low-rank":
            return fprod.BlasCalcDty(self.R.T,fprod.BlasCalcDty(self.L.T,vector,np.ones_like(screen)),screen) # Using screening
#            return fprod.BlasCalcDty(self.R.T,self.L.T.dot(vector),screen) # Using screening
#            return self.R.T.dot(self.L.T.dot(vector)) # without screening
#            return blas.dgemm(alpha=1., a=self.R, b=blas.dgemm(alpha=1., a=self.L, b=vector, trans_a=True), trans_a=True) 
        elif self.opType=="faust":
            return (self.faust.transpose()*vector)/self.normcoef[:,None]
#            return self.faust.transpose()*vector # without normalizing
            
    ######################################################
            
    def normalize(self,dim = 0):
        """
        normalize the dictionnary in row (dim=1) or in column (dim=0)
        """
        
        if (self.opType == "matrix" or self.opType == "sukro" or self.opType == 'low-rank' or self.opType == "faust") and self.normcoef == []:
            if self.data.dtype!=np.float and self.data.dtype!=np.complex:
                self.data = self.data.astype(np.float,copy=False)
            (row,col) = self.data.shape
            
            if not self.data.__class__.__name__ == 'ndarray':
                warnings.warn("Data is not ndarray and normalization may fail")
            if dim==0: #normalize the columns
                self.normcoef = np.sqrt(np.sum(self.data**2,0))   
                self.data /= np.tile(self.normcoef,(row,1))   
                self.normalisation = 'column'
                
            elif dim==1: #normalize the lines
                self.normcoef = np.sqrt(np.sum(self.data**2,1))   
                self.data /= np.tile(self.normcoef,(col,1)).T   
                self.normalisation = 'raw'
    
        

    
    
                
