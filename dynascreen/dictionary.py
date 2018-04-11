# -*- coding: utf-8 -*-
"""
This module defines the Dict class
It is usefull for allowing fast indexed matrix-vector product.

Created on Mon Mar 18 16:29:46 2014

@author: antoine Bonnefoy
"""
import numpy as np  
import fast_mat_prod as fprod
import warnings 


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
        
        if opType=="matrix":
            self.shape = self.data.shape
            if not self.normalized:
                self.normalize()
        # Initializing sukro dictionary
        elif opType=="sukro":
            self.shape = self.data.shape
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
            if not self.normalized:
                self.normalize()
        elif opType=="low-rank":
            self.shape = self.data.shape            
            if not self.normalized:
                self.normalize()            
            # L and R matrices(such that D = L*R) provided via input 'params'
            #TODO test if L and R are provided. If not, do the SVD.
            self.L = params['L']
            self.R = params['R']/ self.normcoef[None,:] # Normalize R
            # Submatrices sizes
            self.nrank = self.L.shape[1]


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
            return self.L.dot(fprod.BlasCalcDx(self.R,vector,screen)) # Using screening
            
    
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
            return fprod.BlasCalcDty(self.R.T,self.L.T.dot(vector),screen) # Using screening
            
    ######################################################
            
    def normalize(self,dim = 0):
        """
        normalize the dictionnary in row (dim=1) or in column (dim=0)
        """
        
        if (self.opType == "matrix" or self.opType == "sukro" or self.opType == 'low-rank') and self.normcoef == []:
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
    
        

    
    
                