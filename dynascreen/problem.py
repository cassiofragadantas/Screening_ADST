# -*- coding: utf-8 -*-
"""
This module defines the Prbolem class so that our solver can solve either Lasso or Group-Lasso while having a simple iterative step understanding.

Created on Wed Oct 29 17:33:11 2014

@author: antoinebonnefoy
"""


import numpy as np
from numpy import linalg as LA
from scipy import linalg as sLA 

from dictionary import Dict 
import fast_mat_prod as fprod


class Problem(object):
    def __init__(self, D, y, Gr = None, pen_param= None, D_bis = None):
        if D.__class__.__name__ != 'Dict':
            D = Dict(D)

        self.D = D
        self.D_bis = D_bis
        self.y = y
        self.pen_param = pen_param 
        self.Gr = Gr
        """
        Gr must contain list of tuples contining groups and associated weight 
        (group_index, (group, groupe_weight))
        (0,([...],1.))        
        """
        self.normy2 = y.T.dot(y)
                
    def objective(self, x, Screen):
        return self.loss(x, Screen) + self.pen_param * self.reg(x, Screen)
    
    def loss(self):
        return

    def grad(self):
        return

    def reg(self):
        return
        
    def prox(self):
        return
        
    def dualGap(self):
        return
        
    def getStar(self):
        return        
    

class Lasso(Problem):
    def __init__(self, D, y, pen_param = None, D_bis = None):
        super(Lasso,self).__init__(D, y, [], pen_param,D_bis)   
    
    def loss(self, x, Screen, res=None):
        if res is None:
            res = self.D.ApplyScreen(x, Screen.screen)-self.y
            
        return 0.5 * res.T.dot(res)
        
    def reg(self, x, Screen):
        return np.abs(x).sum()
        
    def gradient(self, x, Screen):
        app = self.D.ApplyScreen(x,Screen.screen)   
        dualpt = app - self.y
        grad = self.D.ApplyTransposeScreen(dualpt,Screen.screen)
        return app, dualpt, grad
        
    def prox(self, x, a, Screen):
        return SoftThresh(x, a)
        
    def getStar(self):
        scalProd = self.D.ApplyTranspose(self.y)
        star = np.argmax(np.abs(scalProd))
        lstar = np.abs(scalProd[star])
        return star, lstar
        
        
    def dualGap(self, x, Screen = None, dualpt = None, grad = None, feasDual = None ):
        if dualpt is None or grad is None:
            if Screen is None:
                Screen = np.ones_like(x,dtype=np.int)
            app, dualpt, grad = self.gradient(x, Screen)
            #feasibility_coef  =  min(1, 1.0 / LA.norm(grad , np.inf)) # dual scaling
            feasibility_coef = min(1, 1.0 / LA.norm(grad , np.inf)) if LA.norm(grad , np.inf) else 1 # dual scaling - avoiding division by 0
            feasDual = feasibility_coef * dualpt

        dgap = float(self.loss(None,None,dualpt)  \
            + self.pen_param * self.reg(x,None) \
            - 0.5 * self.normy2 \
            + (0.5*self.pen_param**2) *np.dot(feasDual.T-self.y.T/self.pen_param,feasDual-self.y/self.pen_param) )
        
        return dgap            
        
    # MODIFS for normalizing duality gap (MEG experiments)
    def dualGap_all(self, x, Screen = None, dualpt = None, grad = None, feasDual = None ):
        if dualpt is None or grad is None:
            if Screen is None:
                Screen = np.ones_like(x,dtype=np.int)
            app, dualpt, grad = self.gradient(x, Screen)
            #feasibility_coef  =  min(1, 1.0 / LA.norm(grad , np.inf)) # dual scaling
            feasibility_coef = min(1, 1.0 / LA.norm(grad , np.inf)) if LA.norm(grad , np.inf) else 1 # dual scaling - avoiding division by 0
            feasDual = feasibility_coef * dualpt

        primal = self.primal(x,dualpt)
        dual = self.dual(feasDual)
        
        return primal-dual, primal, dual           
        
    def dual(self, feasDual = None ):
        dual = float(0.5 * self.normy2 \
            - (0.5*self.pen_param**2) *np.dot(feasDual.T-self.y.T/self.pen_param,feasDual-self.y/self.pen_param) )
        
        return dual

    def primal(self, x, dualpt = None ):
        primal = float( self.loss(None,None,dualpt)  \
            + self.pen_param * self.reg(x,None)  )
        
        return primal
        
class GroupLasso(Problem):
    def __init__(self, D, y, Gr, pen_param = None):
        self.grnorm = None
        self.grMatNorm = np.zeros(len(Gr))
        super(GroupLasso,self).__init__(D, y, Gr, pen_param)   
    
    def loss(self, x, Screen, res=None):
        if res is None:
            res = self.D.ApplyScreen(x, Screen.screen)-self.y            
        return 0.5 * res.T.dot(res)    
        
    def reg(self, x, Screen):
        grnorm = fprod.BlasGrNorm(x, self.Gr, Screen.screen)
        return (grnorm*[wg for ind,(g,wg) in enumerate(self.Gr)]).sum()
        
    def gradient(self, x, Screen):
        app = self.D.ApplyScreen(x,Screen.screen)   
        dualpt = app - self.y
        grad = self.D.ApplyTransposeScreen(dualpt,Screen.screen)
        return app, dualpt, grad
        
    def prox(self, x, a, Screen):
        res = GroupSoftThresh(x, a, self.Gr, Screen.screen)
        return res
        
    def getStar(self):
        scalProd = self.D.ApplyTranspose(self.y)
#        enumGr =  [(ind,(g,wg)) for ind,(g,wg) in enumerate(self.Gr)]
        self.grnorm = fprod.BlasGrNorm(scalProd, self.Gr, np.ones(self.D.shape[1],dtype = int))
        self.grnorm /= np.asarray([wg for ind,(g,wg) in enumerate(self.Gr)])
        star = np.argmax(self.grnorm)
        lstar = self.grnorm[star]                
        return star, lstar        

    
    def calcGroupMatNorm(self, grMatNorm=None):
        if grMatNorm is None:
            if np.all(self.grMatNorm == 0):        
                for ind, (g, wg) in enumerate(self.Gr):
                    self.grMatNorm[ind] = np.sqrt(\
                            sLA.eigh(self.D.data[:,g].T.dot(self.D.data[:,g]),\
                            eigvals_only=1,eigvals = (len(g)-1,len(g)-1)))
        else:
            self.grMatNorm = grMatNorm
        return self.grMatNorm
    
def SoftThresh(x,a): 
    """
    This is the soft thresholding function    
    
    Parameters
    -----------
     x : 1-column array, float
         observation
         
     a : float
         thresholding parameter
         
    Returns
    ---------
     the shrinked vector
     
    """ 
    tmp=np.abs(x)-a
    return np.where(tmp>=0,tmp*np.sign(x),0)     
    

def GroupSoftThresh(x, a, Gr, screen): 
    """
    This is the soft thresholding function for group lasso
    
    Parameters
    -----------
     x : 1-column array, float
         observation
         
     a : float
         thresholding parameter
         
     Gr :  is the group list (index, (group, weight))
     
     screen : screen vector (int array)
         
    Returns
    ---------
     the shrinked vector
     
    """  
    grnorm = fprod.BlasGrNorm(x,Gr,screen)
    grcoef = np.asarray(grnorm) - a*np.asarray([wg for ind,(g,wg) in enumerate(Gr)])
    coef = np.where(grcoef<=0,0,grcoef/grnorm)      
    x *= np.repeat(coef,[len(g) for ind,(g,wg) in enumerate(Gr)])[:,np.newaxis]
    return x       
    