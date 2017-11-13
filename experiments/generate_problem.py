# -*- coding: utf-8 -*-
"""
Created on Mon May 13 16:23:04 2013

@author: antoinebonnefoy
"""
import numpy as np
from numpy import linalg as la
import timeit

from dynascreen.problem import Problem, Lasso, GroupLasso
from dynascreen.dictionary import Dict

from mnist import MNIST

    
    
def generate(opt, D=None,y=None):
    '''
    Generate the problem corresponding to the options given in opt
    
    Parameters
    ----------
    opt : dict
        a dictionary that contains all the options
        
    D : Dict instance, optional
        if D is already computed not need to recompute it
    
    '''

    if not opt['Gr']:
        if 'dict_params' not in opt.keys():
            opt['dict_params']=None

        D, y = noise(opt['N'],opt['K'], dict_type=opt['dict_type'], data_type=opt['data_type'],\
                     dict_params=opt['dict_params'], D=D,y=y)
        
        prob = Lasso(D, y)
        star,lstar = prob.getStar() 
        prob.pen_param = opt['lasso']*lstar
        
    else:

        Gr= buildGroups(opt['K'],opt['grsize'])
        opt['Gr']=Gr        
        D, y = noise(opt['N'],opt['K'], dict_type=opt['dict_type'], data_type=opt['data_type'], D=D,y=y)
        if opt['sparse'] is not None:
            y, coef_GT = sparse_sig(D, s=opt['sparse'], Gr= Gr, SNR=20)
        prob = GroupLasso(D, y, Gr)
        star,lstar = prob.getStar() 
        prob.pen_param = opt['lasso']*lstar            
        t0 = timeit.time.time()
        prob.calcGroupMatNorm()
        t0_bis = timeit.time.time() 
        opt['matNormTime'] = t0_bis - t0
        
    return prob, opt


def noise(N, K, dict_type="gnoise",  data_type="gnoise", dict_params={}, D=None,y=None):
    '''
    Generate a dictionary and signal where the signal and all atoms follow the
    same distribution gaussians noise or pnoise.
    
    Parameters
    ----------
    N,K : int
        dimension of the dictionary
    
    dict_type : str
        the distribution type ('gnoise' or 'pnoise')
    
    D : Dict instance, optional
        if the ditionary has already been computed

    '''    

    if D and D.__class__.__name__ != 'Dict':
        raise ValueError("The Dictionary is not of the Dict class")
    elif not D:
        if dict_type=='pnoise':
            D = Dict(np.eye(N,1)+0.01*np.tile(np.random.rand(1,K),\
                    (N,1))*np.random.randn(N,K)) 
        elif dict_type=='gnoise':
            D = Dict(np.random.randn(N,K))
        elif dict_type=='sukro':
            D_sukro = np.zeros([N,K])
            # Separable Factors
            A = np.random.randn(dict_params['N1'],dict_params['K1'],dict_params['n_kron'])
            B = np.random.randn(dict_params['N2'],dict_params['K2'],dict_params['n_kron'])
            for k_rank in range(dict_params['n_kron']):
                D_sukro = D_sukro + np.kron(A[:,:,k_rank],B[:,:,k_rank])
                
            D = Dict(D_sukro,opType="sukro",params=dict(A=A,B=B))
        elif dict_type=='low-rank':
            D_lowrank = np.zeros([N,K])
            # Low-rank Factors
            L = np.random.randn(N,dict_params['n_rank'])
            R = np.random.randn(dict_params['n_rank'],K)
            D_lowrank = L.dot(R)
            D = Dict(D_lowrank,opType="low-rank",params=dict(L=L,R=R))        
        elif dict_type.startswith('MNIST'): # MNIST_(fast_dict_type). 
                                            # E.g. 'MNIST_low-rank' or 'MNIST_sukro'
            mndata = MNIST('./datasets/MNIST/LeCun/')
            images_train,_ = mndata.load_training() #labels were ignored
            images_test,_ =mndata.load_testing()
            images_train = np.asarray(images_train).T
            
            # Fast approximation of data
            if dict_type.endswith('low-rank'):
                print()                
                #TODO
                
                             
        D.normalize()
    
    # BEFORE data_type
#    if dict_type=='pnoise':
#        y = np.eye(N,1)+0.01*np.tile(np.random.rand(1,1),\
#                (N,1))*np.random.randn(N,1)
#    elif dict_type=='gnoise':
##        y = np.random.randn(N,1)
#        #TODO teste, bernoulli (not really) gaussian
#        nz = int(0.02*K)
#        beta = np.zeros((K,1))
#        idx = K*np.random.rand(1,nz)
#        beta[idx.astype(int)] = np.random.randn(nz,1)
#        y = D.data.dot(beta)
#    elif dict_type=='sukro':
#        y = np.random.randn(N,1)        
    
    if not y:    
        if data_type=='pnoise':
            y = np.eye(N,1)+0.01*np.tile(np.random.rand(1,1),\
                    (N,1))*np.random.randn(N,1)
        elif data_type=='gnoise':
            y = np.random.randn(N,1)
        elif data_type=="bernoulli-gaussian":
            #Bernoulli (not really) gaussian active features generanting 'y'
            p = 0.02
            nz = int(p*K) # not really bernoulli because the number of active features is deterministic here.
            beta = np.zeros((K,1))
            idx = K*np.random.rand(1,nz)
            beta[idx.astype(int)] = np.random.randn(nz,1)
            y = D.data.dot(beta)        
        
    
    y /= np.linalg.norm(y)
    
    return D,y
 

     
def buildGroups(K,grsize):
    Gr = []
    grwght = np.sqrt(grsize)
    for i in range(int(np.floor(float(K)/grsize))):
        Gr.append( (range(grsize*i, grsize*(i+1)), grwght))
    if Gr[-1][0][-1] != K-1:
        Gr.append( (range(grsize*(i+1), K), np.sqrt(K-grsize*(i+1))))
    
    return Gr
        
       
    
def test_group(D,Gr):
    l = []
    for i, (g,wg) in enumerate(Gr):
        l+=g
    l.sort()
    if l != range(D.shape[1]):
        print('''probleme avec les groupes soit overlap soit tous les 
                éléments ne sont pas dans un groupe''')
        return False
    return True
    

    

def sparse_sig(D, s=0.1, SNR=np.inf, Gr=[]):# a compléter avec d'autres options
    '''
    Generate a noisy sparse signal from dictionary D. The resulting signal is
    a normalize signal with SNR ginven by the third arguments SNR.
    
    By default the sparsity is taken to have a signal which is the linear
    combinaison of 10% of tha tioms in the dictionary, and the signal is not noisy.
    
    Parameters
    ----------
    D : Dict instance
        The dictionary on which the signal is sparse
        
    s : float
        s is the sparsity of the generated signal
        if s is less than one we use a bernouilli gaussian model,
        else s represent the number of non-zeros coeficient
        
    SNR : float
        the desired SNR by default we do not add noise
        
    Gr : list of tuple
        to construct group sparse signals
        
    Returns
    -------
    y the generated signal and x its true sparse representation in D
    
    '''    
    x = np.zeros((D.shape[1],1),dtype = float)
    if Gr==[]:
        if s >= 1: 
            x[np.random.randint(0,D.shape[1],int(s)),0] = 1.
        else:
            x = np.random.binomial(1,s, size = (D.shape[1],1)).astype(float)    
    else:
        test_group(D,Gr)
        if s >= 1:
            actGroup = np.random.randint(0, len(Gr), int(s)).astype(float)
        else:
            actGroup = np.random.binomial(1, s, size = (len(Gr),1)).astype(float)
        x = np.zeros((D.shape[1],1), dtype=float)    
        for ind, (g, wg) in enumerate(Gr):
            if actGroup[ind] == 1:        
                x[g] = 1.    
    
        
    x[x==1] = np.random.randn((x==1).sum())
    y = D.Apply(x)
    y /= la.norm(y)
    n = np.random.randn(D.shape[0],1)
    n /= la.norm(n,ord=2)
    y += 10**(-SNR/20.) * n
    y /= la.norm(y)    
 
    return y,x