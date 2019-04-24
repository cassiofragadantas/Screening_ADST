# -*- coding: utf-8 -*-
"""
Created on Mon May 13 16:23:04 2013

@author: antoinebonnefoy
"""
import numpy as np
from numpy import linalg as la
import timeit
import os

from dynascreen.problem import Problem, Lasso, GroupLasso
from dynascreen.dictionary import Dict
    
    
def generate(opt, D=None,y=None,D_bis=None):
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

        D, y, D_bis  = noise(opt['N'],opt['K'], dict_type=opt['dict_type'], data_type=opt['data_type'],\
                            dict_params=opt['dict_params'], data_params=opt['data_params'],  
                            D=D,y=y,D_bis=D_bis)
        
        prob = Lasso(D, y, D_bis=D_bis)        
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


def noise(N, K, dict_type="gnoise",  data_type="gnoise", dict_params={}, data_params={},D=None,y=None,D_bis=None):
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
        elif dict_type=='sukro':
            D_sukro = np.zeros([N,K])
            # Separable Factors
            A = np.random.randn(dict_params['N1'],dict_params['K1'],dict_params['n_kron'])
            B = np.random.randn(dict_params['N2'],dict_params['K2'],dict_params['n_kron'])

            if dict_params.has_key('svd_decay') and dict_params['svd_decay'] is 'exponential':
                #svd_decay = np.exp(range(0,-dict_params['n_kron'],-1))
                svd_decay = np.exp(-np.linspace(0,9,dict_params['n_kron']))
            else:
                svd_decay = np.ones(dict_params['n_kron'])
            
            for k_rank in range(dict_params['n_kron']):
                A[:,:,k_rank] *= svd_decay[k_rank]
                D_sukro = D_sukro + np.kron(A[:,:,k_rank],B[:,:,k_rank])
                
            D = Dict(D_sukro,opType="sukro",params=dict(A=A,B=B))
        elif dict_type=='sukro_approx': # D is well approximated by SuKro. D_bis is a SuKro dictionary with the maximum n_kron
            D_sukro = np.zeros([N,K])
            if not dict_params.has_key('nkron_max'):
                dict_params['nkron_max'] = max(dict_params['n_kron'])
            nkron_max = dict_params['nkron_max']

            # SVD decay
            if dict_params.has_key('svd_decay') and dict_params['svd_decay'] is 'exponential':
                if not dict_params.has_key('svd_decay_const'):
                    dict_params['svd_decay_const'] = 0.5 #8.5/19 gives round normE for n_kron = {5, 10, 20}
                decay_const = dict_params['svd_decay_const']
                    
                svd_decay = np.exp(-decay_const*np.linspace(0,N-1,N))
            else:
                svd_decay = np.ones(nkron_max+1)
            
            decay_tol = 1e-7
            # Separable Factors
            if ('reuse' in dict_params):
                filename =  './ResSynthData/'+dict_type+'-dict_'+data_type+'-data_N'+str(N)+'_K'+str(K)+'.npz'
                if os.path.isfile(filename): # Load previously generate factors
                    Data = np.load(filename)
                    A = Data['A'][()]
                    B = Data['B'][()]
                else:
                    A = np.random.randn(dict_params['N1'],dict_params['K1'],max(nkron_max+1,sum(svd_decay>decay_tol)))
                    B = np.random.randn(dict_params['N2'],dict_params['K2'],max(nkron_max+1,sum(svd_decay>decay_tol)))
                    np.savez(filename,A=A,B=B)
            else:
                A = np.random.randn(dict_params['N1'],dict_params['K1'],max(nkron_max+1,sum(svd_decay>decay_tol)))
                B = np.random.randn(dict_params['N2'],dict_params['K2'],max(nkron_max+1,sum(svd_decay>decay_tol)))
            
            # Building dictionary
            # structured part (until the biggest chosen n_kron, plus one term)
            for k_rank in range(nkron_max+1):
                A[:,:,k_rank] *= svd_decay[k_rank]
                D_sukro = D_sukro + np.kron(A[:,:,k_rank],B[:,:,k_rank])

            D_bis = Dict(D_sukro,opType="sukro",params=dict(A=A[:,:,range(nkron_max)],B=B[:,:,range(nkron_max)]))            
            
            # rest
            # Ideally, we should go until k_rank < N, but this is too slow
            # Therefore, we keep going at least until the singular value is smaller than decay_tol
            for k_rank in range(nkron_max+1,sum(svd_decay>decay_tol)):
                A[:,:,k_rank] *= svd_decay[k_rank]
                D_sukro = D_sukro + np.kron(A[:,:,k_rank],B[:,:,k_rank])
            
            #D = Dict(D_sukro) # Normalize columns
            #D_sukro = D.data + 1e-3*np.sqrt(1./N)*np.random.randn(N,K) # Add noise
                
            D = Dict(D_sukro)
        elif dict_type=='low-rank':
            D_lowrank = np.zeros([N,K])
            # If rank is not provided
            if not dict_params.has_key('n_rank'):
                dict_params['n_rank'] = min(N,K)/10 + 1
            # Low-rank Factors
            L = np.random.randn(N,dict_params['n_rank'])
            R = np.random.randn(dict_params['n_rank'],K)
            D_lowrank = L.dot(R)
            D = Dict(D_lowrank,opType="low-rank",params=dict(L=L,R=R))
        else: #elif dict_type=='gnoise':
            # Gaussian dictionary is default
            D = Dict(np.random.randn(N,K))
                             
        D.normalize()

    if y is None:    
        if data_type=='pnoise':
            y = np.eye(N,1)+0.01*np.tile(np.random.rand(1,1),\
                    (N,1))*np.random.randn(N,1)
        elif data_type=="bernoulli-gaussian":
            #Bernoulli (not really) gaussian active features generanting 'y'
            if data_params.has_key('p'):
                p = data_params['p']
            else: # default value
                p = 0.02
            nz = int(p*K) # not really bernoulli because the number of active features is deterministic here.
            beta = np.zeros((K,1))
            idx = K*np.random.rand(1,nz)
            beta[idx.astype(int)] = np.random.randn(nz,1)
            y = D.data.dot(beta)        
        else: #if data_type=='gnoise':
            # Gaussian vector is default            
            y = np.random.randn(N,1)        
    
        y /= np.linalg.norm(y)
    
    return D,y,D_bis
 

     
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