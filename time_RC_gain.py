# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 11:45:22 2017

@author: cfragada
"""
import time
import numpy as np
from matplotlib import pyplot as plt 
import scipy.sparse as sps

# sizes will be approximated to the closest square number
N=1024
K=4096
total_it = 1000 # 10000

# INITIALIZATIONS
N = int(round(np.sqrt(N)))**2;
K = int(round(np.sqrt(K)))**2;

N1 = int(np.sqrt(N))
K1 = int(np.sqrt(K))
N2 = N1
K2 = K1
max_rank = 2*N1

complexity_dense = 2*N*K
mean_time_dense = 0

complexity_struct = (2*N*np.sqrt(K)*(1+np.sqrt(K)/np.sqrt(N)))*(np.arange(max_rank)+1) + K
mean_time_struct = np.zeros(max_rank)

# Generate dictionary
#D = np.random.randn(N,K) # Gaussian random
D = np.zeros([N1*N2,K1*K2]);
A = np.random.randn(N1,K1,max_rank);
B = np.random.randn(N2,K2,max_rank);
for r in range(0,max_rank):
    D = D + np.kron(A[:,:,r],B[:,:,r])

Diag = np.diag(1/np.sqrt(np.sum(D**2,axis=0))) # Diagonal matrix that normalizes the dictionary columns
norm_vec = 1/np.sqrt(np.sum(D**2,axis=0))

# Sparse dictionary
sps_acc = sps.coo_matrix((N, K)) # empty matrix
sps_csr = sps.csr_matrix((N, K))
for j in range(1000): # add 100 sets of 100 1's
    r = np.random.randint(N, size=100)
    c = np.random.randint(K, size=100)
    d = np.ones((100,))
    sps_acc = sps_acc + sps.coo_matrix((d, (r, c)), shape=(N, K))
    sps_csr = sps_csr + sps.csr_matrix((d, (r, c)), shape=(N, K))

# Time cost - dense
for k in range(0,total_it):
    x = np.random.randn(K,1)
    tic = time.clock()
    y = D.dot(x)
    toc = time.clock()
    mean_time_dense = mean_time_dense + (toc-tic)/float(total_it)
    
# Time cost - structured
for approx_rank in range(0,max_rank):

    y2 = np.zeros([N1,N2])
    for k in range(0,total_it):
        x = np.random.randn(K)

        tic = time.clock()
        #x = Diag.dot(x)
        x = norm_vec*x
        X = np.reshape(x,[K2,K1]) # Unvec version of x
        for r in range(0,approx_rank+1):
            y2 = y2 + B[:,:,r].dot(X.dot(np.transpose(A[:,:,r])))
        y = np.reshape(y2,N1*N2) # KEEP IT?
        toc = time.clock()
        mean_time_struct[approx_rank]  = mean_time_struct[approx_rank]  + (toc-tic)/float(total_it)
        
        


# RCG - Theoretical
RCG = complexity_dense/complexity_struct
# Time gain - real
time_gain = mean_time_dense/mean_time_struct


## FIGURES
plt.xlabel('Theoretical RCG (Relative Complexity gain)')
plt.ylabel('Time gain')
plt.title('Time acceleration for SuKro dictionaries')
# Practical gain
plt.plot(RCG,time_gain,'ob')
# Linear fit
a,b = np.polyfit(RCG, time_gain, 1)
plt.plot(RCG,a*RCG+b,'--b')
# Show equation
plt.text(0.9*RCG[1], 1.1*time_gain[1], "y = {:.2f}x".format(a))
# Theoretical gain
plt.plot(RCG,RCG,'--k')
# Save figure
plt.savefig('./ResSynthData/time_RC_gain.pdf',bbox_inches = 'tight',bbox_pad = 2)

plt.figure()
plt.semilogy(range(1,max_rank+1),time_gain)


## SAVE RESULTS
np.savez('time_RC_gain',N=N,K=K,N1=N1,N2=N2,K1=K1,K2=K2,total_it=total_it,\
         mean_time_struct=mean_time_struct,mean_time_dense=mean_time_dense,\
         complexity_dense=complexity_dense,complexity_struct=complexity_struct)