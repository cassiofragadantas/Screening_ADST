# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 17:04:01 2014

@author: cassiofraga
"""
import numpy as np
import os
import matplotlib
from matplotlib import pyplot as plt 
import copy
import time

 
from dynascreen.solve import solver
from dynascreen.solve import solver_approx
from . import generate_problem as GP
from .misc import mergeopt, make_file_name, type2name
from .misc import testopt, default_expe, make_pen_param_list

from dynascreen.dictionary import Dict

# Real data experiments
from mnist import MNIST
import scipy.io as sio


      
def first(opt =dict(), **keywords):
    '''
    Plot one execution of the dynamic screening for given options
     
    to modify default option just do first(algo_type = 'FISTA')...
    '''   
    #np.random.seed(0)
    np.random.seed(10) #used for figures with y=X\beta    
    
    lasso_list = [0.6] #[0.5, 0.75, 0.85]
    
    for lasso in lasso_list:
        default =  dict(dict_type = 'gnoise',data_type = 'bernoulli-gaussian', lasso=lasso, N=100, K=500,
                        stop=dict(rel_tol=1e-8, max_iter=10000), scr_type = "ST1", switching='default')
        # Test
#        default =  dict(dict_type = 'gnoise', data_type = 'bernoulli-gaussian', lasso=lasso, N=2500, K=10000,\
#                        stop=dict(dgap_tol=1e-5, max_iter=1000), scr_type = "ST1", switching='default')
                        
                        
        expe = mergeopt(default, opt, keywords)
        expeScrRate(opt=expe)

   
def second(opt=dict(), **keywords):
    '''
    Plot the normalized time and flops for a synthetic Lasso problem
    versus the penalization parameter \lambda/\lambda_*
    you can chosse the dictionary to be gaussian noise or pnoise
    '''        
    np.random.seed(0)
    default = dict(dict_type = 'gnoise',data_type = 'gnoise', N=1000,K=5000,scr_type = "ST1", switching='default')
    # Test
#    default = dict(dict_type = 'gnoise',data_type = 'bernoulli-gaussian', N=1000,K=5000,scr_type = "ST1", switching='default',samp=20, min_reg=0.1,nbRuns=1,stop=dict(dgap_tol=1e-6, max_iter=10000)) #, wstart=True)
    expe = mergeopt(opt, default, keywords)
    res = runLambdas(opt=expe)
    traceLambdas(res['timeRes'], res['nbIter'], res['nbFlops'] ,expe )

def first_sukro(opt =dict(), **keywords): #expe 6
    '''
    Plot one execution of the dynamic screening for given options
    
    /!\ This experiment gives different results for each approximation error
    because the problem is regenerated at each run. The reason for this is that
    the SuKro dictionary remains unchanged and the 'actual' dictionary is
    regenerate from it by adding a gaussian noise with a certain variance.
     
    to modify default option just do first(algo_type = 'FISTA')...
    '''
    np.random.seed(0)
    #np.random.seed(10) #used for figures with y=X\beta           
    
    lasso_list = [0.1] #[0.5, 0.75, 0.85]
    
    for lasso in lasso_list:

        # SuKro
#        default =  dict(dict_type = 'sukro', data_type = 'bernoulli-gaussian', lasso=lasso, N=2500, K=10000,\
#                        dict_params = dict(N1 = 50, N2 = 50, K1 = 100, K2 = 100, n_kron = 20),
#                        stop=dict(dgap_tol=5e-6, max_iter=1000), scr_type = "GAP", switching='default')
        # Low-rank
#        default =  dict(dict_type = 'low-rank', data_type = 'bernoulli-gaussian', lasso=lasso, N=2500, K=10000,\
#                        dict_params = dict(n_rank = 200),
#                        stop=dict(rel_tol=1e-8, max_iter=1000), scr_type = "ST1", switching='default')
        # MNIST
#        default =  dict(dict_type = 'low-rank', data_type = 'MNIST', lasso=lasso,N=784, K=60000,\
#                        dict_params = dict(n_rank = 200),
#                        stop=dict(dgap_tol=5e-6, max_iter=100000), scr_type = "GAP", switching='default')
        # MEG
#        default =  dict(dict_type = 'MEG', data_type = 'bernoulli-gaussian', lasso=lasso,N=204, K=8193,\
#                        stop=dict(dgap_rel_tol=1e-5, max_iter=10000), scr_type = "GAP", switching='default')
        default =  dict(dict_type = 'MEG', data_type = 'bernoulli-gaussian', lasso=lasso,N=204, K=8193,\
                        stop=dict(dgap_tol=1e-4, max_iter=10000), scr_type = "GAP", switching='default', algo_type = 'FISTA')
        # Test
#        default =  dict(dict_type = 'gnoise', data_type = 'bernoulli-gaussian', lasso=lasso, N=2500, K=10000,\
#                        stop=dict(dgap_tol=1e-5, max_iter=1000), scr_type = "ST1", switching='default')
#        default = dict(dict_type = 'sukro',data_type = 'gnoise', lasso=lasso, N=2500,K=10000,scr_type = "GAP",\
#                    dict_params = dict(N1 = 50, N2 = 50, K1 = 100, K2 = 100,n_kron = 20),\
#                    stop=dict(dgap_tol=5e-6, max_iter=1000), switching='default') #, wstart=True)

                        
        expe = mergeopt(default, opt, keywords)
        expeScrRate(opt=expe)

def second_sukro(opt=dict(), **keywords):
    '''
    Plot the normalized time and flops for a synthetic Lasso problem
    versus the penalization parameter \lambda/\lambda_*
    you can chosse the dictionary to be gaussian noise or pnoise
    '''        
    np.random.seed(0)
    # SuKro
#    default = dict(dict_type = 'sukro',data_type = 'bernoulli-gaussian', N=2500,K=10000,scr_type = "GAP",\
#                    dict_params = dict(N1 = 50, N2 = 50, K1 = 100, K2 = 100,n_kron = 20),nbRuns=10,\
#                    stop=dict(dgap_tol=5e-6, max_iter=1000), switching='default') #, wstart=True)
    # Low-rank
#    default = dict(dict_type = 'low-rank',data_type = 'bernoulli-gaussian', N=2500,K=10000,scr_type = "ST1",\
#                    dict_params = dict(n_rank = 200),nbRuns=100,switching='default')
    # MEG
#    default =  dict(dict_type = 'MEG', data_type = 'bernoulli-gaussian',N=204, K=8193,
#                    stop=dict(dgap_rel_tol=1e-5, max_iter=10000), scr_type = "GAP", switching='default',nbRuns=100,
#                    samp=20, min_reg=0.01, samp_type='log')
    # Teste - Real Data
#    default =  dict(dict_type = 'low-rank', data_type = 'MNIST',N=784, K=60000,
#                    #data_params = dict(p=0.2),
#                    stop=dict(rel_tol=1e-8, max_iter=10000), scr_type = "GAP", switching='default',nbRuns=1,
#                    samp=5, min_reg=0.01, samp_type='log')
                    
    # Teste
#    default = dict(dict_type = 'sukro',data_type = 'bernoulli-gaussian', N=2500,K=10000,scr_type = "GAP",\
#                    dict_params = dict(N1 = 50, N2 = 50, K1 = 100, K2 = 100,n_kron = 20),nbRuns=1,\
#                    stop=dict(dgap_tol=5e-6, max_iter=1000), switching='default') #, wstart=True)
    default = dict(dict_type = 'sukro',data_type = 'bernoulli-gaussian', N=2500,K=10000,scr_type = "GAP",\
                    dict_params = dict(N1 = 50, N2 = 50, K1 = 100, K2 = 100,n_kron = 20),nbRuns=1,\
                    stop=dict(dgap_tol=1e-5, max_iter=1000), switching='default',
                    samp=5, min_reg=0.01, samp_type='log') #, wstart=True)
    
    expe = mergeopt(opt, default, keywords)
    res = runLambdas(opt=expe)
    traceLambdas(res['timeRes'], res['nbIter'], res['nbFlops'] ,expe )    

def second_sukro_per_it(opt=dict(), **keywords):
    '''
    Plot the normalized time per iteration for a synthetic Lasso problem
    versus the penalization parameter \lambda/\lambda_*
    you can chosse the dictionary to be gaussian noise or pnoise
    '''        
    #np.random.seed(0)
    default = dict(dict_type = 'sukro',data_type = 'bernoulli-gaussian', N=2500,K=10000,scr_type = "GAP",\
                    dict_params = dict(N1 = 50, N2 = 50, K1 = 100, K2 = 100,n_kron = 20),nbRuns=1,\
                    stop=dict(dgap_tol=5e-6, max_iter=1000), switching='default', samp=100) #,  wstart=True)
    #default = dict(dict_type = 'low-rank',data_type = 'bernoulli-gaussian', N=2500,K=10000,scr_type = "ST1",\
    #                dict_params = dict(n_rank = 200),nbRuns=100,switching='default')
    expe = mergeopt(opt, default, keywords)
    res = runLambdas_per_it(opt=expe)
    #traceLambdas(res['timeRes'], res['nbIter'], res['nbFlops'] ,expe )
    traceLambdas_per_it(res['timePerIt'], res['nbIter'],expe )
    
def third(opt=dict(), **keywords):
    '''
    Plot the normalized time and flops for a synthetic Group-Lasso problem 
    versus the penalization parameter \lambda/\lambda_*
    '''             
    default = dict(dict_type = 'pnoise',data_type = 'pnoise', N=1000,K=5000, Gr = 1, grsize = 10, sparse= 0.05)
    expe = mergeopt(default,opt, keywords)
    res = runLambdas(opt=expe)
    traceLambdas(res['timeRes'], res['nbIter'], res['nbFlops'] ,expe )
    

def estimateRC(D,opt,total_it=1000, verbose=False):
    """
    Estimating practical Relative Complexity of D (the fast dictionary)
    """
    
    if verbose:    
        print "Estimating RC with %d runs"%(total_it)
    
    mean_time_dense_dot, mean_time_denseT_dot = 0,0
    mean_time_dense, mean_time_denseT, mean_time_struct, mean_time_structT = 0,0,0,0

    screen = np.ones(opt['K'],dtype=np.int)
    D_dense = Dict(np.random.randn(opt['N'],opt['K']))
    for k in range(total_it):
        p = float(k+1)/total_it # testing over different sparsity levels
        
        x = np.zeros((opt['K'],1))
        nz = max(int(p*opt['K']),1) # not really bernoulli because the number of active features is deterministic here.
        idx = opt['K']*np.random.rand(1,nz)
        x[idx.astype(int)] = np.random.randn(nz,1)

        xT = np.zeros((opt['N'],1))        
        nz = max(int(p*opt['N']),1) # not really bernoulli because the number of active features is deterministic here.
        idx = opt['N']*np.random.rand(1,nz)
        xT[idx.astype(int)] = np.random.randn(nz,1)    
        
        #x = np.random.randn(opt['K'],1) # results do not change if x is sparse
        #xT = np.random.randn(opt['N'],1)
        
        ### Dense dictionary with np.dot ###        
        tic = time.time()    
        D_dense.Apply(x)
        toc = time.time()
        mean_time_dense_dot = mean_time_dense_dot + (toc-tic)/float(total_it)
        tic = time.time()    
        D_dense.ApplyTranspose(xT)
        toc = time.time()
        mean_time_denseT_dot = mean_time_denseT_dot + (toc-tic)/float(total_it)        
        ### Dense dictionary ###        
        tic = time.time()    
        D_dense.ApplyScreen(x,screen) #D_dense.Apply(x)
        toc = time.time()
        mean_time_dense = mean_time_dense + (toc-tic)/float(total_it)
        tic = time.time()    
        D_dense.ApplyTransposeScreen(xT,screen) #D_dense.ApplyTranspose(xT)
        toc = time.time()
        mean_time_denseT = mean_time_denseT + (toc-tic)/float(total_it)        
        ### Fast Dictionary ###
        tic = time.time()    
        D.ApplyScreen(x,screen) #D.Apply(x)
        toc = time.time()
        mean_time_struct = mean_time_struct + (toc-tic)/float(total_it)
        tic = time.time()    
        D.ApplyTransposeScreen(xT,screen) #D.ApplyTranspose(xT)
        toc = time.time()
        mean_time_structT = mean_time_structT + (toc-tic)/float(total_it)            
    RC_dot = mean_time_struct/mean_time_dense_dot # Comparing to multiplying with np.dot
    RCT_dot = mean_time_structT/mean_time_denseT_dot
    RC = mean_time_struct/mean_time_dense
    RCT = mean_time_structT/mean_time_denseT
    
    if verbose:
        print "RC_dot = %1.3f"%(RC_dot)
        print "RCT_dot = %1.3f"%(RCT_dot)
        print "RC = %1.3f"%(RC)
        print "RCT = %1.3f"%(RCT)
    return RC_dot

def expeScrRate(opt={},**keywords):
    """
    Execute the "screening progression" experiment
    """
    # handle  options
    default = default_expe()
    opt = mergeopt(opt, default, keywords)
    testopt(opt)
    for key, val in opt.items():
        exec(key+'='+ repr(val))
   
    problem, opt = GP.generate(opt) # Generate the problem
    
    # Evaluate RC (Relative Complexity)
    if  opt['dict_type'] is 'MEG':
        RC = 1.0/16 #TODO load from dataset
    #if opt['dict_type']  in {'sukro','low-rank'}:
    elif False:
        total_it = 100
        RC = estimateRC(problem.D,opt,total_it,verbose=True)
    else: # otherwise we just assume RC = 0.5
        RC = 0.5

        
    timeRes, nbIteration, switchIt, sols, flops, dynamicRun, \
    dynamicRun_approx, dynamicRun_approx2, dynamicRun_approx3, noScreenRun = \
        run3versions(problem, RC, opt, warm_start = None)   
    
    if problem.__class__.__name__ == 'GroupLasso':
        print '\n time to compute group norms: %.3f ms'%(opt['matNormTime']*1000)
    print "\ntime to compute with no screening : %.3f ms in %d iterations"%(timeRes['noScreen']*1000, nbIteration['noScreen']) 
    print "time to compute with static screening : %.3f ms in %d iterations"%(timeRes['static']*1000, nbIteration['static'])
    print "time to compute with dynamic screening : %.3f ms in %d iterations"%(timeRes['dynamic']*1000,nbIteration['dynamic'] ) 
    print "time to compute with approximate dynamic screening : %.3f ms in %d iterations"%(timeRes['dynamic_approx']*1000,nbIteration['dynamic_approx'] ) 
    
    print "\n__DYNAMIC__"
    print "wrt no screen: time gain : %.3f, theoretical gain %.3f"\
        %(timeRes['noScreen']/timeRes['dynamic'],\
        float(flops['noScreen'])/flops['dynamic'])
    print "wrt static: time gain : %.3f, theoretical gain %.3f"\
        %(timeRes['static']/timeRes['dynamic'],
        float(flops['static'])/flops['dynamic'])

    print "\n__DYNAMIC APPROX 1e-1__"
    print "wrt no screen: time gain : %.3f, theoretical gain %.3f"\
        %(timeRes['noScreen']/timeRes['dynamic_approx'],\
        float(flops['noScreen'])/flops['dynamic_approx'])
    print "wrt static: time gain : %.3f, theoretical gain %.3f"\
        %(timeRes['static']/timeRes['dynamic_approx'],
        float(flops['static'])/flops['dynamic_approx'])
        
    print "\n__DYNAMIC APPROX 1e-2__"
    print "wrt no screen: time gain : %.3f, theoretical gain %.3f"\
        %(timeRes['noScreen']/timeRes['dynamic_approx2'],\
        float(flops['noScreen'])/flops['dynamic_approx2'])
    print "wrt static: time gain : %.3f, theoretical gain %.3f"\
        %(timeRes['static']/timeRes['dynamic_approx2'],
        float(flops['static'])/flops['dynamic_approx2'])

    print "\n__DYNAMIC APPROX 1e-3__"
    print "wrt no screen: time gain : %.3f, theoretical gain %.3f"\
        %(timeRes['noScreen']/timeRes['dynamic_approx3'],\
        float(flops['noScreen'])/flops['dynamic_approx3'])
    print "wrt static: time gain : %.3f, theoretical gain %.3f"\
        %(timeRes['static']/timeRes['dynamic_approx3'],
        float(flops['static'])/flops['dynamic_approx3'])
            
    #### saving data and plotting result  
    if 'ResSynthData' not in os.listdir('./'):
        os.mkdir('ResSynthData')
    np.savez('./ResSynthData/'+make_file_name(opt)+'_lambda_'+str(opt['lasso'])+'.npz',\
        scrRate=dynamicRun['screenrate'],radius=dynamicRun['radius'],\
        ObjValue=dynamicRun['objective'], opt=opt, RC=RC,\
        nbIter = dynamicRun['nbIter'],\
        zeros = dynamicRun['zeros'], dGaps = dynamicRun['dGaps'],\
        scrRate_approx=dynamicRun_approx['screenrate'],scrRate_est_approx=dynamicRun_approx['screenrate_est'],
        radius_approx=dynamicRun_approx['radius'], ObjValue_approx=dynamicRun_approx['objective'], \
        nbIter_approx = dynamicRun_approx['nbIter'], switch_it = dynamicRun_approx['switch_it'],\
        zeros_approx = dynamicRun_approx['zeros'], dGaps_approx = dynamicRun_approx['dGaps'],\
        dGaps_est_approx = dynamicRun_approx['dGaps_est'], #TESTE dgap_est
        scrRate_approx2=dynamicRun_approx2['screenrate'],scrRate_est_approx2=dynamicRun_approx2['screenrate_est'],
        radius_approx2=dynamicRun_approx2['radius'], ObjValue_approx2=dynamicRun_approx2['objective'], \
        nbIter_approx2 = dynamicRun_approx2['nbIter'], switch_it2 = dynamicRun_approx2['switch_it'],\
        zeros_approx2 = dynamicRun_approx2['zeros'], dGaps_approx2 = dynamicRun_approx2['dGaps'],\
        dGaps_est_approx2 = dynamicRun_approx2['dGaps_est'], #TESTE dgap_est
        scrRate_approx3=dynamicRun_approx3['screenrate'],scrRate_est_approx3=dynamicRun_approx3['screenrate_est'],
        radius_approx3=dynamicRun_approx3['radius'], ObjValue_approx3=dynamicRun_approx3['objective'],\
        nbIter_approx3 = dynamicRun_approx3['nbIter'], switch_it3 = dynamicRun_approx3['switch_it'],\
        zeros_approx3 = dynamicRun_approx3['zeros'], dGaps_approx3 = dynamicRun_approx3['dGaps'],\
        dGaps_est_approx3 = dynamicRun_approx3['dGaps_est']) #TESTE dgap_est

    matplotlib.rc('axes', labelsize = 'xx-large')
    matplotlib.rc('xtick', labelsize = 20)
    matplotlib.rc('ytick', labelsize = 20)
    matplotlib.rc('axes', titlesize = 'xx-large')
    matplotlib.rc('lines', linewidth = 2)
    #plt.figure()
    f , (axScreen, axFlops_it) = \
        plt.subplots(2,1,sharex=True,figsize=2*plt.figaspect(1.3))    
    K = problem.D.shape[1]
    N = problem.D.shape[0]    
    length = dynamicRun['nbIter']+1
    length_approx = dynamicRun_approx['nbIter']+1
    length_approx2 = dynamicRun_approx2['nbIter']+1
    length_approx3 = dynamicRun_approx3['nbIter']+1
    markers_on1 = [dynamicRun_approx['switch_it'] -1] #, length_approx-1]
    markers_on2 = [dynamicRun_approx2['switch_it']-1] #, length_approx2-1]
    markers_on3 = [dynamicRun_approx3['switch_it']-1] #, length_approx3-1]
    
#    plt.plot(np.arange(length),(1 - dynamicRun['screenrate'])*K)
#    markers_on = [dynamicRun_approx['switch_it']]
#    plt.plot(np.arange(length_approx),(1 - dynamicRun_approx['screenrate'])*K, '-x', markevery=markers_on) # Marker on the swithing point 
#    plt.plot(np.arange(length_approx2),(1 - dynamicRun_approx2['screenrate'])*K, '-x', markevery=markers_on) 
#    plt.plot(np.arange(length_approx3),(1 - dynamicRun_approx3['screenrate'])*K, '-x', markevery=markers_on)
#    plt.axis([0,length, 0, K*1.1])
#    plt.grid(True) 
#    plt.ylabel("Size of the dictionary")
#    plt.xlabel("Iteration t")
#    plt.savefig('./ResSynthData/Simu_screenRate_'+make_file_name(opt)+\
#        '_lasso'+str(opt['lasso'])+'approx.pdf',bbox_inches = 'tight',bbox_pad = 2 )

    flops_ns = flop_calc_it("noScreen",K,N,[], noScreenRun['zeros'],[]) #float(flops["noScreen"])/nbIteration["noScreen"]
    flops_d = flop_calc_it("dynamic",K,N,dynamicRun['screenrate'], dynamicRun['zeros'],[])
    flops_d1 = flop_calc_it("dynamic",K,N,dynamicRun_approx['screenrate'], dynamicRun_approx['zeros'],[],RC,dynamicRun_approx['switch_it'])
    flops_d2 = flop_calc_it("dynamic",K,N,dynamicRun_approx2['screenrate'], dynamicRun_approx2['zeros'],[],RC,dynamicRun_approx2['switch_it'])
    flops_d3 = flop_calc_it("dynamic",K,N,dynamicRun_approx3['screenrate'], dynamicRun_approx3['zeros'],[],RC,dynamicRun_approx3['switch_it'])
           
    axScreen.plot(np.arange(length),(1 - dynamicRun['screenrate'])*K, '-k', linewidth = 6) #, 'x', markevery=[length-1]
    axScreen.plot(np.arange(length_approx),(1 - dynamicRun_approx['screenrate'])*K, '-mx', markevery=markers_on1) # Marker on the swithing point 
    axScreen.plot(np.arange(length_approx2),(1 - dynamicRun_approx2['screenrate'])*K, '-gx', markevery=markers_on2) 
    axScreen.plot(np.arange(length_approx3),(1 - dynamicRun_approx3['screenrate'])*K, '-rx', markevery=markers_on3)

    axScreen.axis([0,max(length,length_approx,length_approx2,length_approx3), 0, K*1.1])
    axScreen.grid(True) 
    axScreen.set_ylabel("Size of the dictionary")
    axScreen.set_xlabel("Iteration t")

    #axFlops_it.plot(np.arange(length),flops_ns,'-b')
    axFlops_it.plot(np.arange(length),flops_d, '-k', linewidth = 6) #markevery=[length-1])
    axFlops_it.plot(np.arange(length_approx),flops_d1, '-mx', markevery=markers_on1)
    axFlops_it.plot(np.arange(length_approx2),flops_d2, '-gx', markevery=markers_on2) 
    axFlops_it.plot(np.arange(length_approx3),flops_d3, '-rx', markevery=markers_on3)
    axFlops_it.grid(True)         
    #axFlops_it.set_ylim((-0.19,1.15))
    axFlops_it.set_ylim((0,1.1*max(max(flops_d),max(flops_d1),max(flops_d2),max(flops_d3))))
    axFlops_it.set_ylabel("Flops number") #,fontsize = 24)
    axFlops_it.set_xlabel("Iteration t")
    #axFlops_it.legend(fontsize=22,loc=3,frameon=False)

    f.savefig('./ResSynthData/Simu_screenRate_'+make_file_name(opt)+\
        '_lasso'+str(opt['lasso'])+'approx.pdf',bbox_inches = 'tight',bbox_pad = 2 )
    
#    plt.figure()
#    plt.plot(dynamicRun['objective'].flatten())
    return dynamicRun
    
def flop_calc_it(EmbedTest,K,N,screenrate,zeros,Gr,RC=1,switch_it=0): # Returns the iteration complexity
    nb_gr = len(Gr)
    if EmbedTest == 'dynamic':
        if RC == 1:
            flops = ( N*K*(2 - np.asarray(screenrate) -
                np.asarray(zeros)/K) + \
                    6*(1-np.asarray(screenrate)) + 5*N + 5*nb_gr)                   
                    #6*K + 5*N + 5*nb_gr)
        else:
            flops = ( N*K*(RC + 1 -
                        np.asarray(zeros[0:switch_it])/K) + \
                        7*(1-np.asarray(screenrate[0:switch_it])) + 5*N + 5*nb_gr)
                        #7*K + 5*N + 5*nb_gr)
                        
            flops_bis = ( N*K*(2 - np.asarray(screenrate[switch_it:]) -
                        np.asarray(zeros[switch_it:])/K) + \
                        6*(1-np.asarray(screenrate[switch_it:])) + 5*N + 5*nb_gr)
                        #6*K + 5*N + 5*nb_gr)
            flops = np.append(flops,flops_bis)
    elif EmbedTest == 'static':
        flops = ( N*K*(2 - np.asarray(zeros)/K - 
            np.asarray(screenrate) ) + 4*K + N + 3*nb_gr)
                 #+ N*K But this
    else:
        flops = ( N*K*(2- np.asarray(zeros)/K) + 4*K + N + 3*nb_gr)
        
    return flops            
    
    
def traceLambdas(timeRes, nbIter, nbFlops, opt):
    """
    Plot the normalized time resulting from the experiment.
    Compute it of not done.
    """    
    ####  Handling  options
#    for key, val in opt.items():
#        exec(key+'='+ repr(val))
        
    
    ####  plot the results
    matplotlib.rc('axes', labelsize = 24)
    matplotlib.rc('axes', titlesize = 24)
    matplotlib.rc('xtick', labelsize = 20)
    matplotlib.rc('ytick', labelsize = 20)
    matplotlib.rc('lines', linewidth = 3)
    #matplotlib.rc('text', usetex=True)
    matplotlib.rc('mathtext', fontset='cm')

    markersize = 12
    
   
    q0_d, q1_d, q2_d, q3_d, q4_d = np.percentile(\
        timeRes['dynamic']/timeRes['noScreen'],[10,25,50,75,90],axis=1)
    q0_s, q1_s, q2_s, q3_s, q4_s = np.percentile(\
        timeRes['static']/timeRes['noScreen'],[10,25,50,75,90],axis=1)
    q0_d1, q1_d1, q2_d1, q3_d1, q4_d1 = np.percentile(\
        timeRes['dynamic_approx']/timeRes['noScreen'],[10,25,50,75,90],axis=1)
    q0_d2, q1_d2, q2_d2, q3_d2, q4_d2 = np.percentile(\
        timeRes['dynamic_approx2']/timeRes['noScreen'],[10,25,50,75,90],axis=1)
    q0_d3, q1_d3, q2_d3, q3_d3, q4_d3 = np.percentile(\
        timeRes['dynamic_approx3']/timeRes['noScreen'],[10,25,50,75,90],axis=1)
    
    
    flop_q0_d, flop_q1_d, flop_q2_d, flop_q3_d, flop_q4_d =\
            np.percentile(nbFlops['dynamic'].astype(float)
            /nbFlops['noScreen'],[10,25,50,75,90],axis=1)
    flop_q0_s, flop_q1_s, flop_q2_s, flop_q3_s, flop_q4_s =\
            np.percentile(nbFlops['static'].astype(float)
            /nbFlops['noScreen'],[10,25,50,75,90],axis=1)
    flop_q0_d1, flop_q1_d1, flop_q2_d1, flop_q3_d1, flop_q4_d1 =\
            np.percentile(nbFlops['dynamic_approx'].astype(float)
            /nbFlops['noScreen'],[10,25,50,75,90],axis=1)
    flop_q0_d2, flop_q1_d2, flop_q2_d2, flop_q3_d2, flop_q4_d2 =\
            np.percentile(nbFlops['dynamic_approx2'].astype(float)
            /nbFlops['noScreen'],[10,25,50,75,90],axis=1)
    flop_q0_d3, flop_q1_d3, flop_q2_d3, flop_q3_d3, flop_q4_d3 =\
            np.percentile(nbFlops['dynamic_approx3'].astype(float)
            /nbFlops['noScreen'],[10,25,50,75,90],axis=1)

    itq1_d,it_median_d ,itq3_d= np.percentile(nbIter['dynamic'] ,[25,50,75],axis=1)
    itq1_s,it_median_s ,itq3_s= np.percentile(nbIter['noScreen'] ,[25,50,75],axis=1)            
       
    f , (axTime, axFlops) = \
        plt.subplots(2,1,sharex=True,figsize=2*plt.figaspect(1.3))     

    pen_param_list = make_pen_param_list(opt['samp'],opt['min_reg'],opt['samp_type'])  
    mkevry = max(1,len(pen_param_list)/10)


    if opt['Gr']:
            Gstr = 'G'
    else:
            Gstr =''
                        
    ## Time plot
    if opt['dict_type']  in {'sukro','low-rank'}: 
        f = plt.figure(figsize=1.05*plt.figaspect(0.6))
        # Dynamic
        plt.plot(pen_param_list,q2_d,'ks-' ,markevery= mkevry,markersize = markersize)  
        plt.fill_between(pen_param_list, q1_d, q3_d,alpha = 0.2,facecolor = 'k')       
        plt.fill_between(pen_param_list, q1_d, q3_d,alpha = 0.1, 
                            color = 'none',edgecolor = 'k', hatch = '/')
        # Static         
#        plt.plot(pen_param_list,  q2_s, 'bo-' ,markevery= mkevry,
#                    markersize = markersize) 
#        plt.fill_between(pen_param_list, q1_s, q3_s,alpha = 0.2, facecolor = 'b' )
#        plt.fill_between(pen_param_list, q1_s, q3_s,alpha = 0.1, edgecolor = 'b', 
#                            color = 'none', hatch = '\\')
        # Dynamic approx 1e-1
        plt.plot(pen_param_list, q2_d1,'m^-',
                 label = 'A-D'+Gstr+opt['scr_type']+r'$: \| \mathbf{e}_j \|\!=\!10^{-1}$', 
                 markevery= mkevry, markersize = markersize)  
        plt.fill_between(pen_param_list, q1_d1, q3_d1,alpha = 0.2,
                         facecolor = 'm')   
        # Dynamic approx 1e-2
        plt.plot(pen_param_list, q2_d2,'gD-',
                 label = 'A-D'+Gstr+opt['scr_type']+r'$: \| \mathbf{e}_j \|\!=\!10^{-2}$', 
                 markevery= mkevry, markersize = markersize)  
        plt.fill_between(pen_param_list, q1_d2, q3_d2,alpha = 0.2,
                         facecolor = 'g')           
        # Dynamic approx 1e-3
        plt.plot(pen_param_list, q2_d3,'rv-',
                 label = 'A-D'+Gstr+opt['scr_type']+r'$: \| \mathbf{e}_j \|\!=\!10^{-3}$', 
                 markevery= mkevry, markersize = markersize)  
        plt.fill_between(pen_param_list, q1_d3, q3_d3,alpha = 0.2,
                         facecolor = 'r')           

        if opt['samp_type'] is not 'linear':
            plt.xscale('log')                
        
        plt.grid(True)         
        plt.ylim((0,1.15))
        plt.title("Normalized running times") 
        plt.legend(fontsize=22,loc=3,frameon=False)
        
        if 'ResSynthData' not in os.listdir('./'):
            os.mkdir('ResSynthData')
        f.savefig('./ResSynthData/'+make_file_name(opt)+'_Time_relNoScreen.pdf',bbox_inches = 'tight' )
    
    ## Flops plot
    # Static
##    axFlops.plot(pen_param_list,flop_q2_s,'bo-' ,
##                 label = opt['algo_type'] + ' + '+Gstr+opt['scr_type'],                    
##                 markevery= mkevry, markersize = markersize) 
##    axFlops.fill_between(pen_param_list,flop_q1_s, flop_q3_s,alpha = 0.2,
##                         facecolor = 'b')
##    axFlops.fill_between(pen_param_list,flop_q1_s, flop_q3_s,alpha = 0.1, 
##                         edgecolor = 'b', color = 'none', hatch = '\\')
#    # Dynamic                     
#    axFlops.plot(pen_param_list,flop_q2_d,'ks-',
#                 label = opt['algo_type'] + ' + D'+Gstr+opt['scr_type'],                 
#                 markevery= mkevry, markersize = markersize)  
#    axFlops.fill_between(pen_param_list, flop_q1_d, flop_q3_d,alpha = 0.2,
#                         facecolor = 'k')
#    axFlops.fill_between(pen_param_list,flop_q1_d, flop_q3_d,alpha = 0.1,
#                         edgecolor = 'k', hatch = '/',color='none')
#    # Dynamic approx 1e-1
#    axFlops.plot(pen_param_list,flop_q2_d1,'m^-',
#                 label = opt['algo_type'] + ' + D'+Gstr+opt['scr_type'],                 
#                 markevery= mkevry, markersize = markersize)  
#    axFlops.fill_between(pen_param_list, flop_q1_d1, flop_q3_d1,alpha = 0.2,
#                         facecolor = 'm')
##    axFlops.fill_between(pen_param_list,flop_q1_d1, flop_q3_d1,alpha = 0.1,
##                         edgecolor = 'm', hatch = '/',color='none')
#    # Dynamic approx 1e-2
#    axFlops.plot(pen_param_list,flop_q2_d2,'gD-',
#                 label = opt['algo_type'] + ' + D'+Gstr+opt['scr_type'],                 
#                 markevery= mkevry, markersize = markersize)  
##    axFlops.fill_between(pen_param_list, flop_q1_d2, flop_q3_d2,alpha = 0.2,
##                         facecolor = 'g')
##    axFlops.fill_between(pen_param_list,flop_q1_d2, flop_q3_d2,alpha = 0.1,
##                         edgecolor = 'g', hatch = '/',color='none')
#    # Dynamic approx 1e-3
#    axFlops.plot(pen_param_list,flop_q2_d3,'rv-',
#                 label = opt['algo_type'] + ' + D'+Gstr+opt['scr_type'],                 
#                 markevery= mkevry, markersize = markersize)  
#    axFlops.fill_between(pen_param_list, flop_q1_d3, flop_q3_d3,alpha = 0.2,
#                         facecolor = 'r')
##    axFlops.fill_between(pen_param_list,flop_q1_d3, flop_q3_d3,alpha = 0.1,
##                         edgecolor = 'r', hatch = '/',color='none')                         
#
#       
#    axFlops.grid(True)         
#    axFlops.set_ylim((-0.19,1.15))
#    axFlops.set_ylabel("Normalized flops number",fontsize = 24)
#    axFlops.set_xlabel(r"$\lambda/\lambda_*$")
#    axFlops.legend(fontsize=22,loc=3,frameon=False)
#           
#    f.suptitle(type2name(opt['dict_type']),fontsize=28)


    # Dynamic      
    f = plt.figure(figsize=1.05*plt.figaspect(0.6))                   
    plt.plot(pen_param_list,flop_q2_d,'ks-',
                 label = 'D'+Gstr+opt['scr_type'],                 
                 markevery= mkevry, markersize = markersize)  
    plt.fill_between(pen_param_list, flop_q1_d, flop_q3_d,alpha = 0.2,
                         facecolor = 'k')
#    plt.fill_between(pen_param_list,flop_q1_d, flop_q3_d,alpha = 0.1,
#                         edgecolor = 'k', hatch = '/',color='none')
    # Dynamic approx 1e-1
    plt.plot(pen_param_list,flop_q2_d1,'m^-',
                 label = 'A-D'+Gstr+opt['scr_type']+r'$: \| \mathbf{e}_j \|\!=\!10^{-1}$', 
                 markevery= mkevry, markersize = markersize)  
    plt.fill_between(pen_param_list, flop_q1_d1, flop_q3_d1,alpha = 0.2,
                         facecolor = 'm')
#    plt.fill_between(pen_param_list,flop_q1_d1, flop_q3_d1,alpha = 0.1,
#                         edgecolor = 'm', hatch = '/',color='none')
    # Dynamic approx 1e-2
    plt.plot(pen_param_list,flop_q2_d2,'gD-',
                 label = 'A-D'+Gstr+opt['scr_type']+ r'$: \| \mathbf{e}_j \|\!=\!10^{-2}$',                 
                 markevery= mkevry, markersize = markersize)  
    plt.fill_between(pen_param_list, flop_q1_d2, flop_q3_d2,alpha = 0.2,
                         facecolor = 'g')
#    plt.fill_between(pen_param_list,flop_q1_d2, flop_q3_d2,alpha = 0.1,
#                         edgecolor = 'g', hatch = '/',color='none')
    # Dynamic approx 1e-3
    plt.plot(pen_param_list,flop_q2_d3,'rv-',
                 label = 'A-D'+Gstr+opt['scr_type']+r'$: \| \mathbf{e}_j \|\!=\!10^{-3}$',                 
                 markevery= mkevry, markersize = markersize)  
    plt.fill_between(pen_param_list, flop_q1_d3, flop_q3_d3,alpha = 0.2,
                         facecolor = 'r')
#    plt.fill_between(pen_param_list,flop_q1_d3, flop_q3_d3,alpha = 0.1,
#                         edgecolor = 'r', hatch = '/',color='none')                         

    if opt['samp_type'] is not 'linear':
        plt.xscale('log')
       
    plt.grid(True)         
    plt.ylim((0,1.15))
    plt.ylabel("Normalized flops number",fontsize = 24)
    plt.xlabel(r"$\lambda/\lambda_*$")
    plt.legend(fontsize=20,loc=3,frameon=False)

    f.suptitle(type2name(opt['dict_type'],opt) + ' + ' + opt['algo_type'],fontsize=26)


    if 'ResSynthData' not in os.listdir('./'):
        os.mkdir('ResSynthData')
    f.savefig('./ResSynthData/'+make_file_name(opt)+'_Simu_relNoScreen.pdf',bbox_inches = 'tight' )
    if not opt['disp_fig']:
        plt.close()
        


def run3versions(problem=None, RC=1, opt={}, warm_start = None, **keywords):
    """
    Run the 3 versions of the Algorithm on the same problem recording the computation times
    """    
    ####  handles options
        
    default = default_expe()
    opt = mergeopt(opt, default, keywords)
    testopt(opt)

    if 'switching' not in opt.keys():
        opt['switching'] = 'default'
    
    print r'lambda/lambda* = '+str(opt['lasso']) 
    

    if opt['data_type']=='MNIST':
        mndata = MNIST('./datasets/MNIST/LeCun/')
        images_train,_ = mndata.load_training() #labels were ignored
        images_test,_ =mndata.load_testing()
        images_train = np.asarray(images_train).T
        
        opt['K'] = images_train.shape[1]
        opt['N'] = images_train.shape[0]
        
        D_bis = Dict(images_train)
        # Input vector 'y' is randomly chosen from test set
        idx = np.random.randint(len(images_test))
        y = np.expand_dims(np.asarray(images_test[idx]),axis=1)
        
        # Fast approximation of data
        if opt['dict_type']=='low-rank':
            n_rank = opt['dict_params']['n_rank']   # rank 200 => np.mean(normE) = 0.14
                                                    # rank 400 => np.mean(normE) = 0.04
            L,S,R = np.linalg.svd(D_bis.data, full_matrices=False)
            L = L[:,:n_rank]*S[:n_rank]
            R = R[:n_rank,:]
            D_lowrank = L.dot(R)
            D = Dict(D_lowrank,opType="low-rank",params=dict(L=L,R=R))
            del L,R #,D_lowrank

        problem, opt = GP.generate(opt,D=D,y=y,D_bis=D_bis)
        if warm_start: raise ValueError('Warm start is not available for this configuration, since the input vector y is regenerated every time.')
                    
        E = D.data - D_bis.data
        normE = np.linalg.norm(E,axis=0)
        norm2E = np.linalg.norm(E,2)
        del E
            
        res_dyn_approx = solver_approx( problem=problem, normE=normE, norm2E = norm2E, RC=RC, L=opt['L'], stop=opt['stop'], switching=opt['switching'], \
                                        scr_type=opt['scr_type'], \
                                        EmbedTest='dynamic', algo_type=opt['algo_type'], \
                                        warm_start = warm_start)
        res_dyn_approx2 = res_dyn_approx #TODO delete and test existence of this variable when used elsewhere
        res_dyn_approx3 = res_dyn_approx
    elif opt['dict_type']=='MEG':
        meg_data = sio.loadmat('./datasets/MEG/X_meg.mat')
        D = meg_data['X_fixed'] # unstructured MEG matrix
        
        opt['K'] = D.shape[1]
        opt['N'] = D.shape[0]
        
        D = Dict(D)
        
        # Fast approximation of dictionary
        meg_data = sio.loadmat('./datasets/MEG/faust_approx/M_16.mat') #M_25, M_16, M_8, M_6
        facts = meg_data['facts']
        del meg_data
        
        # Develloping FAuST fators
        n_layers = facts.shape[1]
        D_bis = facts[:,0][0];
        for k in range(1,n_layers):
            assert(facts[:,k][0].shape[0] == D_bis.shape[1])
            D_bis = D_bis.dot(facts[:,k][0])
                
        D_bis = Dict(D_bis.T) #TODO create opType="faust"
        
        # Test synthetic error matrix
#        E = np.sqrt(1./opt['N'])*np.random.randn(opt['N'],opt['K'])
#        normE = 1e-3
#        D_bis = D.data + normE*E
#        D_bis = Dict(D_bis)

        # Load 'y' for benchmark with Matlab
#        meg_data = sio.loadmat('./datasets/MEG/faust_approx/Data.mat')
#        y = meg_data['Data'][:,0]
#        y = np.expand_dims(y, -1)
#        del meg_data

        problem, opt = GP.generate(opt,D=D,D_bis=D_bis)
        if warm_start: raise ValueError('Warm start is not available for this configuration, since the input vector y is regenerated every time.')
                    
        E = D.data - D_bis.data
        normE = np.linalg.norm(E,axis=0)
        norm2E = np.linalg.norm(E,2)
        del E
        
        problem.D, problem.D_bis = problem.D_bis, problem.D # start with the approximate (fast) dictionary 
        
        res_dyn_approx = solver_approx( problem=problem, normE=normE, norm2E = norm2E, RC=RC, L=opt['L'], stop=opt['stop'], switching=opt['switching'], \
                                        scr_type=opt['scr_type'], \
                                        EmbedTest='dynamic', algo_type=opt['algo_type'], \
                                        warm_start = warm_start)
        res_dyn_approx2 = res_dyn_approx #TODO delete and test existence of this variable when used elsewhere
        res_dyn_approx3 = res_dyn_approx

#        meg_data = sio.loadmat('./datasets/MEG/X_meg.mat')
#        D = meg_data['X_fixed'] # unstructured MEG matrix
#        
#        opt['K'] = D.shape[1]
#        opt['N'] = D.shape[0]
#            
#        D = Dict(D)
#        problem, opt = GP.generate(opt,D=D)
#        
#        filenames = ['./datasets/MEG/M_16.mat',
#              './datasets/MEG/M_8.mat',
#              './datasets/MEG/M_6.mat'];
#        res_dyn_approx_list = []
#              
#        for filename in filenames:
#            # After each run: D = slow, D_bis = fast
#
#            # Fast approximation of dictionary
#            meg_data = sio.loadmat(filename)
#            facts = meg_data['facts']
#            del meg_data
#            
#            # Develloping FAuST fators
#            n_layers = facts.shape[1]
#            D_bis = facts[:,0][0];
#            for k in range(1,n_layers):
#                assert(facts[:,k][0].shape[0] == D_bis.shape[1])
#                D_bis = D_bis.dot(facts[:,k][0])
#                    
#            D_bis = Dict(D_bis.T) #TODO create opType="faust"
#                        
#            E = D.data - D_bis.data
#            normE = np.linalg.norm(E,axis=0)
#            norm2E = np.linalg.norm(E,2)
#            del E
#            
#            problem.D_bis = D_bis
#            problem.D, problem.D_bis = problem.D_bis, problem.D # start with the approximate (fast) dictionary 
#            
#            res_dyn_approx_list.append( 
#                 solver_approx( problem=problem, normE=normE, norm2E = norm2E, RC=RC, L=opt['L'], stop=opt['stop'], switching=opt['switching'], \
#                     scr_type=opt['scr_type'], \
#                     EmbedTest='dynamic', algo_type=opt['algo_type'], \
#                     warm_start = warm_start))
#        
#        res_dyn_approx  = res_dyn_approx_list[0]
#        res_dyn_approx2 = res_dyn_approx_list[1]
#        res_dyn_approx3 = res_dyn_approx_list[2]

    else:        
        E = np.sqrt(1./opt['N'])*np.random.randn(opt['N'],opt['K'])
        norm2E = np.linalg.norm(E,2) 
        res_dyn_approx_list = []
        if opt['dict_type'] and problem.D.opType in {'sukro','low-rank'}: # From the second iteration in runLambdas, there is no need to swap anymore
            problem.D, problem.D_bis = problem.D_bis, problem.D # Put fast dict in D_bis initially (as it is the case after each run of solver_approx)
        for normE in np.array([1e-1,1e-2,1e-3]):
            if opt['dict_type'] in {'sukro','low-rank'}:
                # At this point: D = slow (=fast+E), D_bis = fast
                problem.D =  Dict(problem.D_bis.data + normE*E)
                problem.D.normalize()
                if opt['data_type']=='bernoulli-gaussian':                    
                    problem_bg, opt = GP.generate(opt,problem.D,D_bis=problem.D_bis) # Generate the problem. Attention!
                    problem.y = problem_bg.y # Otherwise the reference for object 'problem' changes, detaching it from the 'problem' outside this funtion
                    if warm_start: raise ValueError('Warm start is not available for this configuration, since the input vector y is regenerated every time.')
                problem.D, problem.D_bis = problem.D_bis, problem.D # start with the approximate (fast) dictionary                
            else:
                # At this point: D = slow, D_bis = fast (=slow+E)
                problem.D_bis =  Dict(problem.D.data + normE*E)
                problem.D_bis.normalize()    
                problem.D, problem.D_bis = problem.D_bis, problem.D # The approximate dictionary is used
            
            res_dyn_approx_list.append( 
                solver_approx(problem=problem, normE=normE, norm2E = normE*norm2E, RC=RC, L=opt['L'], stop=opt['stop'], switching=opt['switching'], \
                                    scr_type=opt['scr_type'], \
                                    EmbedTest='dynamic', algo_type=opt['algo_type'], \
                                    warm_start = warm_start))
        res_dyn_approx  = res_dyn_approx_list[0]
        res_dyn_approx2 = res_dyn_approx_list[1]
        res_dyn_approx3 = res_dyn_approx_list[2]                    
    
    # Exact Screening - Dynamic
    res_dyn = solver(problem=problem, L=opt['L'], stop=opt['stop'],\
                            scr_type=opt['scr_type'], \
                            EmbedTest='dynamic', algo_type=opt['algo_type'], \
                            warm_start = warm_start) 

    # Exact Screening - Static                  
    res_static = solver(problem=problem, L=opt['L'], stop=opt['stop'],\
                            scr_type=opt['scr_type'], \
                            EmbedTest='static', algo_type=opt['algo_type'], \
                            warm_start = warm_start)
    
    # No Screening                        
    res_noScreen = solver(problem=problem, L=opt['L'], stop=opt['stop'],\
                            scr_type=opt['scr_type'], \
                            EmbedTest='noScreen', algo_type=opt['algo_type'], \
                            warm_start = warm_start)  

    if opt['dict_type'] and problem.D_bis.opType in {'sukro','low-rank'}: # Put 'sukro' back to D
#    if opt['dict_type'] in {'sukro','low-rank'}: # Put 'sukro' back to D
        problem.D, problem.D_bis = problem.D_bis, problem.D


    timeRes     = { 'noScreen': res_noScreen['time'],
                    'static':   res_static['time'],
                    'dynamic':  res_dyn['time'],
                    'dynamic_approx':  res_dyn_approx['time'],
                    'dynamic_approx2':  res_dyn_approx2['time'],
                    'dynamic_approx3':  res_dyn_approx3['time']}
                
    nbIteration = { 'noScreen': res_noScreen['nbIter'],
                    'static':   res_static['nbIter'],
                    'dynamic':  res_dyn['nbIter'],
                    'dynamic_approx':  res_dyn_approx['nbIter'],
                    'dynamic_approx2':  res_dyn_approx2['nbIter'],
                    'dynamic_approx3':  res_dyn_approx3['nbIter']}
                    
    switchIt    = { 'noScreen': 0,
                    'static':   0,
                    'dynamic':  0,
                    'dynamic_approx':  res_dyn_approx['switch_it'],
                    'dynamic_approx2':  res_dyn_approx2['switch_it'],
                    'dynamic_approx3':  res_dyn_approx3['switch_it']}
                    
    flops       = { 'noScreen': res_noScreen['flops'],
                    'static':   res_static['flops'],
                    'dynamic':  res_dyn['flops'],
                    'dynamic_approx':  res_dyn_approx['flops'],
                    'dynamic_approx2':  res_dyn_approx2['flops'],
                    'dynamic_approx3':  res_dyn_approx3['flops']}            
    
    Solution    = { 'noScreen': res_noScreen['sol'],
                    'static':   res_static['sol'],
                    'dynamic':  res_dyn['sol'],
                    'dynamic_approx':  res_dyn_approx['sol'],
                    'dynamic_approx2':  res_dyn_approx2['sol'],
                    'dynamic_approx3':  res_dyn_approx3['sol']}       
    
    return timeRes,nbIteration,switchIt,Solution,flops,res_dyn, res_dyn_approx, res_dyn_approx2,res_dyn_approx3,res_noScreen

def runLambdas(opt={},**keywords):
    """
    Run the Algorithm for a fix problem with all the value of lambda/lambda*
    in pen_param_list
    
    Return the results for opt['nbRuns'] problems
    """
    #### handles options
    default = default_expe()
                
    opt = mergeopt(opt, default, keywords)
#    for key, val in opt.items():
#        exec(key+'='+ repr(val))
        

    pen_param_list = make_pen_param_list(opt['samp'],opt['min_reg'],opt['samp_type'])
    
    #### run the algorithm if needed
    if 'ResSynthData' in os.listdir('./') and \
       (not opt['recalc']) and (make_file_name(opt)+'_done.npz') in os.listdir('./ResSynthData'):
        print 'experiment already computed'        
        Data = np.load('./ResSynthData/'+make_file_name(opt)+'_done.npz' )
        if 'time' in Data.keys():
            timeRes = Data['time'][()]
        else:
            timeRes = Data['timeRes'][()]
        if 'matNormTime' in Data['opt'][()].keys():
            opt['matNormTime'] = Data['opt'][()]['matNormTime']
        nbIter = Data['nbIter'][()]
        nbFlops = Data['nbFlops'][()]
        if opt['disp_fig'] or not (make_file_name(opt)+'_Simu_relNoScreen.pdf') in\
            os.listdir('./ResSynthData'):
            traceLambdas( timeRes, nbIter, nbFlops ,opt )
        ret = { 'timeRes' : timeRes, \
                'nbIter': nbIter, \
                'nbFlops': nbFlops, \
                'opt': opt }
        return ret
 
    opt['lasso'] = 0.99           
    problem, opt = GP.generate(opt) 
    opt['K'] = problem.D.shape[1]
    
    avg = opt['nbRuns']
    nblambdas = len(pen_param_list)
    timeRes = dict(noScreen=np.zeros((nblambdas,avg), dtype = float),\
                static=np.zeros((nblambdas,avg), dtype = float),\
                dynamic=np.zeros((nblambdas,avg), dtype = float), \
                dynamic_approx=np.zeros((nblambdas,avg), dtype = float), \
                dynamic_approx2=np.zeros((nblambdas,avg), dtype = float), \
                dynamic_approx3=np.zeros((nblambdas,avg), dtype = float))
                
    nbIter = dict(noScreen=np.zeros((nblambdas,avg)),\
                static=np.zeros((nblambdas,avg)),\
                dynamic=np.zeros((nblambdas,avg)), \
                dynamic_approx=np.zeros((nblambdas,avg)), \
                dynamic_approx2=np.zeros((nblambdas,avg)), \
                dynamic_approx3=np.zeros((nblambdas,avg)))
                
    switchIter = dict(noScreen=np.zeros((nblambdas,avg)),\
                static=np.zeros((nblambdas,avg)),\
                dynamic=np.zeros((nblambdas,avg)), \
                dynamic_approx=np.zeros((nblambdas,avg)), \
                dynamic_approx2=np.zeros((nblambdas,avg)), \
                dynamic_approx3=np.zeros((nblambdas,avg)))
                
    nbFlops = dict(noScreen=np.zeros((nblambdas,avg), dtype = float),\
                static=np.zeros((nblambdas,avg), dtype = float),\
                dynamic=np.zeros((nblambdas,avg), dtype = float), \
                dynamic_approx=np.zeros((nblambdas,avg), dtype = float), \
                dynamic_approx2=np.zeros((nblambdas,avg), dtype = float), \
                dynamic_approx3=np.zeros((nblambdas,avg), dtype = float))                
                
    xFinal = dict(noScreen=np.zeros((nblambdas,avg,opt['K'])),\
                static=np.zeros((nblambdas,avg,opt['K'])),\
                dynamic=np.zeros((nblambdas,avg,opt['K'])),\
                dynamic_approx=np.zeros((nblambdas,avg,opt['K'])),\
                dynamic_approx2=np.zeros((nblambdas,avg,opt['K'])),\
                dynamic_approx3=np.zeros((nblambdas,avg,opt['K'])))
                
    sig = np.zeros((opt['nbRuns'],opt['N']))
    
    # Evaluate RC (Relative Complexity)
    if  opt['dict_type'] is 'MEG':
        RC = 1.0/16 #TODO load from dataset
#    if opt['dict_type'] in {'sukro','low-rank'}:
    elif False:
        total_it = 10000
        RC = estimateRC(problem.D,opt,total_it,verbose=True)
    else: # otherwise we just assume RC = 0.5
        RC = 0.5
    print "RC = %1.2f"%(RC)    

    for j in range(avg):
        start = None
        res = None
        problem, opt = GP.generate(opt, problem.D)

        star,lstar = problem.getStar() 
        sig[j,:] = problem.y.flatten()
        for i,lasso_ in enumerate(pen_param_list[::-1]):
            if not opt['wstart']:
                start = None
            elif res!=None:
                start = res['noScreen']
            opt['lasso'] = lasso_
            problem.pen_param = opt['lasso']*lstar
            timeIt, nbIt, switchIt, res, flops, junk, junk1, junk2, junk3, junk_ns = \
                run3versions(problem,RC,opt,start)
            for key in timeRes.iterkeys(): 
                timeRes[key][nblambdas-i-1,j] = timeIt[key]
                nbIter[key][nblambdas-i-1,j] = nbIt[key]
                switchIter[key][nblambdas-i-1,j] = switchIt[key]
                nbFlops[key][nblambdas-i-1,j] = flops[key]
                if not opt['dict_type'] =="MNIST"  :
                    xFinal[key][nblambdas-i-1,j] = res[key].flatten()
        print "problem %d over %d"%(j+1,avg)           
                 
    print('Done') 


    if 'ResSynthData' not in os.listdir('./'):
        os.mkdir('ResSynthData')
    np.savez('./ResSynthData/'+make_file_name(opt)+'_done.npz',\
        timeRes=timeRes, nbIter=nbIter, switchIter=switchIter, opt=opt, xFinal = xFinal,\
        nbFlops=nbFlops,sig=sig,RC=RC)
    # Light save - xFinal (solution vector) is not saved
    np.savez('./ResSynthData/'+make_file_name(opt)+'_light_done.npz',\
        timeRes=timeRes, nbIter=nbIter,switchIter=switchIter, opt=opt,\
        nbFlops=nbFlops,sig=sig,RC=RC)
    ret = { 'timeRes' : timeRes, \
            'nbIter': nbIter, \
            'nbFlops': nbFlops, \
            'opt': opt }
    return ret
    
    
def runLambdas_per_it(opt={},**keywords): #TODO merge with runLambdas
    """
    Run the Algorithm for a fix problem with all the value of lambda/lambda*
    in pen_param_list
    
    Return the results for opt['nbRuns'] problems
    """
    #### handles options
    default = default_expe()
                
    opt = mergeopt(opt, default, keywords)
#    for key, val in opt.items():
#        exec(key+'='+ repr(val))
        
    #pen_param_list = make_pen_param_list(opt['samp'],opt['min_reg'],opt['samp_type'])
    pen_param_list = np.logspace(-2,0,num=opt['samp']) # TODO modified for log spacing between 10^{-2} and 10^{0}
    #### run the algorithm if needed
    if  'ResSynthData' in os.listdir('./') and \
        (not opt['recalc']) and (make_file_name(opt)+'_done.npz') in os.listdir('./ResSynthData'):
        print 'experiment already computed'        
        Data = np.load('./ResSynthData/'+make_file_name(opt)+'_done.npz' )
        if 'time' in Data.keys():
            timeRes = Data['time'][()]
        else:
            timeRes = Data['timeRes'][()]
        if 'matNormTime' in Data['opt'][()].keys():
            opt['matNormTime'] = Data['opt'][()]['matNormTime']
        nbIter = Data['nbIter'][()]
        nbFlops = Data['nbFlops'][()]
        timePerIt = Data['timePerIt'][()]
#        if opt['disp_fig'] or not (make_file_name(opt)+'_Simu_relNoScreen.pdf') in\
#            os.listdir('./ResSynthData'):
#            traceLambdas( timeRes, nbIter, nbFlops ,opt )
        ret = { 'timeRes' : timeRes, \
                'nbIter': nbIter, \
                'nbFlops': nbFlops, \
                'timePerIt': timePerIt, \
                'opt': opt }
        return ret
 
    opt['lasso'] = 0.99           
    problem, opt = GP.generate(opt) 
    opt['K'] = problem.D.shape[1]
    
    avg = opt['nbRuns']
    nblambdas = len(pen_param_list)
    timeRes = dict(noScreen=np.zeros((nblambdas,avg), dtype = float),\
                static=np.zeros((nblambdas,avg), dtype = float),\
                dynamic=np.zeros((nblambdas,avg), dtype = float), \
                dynamic_approx=np.zeros((nblambdas,avg), dtype = float), \
                dynamic_approx2=np.zeros((nblambdas,avg), dtype = float), \
                dynamic_approx3=np.zeros((nblambdas,avg), dtype = float))
                
    nbIter = dict(noScreen=np.zeros((nblambdas,avg)),\
                static=np.zeros((nblambdas,avg)),\
                dynamic=np.zeros((nblambdas,avg)), \
                dynamic_approx=np.zeros((nblambdas,avg)), \
                dynamic_approx2=np.zeros((nblambdas,avg)), \
                dynamic_approx3=np.zeros((nblambdas,avg)))
                
    switchIter = dict(noScreen=np.zeros((nblambdas,avg)),\
                static=np.zeros((nblambdas,avg)),\
                dynamic=np.zeros((nblambdas,avg)), \
                dynamic_approx=np.zeros((nblambdas,avg)), \
                dynamic_approx2=np.zeros((nblambdas,avg)), \
                dynamic_approx3=np.zeros((nblambdas,avg)))
                
    nbFlops = dict(noScreen=np.zeros((nblambdas,avg), dtype = float),\
                static=np.zeros((nblambdas,avg), dtype = float),\
                dynamic=np.zeros((nblambdas,avg), dtype = float), \
                dynamic_approx=np.zeros((nblambdas,avg), dtype = float), \
                dynamic_approx2=np.zeros((nblambdas,avg), dtype = float), \
                dynamic_approx3=np.zeros((nblambdas,avg), dtype = float))                
                
    xFinal = dict(noScreen=np.zeros((nblambdas,avg,opt['K'])),\
                static=np.zeros((nblambdas,avg,opt['K'])),\
                dynamic=np.zeros((nblambdas,avg,opt['K'])),\
                dynamic_approx=np.zeros((nblambdas,avg,opt['K'])),\
                dynamic_approx2=np.zeros((nblambdas,avg,opt['K'])),\
                dynamic_approx3=np.zeros((nblambdas,avg,opt['K'])))
                
    #TODO replace 1000 by max_iter in opt    
    timePerIt = dict(noScreen=np.zeros((nblambdas,1000,avg), dtype = float),\
                static=np.zeros((nblambdas,1000,avg), dtype = float),\
                dynamic=np.zeros((nblambdas,1000,avg), dtype = float), \
                dynamic_approx=np.zeros((nblambdas,1000,avg), dtype = float), \
                dynamic_approx2=np.zeros((nblambdas,1000,avg), dtype = float), \
                dynamic_approx3=np.zeros((nblambdas,1000,avg), dtype = float))
                
    sig = np.zeros((opt['nbRuns'],opt['N']))
    
    # Evaluate RC (Relative Complexity)
    if opt['dict_type'] in {'sukro','low-rank'}:
#    if False:
        total_it = 1
        RC = estimateRC(problem.D,opt,total_it,verbose=True)
    else: # otherwise we just assume RC = 0.5
        RC = 0.5
    print "RC = %1.2f"%(RC)    

    for j in range(avg):
        start = None
        res = None
        problem, opt = GP.generate(opt, problem.D)

        star,lstar = problem.getStar() 
        sig[j,:] = problem.y.flatten()
        for i,lasso_ in enumerate(pen_param_list[::-1]):
            if not opt['wstart']:
                start = None
            elif res!=None:
                start = res['noScreen']
            opt['lasso'] = lasso_
            problem.pen_param = opt['lasso']*lstar
            timeIt, nbIt, switchIt, res, flops, junk, junk1, junk2, junk3, junk_ns = \
                run3versions(problem,RC,opt,start)
            for key in timeRes.iterkeys(): 
                timeRes[key][nblambdas-i-1,j] = timeIt[key]
                nbIter[key][nblambdas-i-1,j] = nbIt[key]
                switchIter[key][nblambdas-i-1,j] = switchIt[key]
                nbFlops[key][nblambdas-i-1,j] = flops[key]
                if not opt['dict_type'] =="MNIST"  :
                    xFinal[key][nblambdas-i-1,j] = res[key].flatten()
                    
            timePerIt['noScreen'][nblambdas-i-1,:,j] = np.lib.pad(junk_ns['time_per_it'], (0,1000-junk_ns['time_per_it'].size), 'constant')
            timePerIt['dynamic'][nblambdas-i-1,:,j] = np.lib.pad(junk['time_per_it'], (0,1000-junk['time_per_it'].size), 'constant')
            timePerIt['dynamic_approx'][nblambdas-i-1,:,j] = np.lib.pad(junk1['time_per_it'], (0,1000-junk1['time_per_it'].size), 'constant')
            timePerIt['dynamic_approx2'][nblambdas-i-1,:,j] = np.lib.pad(junk2['time_per_it'], (0,1000-junk2['time_per_it'].size), 'constant')
            timePerIt['dynamic_approx3'][nblambdas-i-1,:,j] = np.lib.pad(junk3['time_per_it'], (0,1000-junk3['time_per_it'].size), 'constant')
            
        print "problem %d over %d"%(j+1,avg)           
                 
    print('Done') 


    if 'ResSynthData' not in os.listdir('./'):
        os.mkdir('ResSynthData')
    np.savez('./ResSynthData/'+make_file_name(opt)+'_perIt_done.npz',\
        timeRes=timeRes, nbIter=nbIter, switchIter=switchIter, opt=opt, xFinal = xFinal,\
        nbFlops=nbFlops,sig=sig,RC=RC, timePerIt=timePerIt)
    
    ret = { 'timeRes' : timeRes, \
            'nbIter': nbIter, \
            'nbFlops': nbFlops, \
            'timePerIt': timePerIt, \
            'opt': opt }
    return ret

def traceLambdas_per_it(timePerIt, nbIter, opt): 
    """
    Colormap of the normalized time per iteration resulting from the experiment.
    Compute it if not done.
    """    
    
    mc = 0 # using only the first MC iteration, if more than one was performed
    # Calculate reference value
    time_it_noScreen = np.true_divide(timePerIt['noScreen'].sum(1),(timePerIt['noScreen']!=0).sum(1)).mean()
    # Calculate max number of iterations
    max_iter = int(nbIter['dynamic'].max())
    max_iter = int(round(max_iter,-2)) # round to nearest hundred
    # Normalizing and clipping data
    time_d = (timePerIt['dynamic'][:,2:max_iter,mc].T/time_it_noScreen).clip(0,1)
    time_d1 = (timePerIt['dynamic_approx'][:,:max_iter,mc].T/time_it_noScreen).clip(0,1)
    time_d2 = (timePerIt['dynamic_approx2'][:,:max_iter,mc].T/time_it_noScreen).clip(0,1)
    time_d3 = (timePerIt['dynamic_approx3'][:,:max_iter,mc].T/time_it_noScreen).clip(0,1)
    
    #pen_param_list = make_pen_param_list(opt['samp'],opt['min_reg'],opt['samp_type'])
    pen_param_list = np.logspace(-2,0,num=opt['samp']) # TODO modified for log spacing between 10^{-2} and 10^{0}
    # Defining display limits
    extent=[np.log10(pen_param_list[0]),np.log10(pen_param_list[-1]),max_iter,1]
    
    # Figure
    f , (GAP, AGAP) = plt.subplots(2,1,sharex=True, figsize=1.4*plt.figaspect(0.35))  
    
    # Colormap
    cmap = 'OrRd' #'hot_r', 'inferno_r', 'binary', 'Spectral'
    cmap_str = cmap
    # White to Red
    cmap = plt.cm.seismic(np.arange(plt.cm.seismic.N))
    cmap = cmap[len(cmap)/2:,0:3]
    #cmap[:,0:3] *= 0.6 # change luminosity
    cmap = matplotlib.colors.ListedColormap(cmap)
    cmap_str = 'wr'
    
    ######### PLOT DATA #########
    GAP.imshow(time_d,extent=extent, cmap=cmap, aspect = 'auto') #aspect=1./1000,
    im = AGAP.imshow(time_d2, extent=extent,cmap=cmap, aspect = 'auto') # aspect=1./1000,
    #GAP.contourf(pen_param_list,np.arange(max_iter,0,-1),time_d,200, cmap=cmap), plt.xscale('log') # N=200 color levels
    #im = AGAP.contourf(pen_param_list,np.arange(max_iter,0,-1),time_d2,200, cmap=cmap), plt.xscale('log') # N=200 color levels
    f.colorbar(im, ax=[GAP,AGAP])
    #############################
    
    # labels
    GAP.set_ylabel("Iteration") # ,fontsize = 24)
    AGAP.set_ylabel("Iteration") # ,fontsize = 24)
    # log10(lambda/lambda_max)
    plt.setp((GAP, AGAP), xticks=np.log10([0.01, 10**(-1.5), 0.1, 10**(-0.5), 1]), xticklabels=['-2','-1.5','-1','-0.5','0'])
    AGAP.set_xlabel(r"$\log_{10}(\lambda/\lambda_{\mathrm{max}})$")
    # lambda/lambda_max
    #plt.setp((GAP, AGAP), xticks=np.log10(np.concatenate((np.linspace(0.01,0.1,10), np.linspace(0.2,1,9)))), xticklabels=['0.01','','','','','','','','','0.1','','','','','','','','','1'])
    #AGAP.set_xlabel(r"$\lambda/\lambda_{\mathrm{max}}$")
    
    #GAP.set_yscale("log", nonposy='clip', basey=2)
    GAP.set_ylim([max_iter,1])
    matplotlib.rc('ytick', labelsize = 16)
    
    # Text
    GAP.text(-0.4, max_iter*0.85, r'GAP', size=24) 
    AGAP.text(-0.65, max_iter*0.85, r'A-GAP: $\|\mathbf{e}_j\| \!=\! 10^{-2}$', size=24) 
    
    f.suptitle(' Normalized running times per iteration',fontsize=26)
    
    f.savefig('./ResSynthData/PerIt_'+make_file_name(opt)+ '_Simu_relNoScreen_'+ cmap_str + '.pdf',bbox_inches = 'tight',bbox_pad = 2)
