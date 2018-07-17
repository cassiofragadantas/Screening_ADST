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
from dynascreen.solve import solver_multiple
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
    np.random.seed(0)
    
    lasso_list = [0.03] #[0.5, 0.75, 0.85]
    
    for lasso in lasso_list:

        # SuKro
#        default = dict(dict_type = 'sukro_approx',data_type = 'gnoise', lasso=lasso, N=2500,K=10000,scr_type = "GAP",\
#                    dict_params = dict(N1 = 50, N2 = 50, K1 = 100, K2 = 100,n_kron = [5, 10, 15, 20], svd_decay = 'exponential' ),\
#                    stop=dict(dgap_tol=1e-6, max_iter=1000), switching='default') #, wstart=True)
        # Test
        default = dict(dict_type = 'sukro_approx',data_type = 'bernoulli-gaussian', lasso=lasso, N=2500,K=10000,scr_type = "GAP",\
                    dict_params = dict(N1 = 50, N2 = 50, K1 = 100, K2 = 100,n_kron = [5,15], svd_decay = 'exponential' ),\
                    stop=dict(dgap_tol=1e-6, max_iter=1000), switching='default') #, wstart=True)
                        
        expe = mergeopt(default, opt, keywords)
        expeScrRate(opt=expe)
   
   
   
   
def second(opt=dict(), **keywords):
    '''
    Plot the normalized time and flops for a synthetic Lasso problem
    versus the penalization parameter \lambda/\lambda_*
    you can chosse the dictionary to be gaussian noise or pnoise
    '''        
    np.random.seed(0)
    # SuKro
    default = dict(dict_type = 'sukro_approx',data_type = 'bernoulli-gaussian', N=2500,K=10000,scr_type = "GAP",\
                    dict_params = dict(N1 = 50, N2 = 50, K1 = 100, K2 = 100,n_kron = range(5,20), svd_decay = 'exponential' ),nbRuns=1,\
                    stop=dict(dgap_tol=1e-6, max_iter=1000), switching='default',
                    samp=5, min_reg=0.01, samp_type='log') #, wstart=True)
    
    expe = mergeopt(opt, default, keywords)
    res = runLambdas(opt=expe)
    traceLambdas(res['timeRes'], res['nbIter'], res['nbFlops'] ,expe )    

def second_per_it(opt=dict(), **keywords):
    '''
    Plot the normalized time per iteration for a synthetic Lasso problem
    versus the penalization parameter \lambda/\lambda_*
    you can chosse the dictionary to be gaussian noise or pnoise
    '''        
    #np.random.seed(0)
    default = dict(dict_type = 'sukro',data_type = 'bernoulli-gaussian', N=2500,K=10000,scr_type = "GAP",\
                    dict_params = dict(N1 = 50, N2 = 50, K1 = 100, K2 = 100,n_kron = 20),nbRuns=1,\
                    stop=dict(dgap_tol=1e-6, max_iter=1000), switching='default', samp=100) #,  wstart=True)
    #default = dict(dict_type = 'low-rank',data_type = 'bernoulli-gaussian', N=2500,K=10000,scr_type = "ST1",\
    #                dict_params = dict(n_rank = 200),nbRuns=100,switching='default')
    expe = mergeopt(opt, default, keywords)
    res = runLambdas_per_it(opt=expe)
    #traceLambdas(res['timeRes'], res['nbIter'], res['nbFlops'] ,expe )
    traceLambdas_per_it(res['timePerIt'], res['nbIter'],expe ) 

def estimateRC(D,opt,total_it=1000, verbose=False, experimental=True):
    """
    Estimating practical Relative Complexity of D (the fast dictionary)
    """
    
    if experimental:
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
            
        RC = RC_dot
    else: #only theoretical RC
        print ">>>> Using theoretical RC only! <<<<"
        dense_cost = 2*opt['K']*opt['N']
        # Cost of the structure dictionary
        if D.opType is 'sukro':
            struct_cost = 2*D.nkron*D.K1*D.N2*(D.K2 + D.N1)
        elif D.opType is 'low-rank':
            struct_cost = 2*D.nrank*(opt['K'] + opt['N'])
        else:
            print "Theoretical RC not implemented for this type of structure, using RC = 0.5"
            RC = 0.5
            
        RC = float(struct_cost)/dense_cost
        print "RC = %1.3f"%(RC)
    
    return RC

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
    
    ########################################################################
    #### Evaluate RC (Relative Complexity) and Approximation errors 
    ########################################################################
    normE_all = []
    norm2E_all = []
    RC = []
    if opt['dict_type'] is 'sukro_approx':
        D_sukro = np.zeros_like(problem.D_bis.data)            
        for nkron in range(1,max(opt['dict_params']['n_kron'])+1): # Reconstruct the dictionary until nkron to calculate approximation error
            D_sukro = D_sukro + np.kron(problem.D_bis.A[:,:,nkron-1],problem.D_bis.B[:,:,nkron-1])
            if nkron in opt['dict_params']['n_kron']:
                D_sukro_norm = D_sukro/np.tile(problem.D_bis.normcoef,(D_sukro.shape[0],1)) # normalizing
                # Calculate errors
                E = D_sukro_norm - problem.D.data
                norm2E_all.append(np.linalg.norm(E,2))
                normE_all.append(np.linalg.norm(E,axis=0))
            
                # Estimate RC
                problem.D_bis.nkron = nkron
                RC.append(estimateRC(problem.D_bis,opt,total_it=1000,verbose=True,experimental=False))
    else:
        raise NotImplementedError('\n Implemented only for SuKro dictionaries')

    ########################################################################
    #### Run simulations
    ########################################################################
        
    timeRes, nbIteration, switchIt, sols, flops, dynamicRun, \
    dynamicRun_approx, noScreenRun  = \
        run3versions(problem, RC, normE_all, norm2E_all, opt, warm_start = None)   
        
    ########################################################################
    #### Print results
    ########################################################################    
    if problem.__class__.__name__ == 'GroupLasso':
        #print '\n time to compute group norms: %.3f ms'%(opt['matNormTime']*1000)
        print '\n GroupLasso not imple;ented'

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

    print "\n__DYNAMIC APPROX MULTIPLE__"
    print "wrt no screen: time gain : %.3f, theoretical gain %.3f"\
        %(timeRes['noScreen']/timeRes['dynamic_approx'],\
        float(flops['noScreen'])/flops['dynamic_approx'])
    print "wrt static: time gain : %.3f, theoretical gain %.3f"\
        %(timeRes['static']/timeRes['dynamic_approx'],
        float(flops['static'])/flops['dynamic_approx'])
            
    ########################################################################
    #### Saving data and plotting result  
    ########################################################################
            
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
        dGaps_est_approx = dynamicRun_approx['dGaps_est']) #TESTE dgap_est


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
    markers_on1 = [x-1 for x in dynamicRun_approx['switch_it']]
    
    flops_ns = flop_calc_it("noScreen",K,N,[], noScreenRun['zeros'],[]) #float(flops["noScreen"])/nbIteration["noScreen"]
    flops_d = flop_calc_it("dynamic",K,N,dynamicRun['screenrate'], dynamicRun['zeros'],[])
    flops_d1 = flop_calc_it_multiple("dynamic",K,N,dynamicRun_approx['screenrate'], dynamicRun_approx['zeros'],[],RC,dynamicRun_approx['switch_it'])
           
    axScreen.plot(np.arange(length),(1 - dynamicRun['screenrate'])*K, '-k', linewidth = 6) #, 'x', markevery=[length-1]
    axScreen.plot(np.arange(length_approx),(1 - dynamicRun_approx['screenrate'])*K, '-mx', markevery=markers_on1) # Marker on the swithing point 

    axScreen.axis([0,max(length,length_approx), 0, K*1.1])
    axScreen.grid(True) 
    axScreen.set_ylabel("Size of the dictionary")
    axScreen.set_xlabel("Iteration t")

    #axFlops_it.plot(np.arange(length),flops_ns,'-b')
    axFlops_it.plot(np.arange(length),flops_d, '-k', linewidth = 6) #markevery=[length-1])
    axFlops_it.plot(np.arange(length_approx),flops_d1, '-mx', markevery=markers_on1)
    axFlops_it.grid(True)         
    #axFlops_it.set_ylim((-0.19,1.15))
    axFlops_it.set_ylim((0,1.1*max(max(flops_d),max(flops_d1))))
    axFlops_it.set_ylabel("Flops number") #,fontsize = 24)
    axFlops_it.set_xlabel("Iteration t")
    #axFlops_it.legend(fontsize=22,loc=3,frameon=False)

    f.savefig('./ResSynthData/Simu_screenRate_'+make_file_name(opt)+\
        '_lasso'+str(opt['lasso'])+'approx.pdf',bbox_inches = 'tight',bbox_pad = 2 )
    
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

def flop_calc_it_multiple(EmbedTest,K,N,screenrate,zeros,Gr,RC=1,switch_it=0): # Returns the iteration complexity
    nb_gr = len(Gr)
    if EmbedTest == 'dynamic':
        if RC == 1:
            flops = ( N*K*(2 - np.asarray(screenrate) -
                np.asarray(zeros)/K) + \
                    6*(1-np.asarray(screenrate)) + 5*N + 5*nb_gr)                   
                    #6*K + 5*N + 5*nb_gr)
        else:                    
            # Iterations with approximate dictionaries
            flops = []
            switch_it = [0] + switch_it
            for k in range(len(RC)):
                #N*K*(RC[k] + 1 - #TODO verify which is right 
                flops = np.append( flops, ( N*K*RC[k]*(1 + 1 - 
                                  np.asarray(zeros[switch_it[k]:switch_it[k+1]])/K) + \
                                  7*(1-np.asarray(screenrate[switch_it[k]:switch_it[k+1]])) + 5*N + 5*nb_gr) )
            
            # Final iterations: original dictionary            
            flops_bis = ( N*K*(2 - np.asarray(screenrate[switch_it[-1]:]) -
                        np.asarray(zeros[switch_it[-1]:])/K) + \
                        6*(1-np.asarray(screenrate[switch_it[-1]:])) + 5*N + 5*nb_gr)
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
    
    
    flop_q0_d, flop_q1_d, flop_q2_d, flop_q3_d, flop_q4_d =\
            np.percentile(nbFlops['dynamic'].astype(float)
            /nbFlops['noScreen'],[10,25,50,75,90],axis=1)
    flop_q0_s, flop_q1_s, flop_q2_s, flop_q3_s, flop_q4_s =\
            np.percentile(nbFlops['static'].astype(float)
            /nbFlops['noScreen'],[10,25,50,75,90],axis=1)
    flop_q0_d1, flop_q1_d1, flop_q2_d1, flop_q3_d1, flop_q4_d1 =\
            np.percentile(nbFlops['dynamic_approx'].astype(float)
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
        # Dynamic approx multiple
        plt.plot(pen_param_list, q2_d1,'m^-',
                 label = 'A-D'+Gstr+opt['scr_type']+r'$: \| \mathbf{e}_j \|\!=\!10^{-1}$', 
                 markevery= mkevry, markersize = markersize)  
        plt.fill_between(pen_param_list, q1_d1, q3_d1,alpha = 0.2,
                         facecolor = 'm')   

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

    # Dynamic      
    f = plt.figure(figsize=1.05*plt.figaspect(0.6))                   
    plt.plot(pen_param_list,flop_q2_d,'ks-',
                 label = 'D'+Gstr+opt['scr_type'],                 
                 markevery= mkevry, markersize = markersize)  
    plt.fill_between(pen_param_list, flop_q1_d, flop_q3_d,alpha = 0.2,
                         facecolor = 'k')

    # Dynamic approx multiple
    plt.plot(pen_param_list,flop_q2_d1,'m^-',
                 label = 'A-D'+Gstr+opt['scr_type']+' '+str(opt['dict_params']['n_kron']), 
                 markevery= mkevry, markersize = markersize)  
    plt.fill_between(pen_param_list, flop_q1_d1, flop_q3_d1,alpha = 0.2,
                         facecolor = 'm')

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
        


def run3versions(problem=None, RC=1, normE_all=[], norm2E_all=[], opt={}, warm_start = None, **keywords):
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
    
    #### run
                                
    # Multiple dictionaries (only for sukro)
                                
    if opt['dict_type'] == 'sukro_approx':
        # At this point: D = slow (=fast+E), D_bis = fast
        problem.D, problem.D_bis = problem.D_bis, problem.D # start with the approximate (fast) dictionary
                 
        res_dyn_approx = \
            solver_multiple(problem=problem, normE_all=normE_all, norm2E_all = norm2E_all, RC=RC, L=opt['L'], stop=opt['stop'], switching=opt['switching'], \
                                scr_type=opt['scr_type'], \
                                dict_specs = opt['dict_params']['n_kron'], \
                                EmbedTest='dynamic', algo_type=opt['algo_type'], \
                                warm_start = warm_start)
    else:
        raise NotImplementedError('\n Implemented only for SuKro dictionaries')


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

#    if opt['dict_type'] and problem.D_bis.opType in {'sukro','low-rank'}: # Put 'sukro' back to D
##    if opt['dict_type'] in {'sukro','low-rank'}: # Put 'sukro' back to D
#        problem.D, problem.D_bis = problem.D_bis, problem.D


    timeRes     = { 'noScreen': res_noScreen['time'],
                    'static':   res_static['time'],
                    'dynamic':  res_dyn['time'],
                    'dynamic_approx':  res_dyn_approx['time']}
                
    nbIteration = { 'noScreen': res_noScreen['nbIter'],
                    'static':   res_static['nbIter'],
                    'dynamic':  res_dyn['nbIter'],
                    'dynamic_approx':  res_dyn_approx['nbIter']}
                    
    switchIt    = { 'noScreen': 0,
                    'static':   0,
                    'dynamic':  0,
                    'dynamic_approx':  res_dyn_approx['switch_it']}
                    
    flops       = { 'noScreen': res_noScreen['flops'],
                    'static':   res_static['flops'],
                    'dynamic':  res_dyn['flops'],
                    'dynamic_approx':  res_dyn_approx['flops']}            
    
    Solution    = { 'noScreen': res_noScreen['sol'],
                    'static':   res_static['sol'],
                    'dynamic':  res_dyn['sol'],
                    'dynamic_approx':  res_dyn_approx['sol']}
                    
    return timeRes,nbIteration,switchIt,Solution,flops,res_dyn, res_dyn_approx, res_noScreen

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
                dynamic_approx=np.zeros((nblambdas,avg), dtype = float))
                
    nbIter = dict(noScreen=np.zeros((nblambdas,avg)),\
                static=np.zeros((nblambdas,avg)),\
                dynamic=np.zeros((nblambdas,avg)), \
                dynamic_approx=np.zeros((nblambdas,avg)))
                
    switchIter = dict(noScreen=np.zeros((nblambdas,avg,len(opt['dict_params']['n_kron']))),\
                static=np.zeros((nblambdas,avg,len(opt['dict_params']['n_kron']))),\
                dynamic=np.zeros((nblambdas,avg,len(opt['dict_params']['n_kron']))), \
                dynamic_approx=np.zeros((nblambdas,avg,len(opt['dict_params']['n_kron']))))
                
    nbFlops = dict(noScreen=np.zeros((nblambdas,avg), dtype = float),\
                static=np.zeros((nblambdas,avg), dtype = float),\
                dynamic=np.zeros((nblambdas,avg), dtype = float), \
                dynamic_approx=np.zeros((nblambdas,avg), dtype = float))
                
    xFinal = dict(noScreen=np.zeros((nblambdas,avg,opt['K'])),\
                static=np.zeros((nblambdas,avg,opt['K'])),\
                dynamic=np.zeros((nblambdas,avg,opt['K'])),\
                dynamic_approx=np.zeros((nblambdas,avg,opt['K'])))
                
    sig = np.zeros((opt['nbRuns'],opt['N']))
    
    # Evaluate RC (Relative Complexity) and Approximation errors
    normE_all = []
    norm2E_all = []
    RC = []
    if opt['dict_type'] is 'sukro_approx':
        D_sukro = np.zeros_like(problem.D_bis.data)            
        for nkron in range(1,max(opt['dict_params']['n_kron'])+1): # Reconstruct the dictionary until nkron to calculate approximation error
            D_sukro = D_sukro + np.kron(problem.D_bis.A[:,:,nkron-1],problem.D_bis.B[:,:,nkron-1])
            if nkron in opt['dict_params']['n_kron']:
                D_sukro_norm = D_sukro/np.tile(problem.D_bis.normcoef,(D_sukro.shape[0],1)) # normalizing
                # Calculate errors
                E = D_sukro_norm - problem.D.data
                norm2E_all.append(np.linalg.norm(E,2))
                normE_all.append(np.linalg.norm(E,axis=0))
            
                # Estimate RC
                problem.D_bis.nkron = nkron
                RC.append(estimateRC(problem.D_bis,opt,total_it=1000,verbose=True,experimental=False))
    else:
        raise NotImplementedError('\n Implemented only for SuKro dictionaries')

    for j in range(avg):
        start = None
        res = None
        problem, opt = GP.generate(opt, D = problem.D, D_bis = problem.D_bis)

        star,lstar = problem.getStar() 
        sig[j,:] = problem.y.flatten()
        for i,lasso_ in enumerate(pen_param_list[::-1]):
            if not opt['wstart']:
                start = None
            elif res!=None:
                start = res['noScreen']
            opt['lasso'] = lasso_
            problem.pen_param = opt['lasso']*lstar
            timeIt, nbIt, switchIt, res, flops, junk, junk1, junk_ns = \
                run3versions(problem,RC,normE_all,norm2E_all,opt,start)
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
