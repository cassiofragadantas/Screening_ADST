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
from dynascreen.solve import solver_approx_parallel
from dynascreen.solve import solver_multiple
from . import generate_problem as GP
from .misc import mergeopt, make_file_name, type2name
from .misc import testopt, default_expe, make_pen_param_list

from dynascreen.dictionary import Dict

from FaustPy import Faust #For version faust-2.2rc10 or earlier. Use directly Faust
#import pyfaust # For version faust-2.3rc2 or later. Also must use pyfaust.Faust

# Real data experiments
from mnist import MNIST
import scipy.io as sio

# For multicolored line in legend 
# https://stackoverflow.com/questions/49223702/adding-a-legend-to-a-matplotlib-plot-with-a-multicolored-line
from matplotlib.legend_handler import HandlerLineCollection
from matplotlib.collections import LineCollection

class HandlerColorLineCollection(HandlerLineCollection):
    def create_artists(self, legend, artist ,xdescent, ydescent,
                        width, height, fontsize,trans):
        x = np.linspace(0,width,self.get_numpoints(legend)+1)
        y = np.zeros(self.get_numpoints(legend)+1)+height/2.-ydescent
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=artist.cmap,
                     transform=trans)
        lc.set_array(x)
        lc.set_linewidth(artist.get_linewidth())
        return [lc]

      
def first(opt =dict(), **keywords):
    '''
    Plot one execution of the dynamic screening for given options
     
    to modify default option just do first(algo_type = 'FISTA')...
    '''   
    np.random.seed(0)
    
    lasso_list = [0.5] #[0.5, 0.75, 0.85]
    
    for lasso in lasso_list:

        # SuKro
#        default = dict(dict_type = 'sukro_approx',data_type = 'gnoise', lasso=lasso, N=2500,K=10000,scr_type = "GAP",\
#                    dict_params = dict(N1 = 50, N2 = 50, K1 = 100, K2 = 100,n_kron = [5, 10, 15, 20], svd_decay = 'exponential' ),\
#                    stop=dict(dgap_tol=1e-6, max_iter=1000), switching='default') #, wstart=True)
        # MEG
#        default =  dict(dict_type = 'MEG', data_type = 'bernoulli-gaussian',  lasso=lasso,N=204, K=8193,\
#                        data_params = dict(p = 0.001),
#                        stop=dict(dgap_tol=1e-4, max_iter=1000000), scr_type = "GAP", switching='default', algo_type = 'ISTA')
        # Test
        default = dict(dict_type = 'sukro_approx',data_type = 'bernoulli-gaussian', lasso=lasso, N=900,K=10000,scr_type = "GAP",\
                    data_params = dict(p = 0.1),
                    dict_params = dict(N1 = 30, N2 = 30, K1 = 100, K2 = 100,n_kron = [15], svd_decay = 'exponential' ),\
                    stop=dict(dgap_tol=1e-6, max_iter=1000), switching='screening_only', algo_type = 'ISTA') #, wstart=True)
                        
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
#    default = dict(dict_type = 'sukro_approx',data_type = 'bernoulli-gaussian', N=2500,K=10000,scr_type = "GAP",\
#                    dict_params = dict(N1 = 50, N2 = 50, K1 = 100, K2 = 100,n_kron = range(5,20), svd_decay = 'exponential' ),nbRuns=1,\
#                    stop=dict(dgap_tol=1e-6, max_iter=1000), switching='default',
#                    samp=5, min_reg=0.01, samp_type='log') #, wstart=True)
    # MEG
    default =  dict(dict_type = 'MEG', data_type = 'bernoulli-gaussian',N=204, K=8193,
                    data_params = dict(p = 0.001), # p = 0.001 = 8 active sources
                    stop=dict(dgap_tol=1e-5, max_iter=1000000), scr_type = "GAP", switching='default',nbRuns=1, #50
                    samp=5, min_reg=0.01, samp_type='log', algo_type = 'FISTA',switching_gamma=0.2)# , wstart=True)
    # Teste
#    default = dict(dict_type = 'sukro_approx',data_type = 'bernoulli-gaussian', N=2500,K=10000,scr_type = "GAP",\
#                    dict_params = dict(N1 = 50, N2 = 50, K1 = 100, K2 = 100,n_kron = [5, 10, 15, 20], svd_decay = 'exponential' ),nbRuns=1,\
#                    stop=dict(dgap_tol=1e-6, max_iter=1000), switching='default',
#                    samp=5, min_reg=0.01, samp_type='log', wstart=True) #, wstart=True)
    
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
    

def complete(opt=dict(), **keywords):
    '''
    Plot the normalized time and flops for a synthetic Lasso problem
    versus the penalization parameter \lambda/\lambda_*
    you can chosse the dictionary to be sukro_approx noise or MEG
    
    /!\ In this particular experiment:
        - n_kron field dict_params must be a list of list, (i.e. in the form [[],[],[]]
        - scr_type, stop, switching_gamma and algo_type fields must be lists
    '''        
    np.random.seed(0)
    # SuKro - protocol
    default = dict(dict_type = 'sukro_approx',data_type = 'bernoulli-gaussian', N=2500,K=10000,scr_type = ['ST1','GAP'],\
                    dict_params = dict(N1 = 50, N2 = 50, K1 = 100, K2 = 100,n_kron = [[5], [10], [15], [20], [5, 10, 15, 20]], svd_decay = 'exponential', reuse = True),nbRuns=1,\
                    stop=[dict(dgap_tol=1e-4, max_iter=100000), dict(dgap_tol=1e-5, max_iter=100000), dict(dgap_tol=1e-6, max_iter=100000)], 
                    switching='default', switching_gamma=[0.2, 0.5, 0.8],
                    svd_decay_const_list = [0.1, 0.3, 0.5],
                    samp=20, min_reg=0.01, samp_type='log',  algo_type = ['FISTA','ISTA']) #, wstart=True)
    # Choosing Gamma
#    default = dict(dict_type = 'sukro_approx',data_type = 'bernoulli-gaussian', N=2500,K=10000,scr_type = ['GAP'],\
#                    dict_params = dict(N1 = 50, N2 = 50, K1 = 100, K2 = 100,n_kron = [[5, 10, 15, 20]], svd_decay = 'exponential', reuse = True),nbRuns=1,\
#                    stop=[dict(dgap_tol=1e-4, max_iter=100000)], #, dict(dgap_tol=1e-5, max_iter=100000), dict(dgap_tol=1e-6, max_iter=100000)], 
#                    switching='default', switching_gamma=list(np.round(np.logspace(-2,np.log10(0.8),20),3)), #[0.5],
#                    svd_decay_const_list = [0.1, 0.3, 0.5],
#                    samp=20, min_reg=0.01, samp_type='log',  algo_type = ['ISTA']) #, wstart=True)

    expe = mergeopt(opt, default, keywords)
        
    if not isinstance(opt['algo_type'],list): opt['algo_type'] = [opt['algo_type']]
    if not isinstance(opt['scr_type'],list): opt['scr_type'] = [opt['scr_type']]
    if not isinstance(opt['svd_decay_const_list'],list): opt['svd_decay_const_list'] = [float(opt['svd_decay_const_list'])]

    # Run simulations
    if default['dict_type'] == 'sukro_approx':
        for decay in opt['svd_decay_const_list']:
            expe['dict_params']['svd_decay_const'] = decay
            runProtocol(opt=expe)
    else:
        print('Watch out! Not sure this will work.')
        runProtocol(opt=expe)    
    
def scatterplot_screening_rates(opt=dict(), **keywords):
    '''
    Scatterplot comparing screening rates obtained with original dictionary and
    its different approximations. Also scatterplots the heuristic estimation
    of the screening rate used in the switching criterion.
    Results show that the heuristic rate estimation correlates very well with
    actual screening rate
    '''

    #TODO Legend and axes titles on the plots

    np.random.seed(0)
    #np.random.seed(10) #used for figures with y=X\beta           

    default =  dict(dict_type = 'gnoise', data_type = 'bernoulli-gaussian', N=2500, K=10000, #lasso=lasso
                    data_params=dict(p=0.005),
                    stop=dict(dgap_tol=1e-5, max_iter=1000), scr_type = "ST1", switching='off',\
                     algo_type = 'ISTA', nbRuns = 100)

    opt = mergeopt(default, opt, keywords)
    default = default_expe()
    opt = mergeopt(opt, default, keywords)
    testopt(opt)
    
    # Parameters    
#    nbReps = 50
    normE_vec = np.array([1e-1,1e-2,1e-3])    
    RC = 0.5 # doesn't matter here  
    lasso_list = [0.5, 0.7, 0.8] #[0.5, 0.75, 0.85]
    
    # config plot
    matplotlib.rc('axes', labelsize = 24)
    matplotlib.rc('xtick', labelsize = 24)
    matplotlib.rc('ytick', labelsize = 24)
    matplotlib.rc('axes', titlesize = 24)
    matplotlib.rc('lines', linewidth = 3) 
    matplotlib.rc('mathtext', fontset='cm') 
#    matplotlib.rc('text', usetex=True)
#    plt.rcParams['font.family'] = 'serif'
#    plt.rcParams['font.serif'] = ['Times']
#    matplotlib.rc('font',**{'family':'serif','serif':['Times']})

    color = ['m','g','r']

    
    problem, opt = GP.generate(opt) # Generate the problem
    E = np.sqrt(1./opt['N'])*np.random.randn(opt['N'],opt['K'])
    norm2E = np.linalg.norm(E,2)  


    for lasso in lasso_list:
        # Figures
        f2 , rate_all = \
            plt.subplots(1,2,sharex=True,figsize=1.4*plt.figaspect(0.37))
        filename = 'scatterplot_'+make_file_name(opt)+'_lasso'+str(lasso)
            
        res_dyn_screenrate_list = []
        res_dyn_approx_screenrate_list = []
        res_dyn_approx_screenrate_est_list = []
        Data = None
        
        ######### Verify if the results are already available. ##########
        
        # Doesn't run the simulations in this case.
        if 'ResSynthData' in os.listdir('./') and \
           (not opt['recalc']) and (filename+'.npz') in os.listdir('./ResSynthData'):
               
            print 'experiment already computed'        
            Data = np.load('./ResSynthData/'+filename+'.npz' )
            res_dyn_screenrate_list = Data['res_dyn_screenrate_list'][()]
            res_dyn_approx_screenrate_list = Data['res_dyn_approx_screenrate_list'][()]
            res_dyn_approx_screenrate_est_list = Data['res_dyn_approx_screenrate_est_list'][()]
            for k in range(len(normE_vec)):
                rate_all[0].scatter(1-res_dyn_screenrate_list[k],1-res_dyn_approx_screenrate_list[k],c=color[k])
                rate_all[1].scatter(1-res_dyn_approx_screenrate_est_list[k],1-res_dyn_screenrate_list[k],c=color[k])
            
            rate_all[0].set_xlabel(r'Oracle $|\mathcal{A}_t|/K$',fontsize = 24)
            rate_all[0].set_ylabel(r'$|\mathcal{A}_t|/K$  using $\tilde{\mathbf{A}}$',fontsize = 24)
#            rate_all[0].grid(True)         
            rate_all[0].set_ylim((-0.05,1.05))
            rate_all[0].set_xlim((-0.05,1.05))
            rate_all[0].legend([r'$\epsilon_j=10^{-1}$ $\forall j$', r'$\epsilon_j=10^{-2}$ $\forall j$', r'$\epsilon_j=10^{-3}$ $\forall j$'],fontsize=24,loc=4,frameon=False)
            # plot bisectrix            
            rate_all[0].plot([-1, 2], [-1, 2],':',lw=1, color="0.3")

            rate_all[1].set_xlabel(r'Oracle $|\mathcal{A}_t|/K$',fontsize = 24)
            rate_all[1].set_ylabel(r'$K_t/K$',fontsize = 24) # |\mathcal{A}^\prime|
#            rate_all[1].grid(True)          
            rate_all[1].set_ylim((-0.05,1.05))
            rate_all[1].set_xlim((-0.05,1.05))
            rate_all[1].plot([-1, 2], [-1, 2],':',lw=1, color="0.3")
    
            f2.savefig('./ResSynthData/'+filename+'_together.pdf',bbox_inches = 'tight' )
            f2.savefig('./ResSynthData/'+filename+'_together.eps',bbox_inches = 'tight' )
            
        #################################################################
        else:            
            # Just to estimate number of iterations            
            res_dyn = solver(problem=problem, L=opt['L'], stop=opt['stop'],\
                                scr_type=opt['scr_type'], \
                                EmbedTest='dynamic', algo_type=opt['algo_type'])
            stop = dict(max_iter=res_dyn['nbIter'])                            
                
            for (normE,k) in zip(normE_vec,range(len(normE_vec))):
                print "normE = %.2E"%(normE)
                # approximate dictionary
                problem.D_bis =  Dict(problem.D.data + normE*E)
                problem.D_bis.normalize()       
    
                res_dyn_screenrate = []
                res_dyn_approx_screenrate = []
                res_dyn_approx_screenrate_est = [] 
                for k_iter in range(opt['nbRuns']):
                    # Generate the problem                
    #                problem, opt = GP.generate(opt)                  
    #                problem.D_bis =  Dict(problem.D.data + normE*E)
    #                problem.D_bis.normalize()   
                    # Generate the problem  
                    problem, opt = GP.generate(opt,D=problem.D,D_bis=problem.D_bis)
    
                    #######################################################################
                    #### Run simulations
                    #######################################################################                                   
    #                res_dyn = solver(problem=problem, L=opt['L'], stop=opt['stop'],\
    #                                    scr_type=opt['scr_type'], \
    #                                    EmbedTest='dynamic', algo_type=opt['algo_type'])
    #                                    
    #                problem.D, problem.D_bis = problem.D_bis, problem.D # start with the approximate (fast) dictionary
    #                stop = dict(max_iter=res_dyn['nbIter']) # running same number of iterations as previous solver
    #                res_dyn_approx = \
    #                    solver_approx(problem=problem, normE=normE, norm2E = normE*norm2E, RC=RC, L=opt['L'], stop=stop, switching=opt['switching'], \
    #                                        scr_type=opt['scr_type'], \
    #                                        EmbedTest='dynamic', algo_type=opt['algo_type'])
    #
    #                res_dyn_screenrate = np.append(res_dyn_screenrate, res_dyn['screenrate'])
    #                res_dyn_approx_screenrate = np.append(res_dyn_approx_screenrate,res_dyn_approx['screenrate'][:-1])
    #                res_dyn_approx_screenrate_est = np.append(res_dyn_approx_screenrate_est,res_dyn_approx['screenrate_est'][:-1])
     
                    # Runs solver with approximate dictionary and in parallel calculates the screening rate obtained with the original dictionary
                    problem.D, problem.D_bis = problem.D_bis, problem.D # start with the approximate (fast) dictionary
                    res_dyn_parallel = \
                        solver_approx_parallel(problem=problem, normE=normE, norm2E = normE*norm2E, RC=RC, L=opt['L'], stop=stop, switching=opt['switching'], \
                                            scr_type=opt['scr_type'], \
                                            EmbedTest='dynamic', algo_type=opt['algo_type'])
    
                    res_dyn_screenrate = np.append(res_dyn_screenrate, res_dyn_parallel['screenrate_conv'][:-1])
                    res_dyn_approx_screenrate = np.append(res_dyn_approx_screenrate,res_dyn_parallel['screenrate'][:-1])
                    res_dyn_approx_screenrate_est = np.append(res_dyn_approx_screenrate_est,res_dyn_parallel['screenrate_est'][:-1])
                
    
                # Store results
                res_dyn_screenrate_list.append(res_dyn_screenrate)
                res_dyn_approx_screenrate_list.append(res_dyn_approx_screenrate)
                res_dyn_approx_screenrate_est_list.append(res_dyn_approx_screenrate_est)
                
                # Plot results
               
                rate_all[0].scatter(1-res_dyn_screenrate,1-res_dyn_approx_screenrate,c=color[k])
                rate_all[1].scatter(1-res_dyn_approx_screenrate_est,1-res_dyn_screenrate,c=color[k])
    
            rate_all[0].set_xlabel(r'Oracle $|\mathcal{A}_t|/K$',fontsize = 24)
            rate_all[0].set_ylabel(r'$|\mathcal{A}_t|/K$  using $\tilde{\mathbf{A}}$',fontsize = 24)
            rate_all[0].grid(True)         
            rate_all[0].set_ylim((-0.05,1.05))
            rate_all[0].set_xlim((-0.05,1.05))
            rate_all[0].legend([r'$\epsilon_j=10^{-1}$ $\forall j$', r'$\epsilon_j=10^{-2}$ $\forall j$', r'$\epsilon_j=10^{-3}$ $\forall j$'],fontsize=24,loc=4,frameon=False)

            rate_all[1].set_xlabel(r'Oracle $|\mathcal{A}_t|/K$',fontsize = 24)
            rate_all[1].set_ylabel(r'$K_t/K$',fontsize = 24) #|\mathcal{A}^\prime|
            rate_all[1].grid(True)          
            rate_all[1].set_ylim((-0.05,1.05))
            rate_all[1].set_xlim((-0.05,1.05))
    
            # Save figures and results    
            if 'ResSynthData' not in os.listdir('./'):
                os.mkdir('ResSynthData')
    
            f2.savefig('./ResSynthData/'+filename+'_together.pdf',bbox_inches = 'tight' )
            f2.savefig('./ResSynthData/'+filename+'_together.eps',bbox_inches = 'tight' )
            np.savez('./ResSynthData/'+filename+'.npz',\
                res_dyn_screenrate_list=res_dyn_screenrate_list,
                res_dyn_approx_screenrate_list=res_dyn_approx_screenrate_list,
                res_dyn_approx_screenrate_est_list=res_dyn_approx_screenrate_est_list)
        
def gap_evolution_it_time(opt =dict(), **keywords):
    '''
    Plot one execution of the dynamic screening for given options. Also plots
    duality gap per iteration and as a function of time.
     
    to modify default option just do first(algo_type = 'FISTA')...
    '''   
    np.random.seed(0)
    
    lasso_list = [0.2] #[0.5, 0.75, 0.85]
    
    for lasso in lasso_list:

        # SuKro approx
        default = dict(dict_type = 'sukro_approx',data_type = 'bernoulli-gaussian', N=2500,K=10000,scr_type = 'GAP', lasso=lasso,\
                    dict_params = dict(N1 = 50, N2 = 50, K1 = 100, K2 = 100,n_kron = [5, 10, 15, 20], svd_decay = 'exponential', svd_decay_const = 0.5,reuse = True),\
                    stop=dict(dgap_tol=1e-4, max_iter=100000),
                    switching='default', switching_gamma=0.2, algo_type = 'ISTA') #, wstart=True)
                        
        opt = mergeopt(default, opt, keywords)
        default = default_expe()
        opt = mergeopt(opt, default, keywords)
        
        filename = './ResSynthData/'+make_file_name(opt)+'_lambda_'+str(opt['lasso'])+'.npz'        
        # Run only if necessary
        if os.path.isfile(filename):
            Data = np.load(filename)
            res_dyn = Data['res_dyn'][()] 
            res_dyn_approx = Data['res_dyn_approx'][()]
            RC = Data['RC'].tolist()
        else:
            res_dyn, res_dyn_approx, no_screen, RC = expeScrRate(opt=opt)
            del res_dyn['sol']; del res_dyn_approx['sol']
            del res_dyn['problem']; del res_dyn_approx['problem']            
            np.savez('./ResSynthData/'+make_file_name(opt)+'_lambda_'+str(opt['lasso'])+'.npz',\
                     res_dyn=res_dyn,res_dyn_approx=res_dyn_approx, no_screen=no_screen, opt=opt,RC=RC)
        # Plot results
        traceGaps(res_dyn,res_dyn_approx,opt,RC)
   
def gap_evolution_it_time_tol(opt =dict(), **keywords):
    '''
    Same as gap_evolution_it_time, but used to generate the time colormap 
    as a function of the convergence tolerande and the regularization.
    
    Plot one execution of the dynamic screening for given options. Also plots
    duality gap per iteration and as a function of time.
     
    to modify default option just do first(algo_type = 'FISTA')...
    '''   
    for seed in range(5):
#        seed = 0
        np.random.seed(seed)
        
        lasso_list = [0.2] #[0.5, 0.75, 0.85]
        samp=20; min_reg=0.01; samp_type='log'
        lasso_list = make_pen_param_list(samp,min_reg,samp_type)  
        
        for lasso in lasso_list:
    
            # SuKro approx
            default = dict(dict_type = 'sukro_approx',data_type = 'bernoulli-gaussian', N=2500,K=10000,scr_type = 'GAP', lasso=lasso,\
                        dict_params = dict(N1 = 50, N2 = 50, K1 = 100, K2 = 100,n_kron = [5, 10, 15, 20], svd_decay = 'exponential', svd_decay_const = 0.3,reuse = True),\
                        stop=dict(dgap_tol=1e-6, max_iter=100000),
                        switching='default', switching_gamma=0.5, algo_type = 'FISTA') #, wstart=True)
                            
            opt = mergeopt(default, opt, keywords)
            default = default_expe()
            opt = mergeopt(opt, default, keywords)
            
            filename = './ResSynthData/'+make_file_name(opt)+'_gamma'+str(opt['switching_gamma'])+'_lambda_'+str(opt['lasso'])+'_seed'+str(seed)+'.npz'
            # Run only if necessary
            if os.path.isfile(filename):
                Data = np.load(filename)
                res_dyn = Data['res_dyn'][()] 
                res_dyn_approx = Data['res_dyn_approx'][()]
                RC = Data['RC'].tolist()
            else:
                res_dyn, res_dyn_approx, no_screen, RC = expeScrRate(opt=opt)
                del res_dyn['sol']; del res_dyn_approx['sol']; del no_screen['sol']
                del res_dyn['problem']; del res_dyn_approx['problem']; del no_screen['problem']
                np.savez('./ResSynthData/'+make_file_name(opt)+'_gamma'+str(opt['switching_gamma'])+'_lambda_'+str(opt['lasso'])+'_seed'+str(seed)+'.npz',\
                         res_dyn=res_dyn,res_dyn_approx=res_dyn_approx,no_screen=no_screen,opt=opt,RC=RC)
                os.remove('./ResSynthData/'+make_file_name(opt)+'_lambda_'+str(opt['lasso'])+'.npz')
            # Plot results
            #traceGaps(res_dyn,res_dyn_approx,opt,RC)   

def MEG_gap_evolution_it_time_tol(opt =dict(), **keywords):
    '''
    Same as gap_evolution_it_time, but used to generate the time colormap 
    as a function of the convergence tolerande and the regularization.
    
    Plot one execution of the dynamic screening for given options. Also plots
    duality gap per iteration and as a function of time.
     
    to modify default option just do first(algo_type = 'FISTA')...
    '''   
    for seed in range(30):
#        seed = 0
        np.random.seed(seed)
        
        lasso_list = [0.2] #[0.5, 0.75, 0.85]
        samp=20; min_reg=0.01; samp_type='log'
        lasso_list = make_pen_param_list(samp,min_reg,samp_type)
#        lasso_list = [0.1]
        
        for lasso in lasso_list:
    
            # MEG - FAuST
#            default =  dict(dict_type = 'MEG', data_type = 'bernoulli-gaussian',N=204, K=8193, lasso=lasso,
#                            data_params = dict(p = 0.001), # p = 0.001 = 8 active sources
#                            stop=dict(dgap_tol=1e-5, max_iter=1000000), scr_type = "GAP", switching='default',
#                            switching_gamma=0.3, algo_type = 'FISTA')# , wstart=True
            # MEG - Low-rank
            default =  dict(dict_type = 'MEG_low-rank', data_type = 'bernoulli-gaussian',N=204, K=8193, lasso=lasso,
                            data_params = dict(p = 0.001), # p = 0.001 = 8 active sources
                            dict_params = dict(ranks = [20, 40, 60, 80, 100] ), # [20, 40, 60, 80, 100, 120], other: [10, 20, 30, 40, 50, 60, 70, 80] [20, 40, 60, 80] 
                            stop=dict(dgap_tol=1e-5, max_iter=1000000), scr_type = "GAP", switching='default',
                            #L = 'fixed', #'backtracking'
                            switching_gamma=0.3, algo_type = 'FISTA')# , wstart=True   
                            
            opt = mergeopt(default, opt, keywords)
            default = default_expe()
            opt = mergeopt(opt, default, keywords)
            
            filename = './ResSynthData/'+make_file_name(opt)+'_gamma'+str(opt['switching_gamma'])+'_lambda_'+str(opt['lasso'])+'_seed'+str(seed)+'.npz'
            # Run only if necessary
            if os.path.isfile(filename):
                Data = np.load(filename)
                res_dyn = Data['res_dyn'][()] 
                res_dyn_approx = Data['res_dyn_approx'][()]
                RC = Data['RC'].tolist()
            else:
                res_dyn, res_dyn_approx, no_screen, RC = expeScrRate(opt=opt)
                del res_dyn['sol']; del res_dyn_approx['sol']; del no_screen['sol']
                del res_dyn['problem']; del res_dyn_approx['problem']; del no_screen['problem']
                np.savez('./ResSynthData/'+make_file_name(opt)+'_gamma'+str(opt['switching_gamma'])+'_lambda_'+str(opt['lasso'])+'_seed'+str(seed)+'.npz',\
                         res_dyn=res_dyn,res_dyn_approx=res_dyn_approx,no_screen=no_screen,opt=opt,RC=RC)
                os.remove('./ResSynthData/'+make_file_name(opt)+'_lambda_'+str(opt['lasso'])+'.npz')
            # Plot results
            #traceGaps(res_dyn,res_dyn_approx,opt,RC)       

def approx_RC_compromise(opt=dict(), **keywords):
    '''
    Plot a Approximation error vs. RC compromise curves for different decays
    
    Comment: for decay_const=0 (i.e. eigenvalues all equal), the error decay
             isn't linear. It is rather the **squared** approximation error that 
             decays linearly.
    '''  
    decay_const_list = [0.1, 0.3, 0.5] #np.linspace(0,0.5,6)
    default = dict(dict_type = 'sukro_approx',data_type = 'bernoulli-gaussian', N=2500,K=10000,\
                    dict_params = dict(N1 = 50, N2 = 50, K1 = 100, K2 = 100,n_kron = range(1,34), svd_decay = 'exponential' ))
    opt = mergeopt(opt, default, keywords)
    opt = mergeopt(opt, default_expe(),keywords)

    testopt(opt)
    for key, val in opt.items():
        exec(key+'='+ repr(val))

    if opt['dict_type'] is not 'sukro_approx':
        raise NotImplementedError('\n Implemented only for SuKro dictionaries')
        
    if 'ResSynthData' not in os.listdir('./'):
        os.mkdir('ResSynthData')

    # plot configurations
    matplotlib.rc('axes', labelsize = 24)
    matplotlib.rc('xtick', labelsize = 24)
    matplotlib.rc('ytick', labelsize = 24)
    matplotlib.rc('axes', titlesize = 24)
    matplotlib.rc('lines', linewidth = 3)
    # Option 1
#    matplotlib.rc('mathtext', fontset='cm')
    # Option 2
    matplotlib.rc('text', usetex=True)
#    matplotlib.rc('font',**{'family':'serif','serif':['Times']})
    matplotlib.rc('font',family='serif' ) #default with usetex=True is Computer Modern Roman
    linestyles = ['-', '--', ':', '-.'] # for gammas
    
    f , (singVals, approxRC) = \
        plt.subplots(2,1,sharex=False,figsize=2*plt.figaspect(1.3))
    f2 , (approx_nkron) = \
        plt.subplots(1,1,sharex=False,figsize=(7.5,5))
    color=iter(plt.cm.autumn(np.linspace(0,1,len(decay_const_list))))
    color=iter(plt.cm.viridis(np.linspace(0,1,len(decay_const_list)+1)))
        
    for decay_const in decay_const_list:
        opt['dict_params']['svd_decay_const'] = decay_const
        
        filename = 'RC-approx_'+make_file_name(opt)
        # Runs only if results are not already available
        if 'ResSynthData' in os.listdir('./') and (filename+'.npz') in os.listdir('./ResSynthData'):
            # Simply load existing results
            Data = np.load('./ResSynthData/'+filename+'.npz')
            RC = Data['RC'][()]            
            normE_all = Data['normE_all'][()]
        else: # Run simulation            
            if decay_const == 0:
                opt['dict_params']['svd_decay'] = 'constant'
            else:
                opt['dict_params']['svd_decay'] = 'exponential'
            problem, opt = GP.generate(opt) # Generate the problem
            
            ########################################################################
            #### Evaluate RC (Relative Complexity) and Approximation errors 
            ########################################################################
            normE_all = []
            #norm2E_all = []
            RC = []
        
            D_sukro = np.zeros_like(problem.D_bis.data)            
            for nkron in range(1,max(opt['dict_params']['n_kron'])+1): # Reconstruct the dictionary until nkron to calculate approximation error
                D_sukro = D_sukro + np.kron(problem.D_bis.A[:,:,nkron-1],problem.D_bis.B[:,:,nkron-1])
                if nkron in opt['dict_params']['n_kron']:
                    D_sukro_norm = D_sukro/np.tile(problem.D_bis.normcoef,(D_sukro.shape[0],1)) # normalizing
                    # Calculate errors
                    E = D_sukro_norm - problem.D.data
                    #norm2E_all.append(np.linalg.norm(E,2))
                    normE_all.append(np.linalg.norm(E,axis=0))
                
                    # Estimate RC
                    problem.D_bis.nkron = nkron
                    RC.append(estimateRC(problem.D_bis,opt,total_it=1000,verbose=True,experimental=False))
                    
            ## Saving data
            np.savez('./ResSynthData/'+filename+'.npz', nkron=opt['dict_params']['n_kron'],RC=RC,normE_all=normE_all) #,norm2E_all)
    
        ########################################################################
        #### Plotting result  
        ########################################################################
        svd_decay = np.exp(-decay_const*np.linspace(0,opt['N']-1,opt['N']))
        
        approxRC.semilogy(RC,np.mean(normE_all,1), linewidth = 6, label = 'decay = %.2f' %decay_const)
        singVals.semilogy(opt['dict_params']['n_kron'],svd_decay[opt['dict_params']['n_kron']], linewidth = 6, label = 'decay = %.2f' %decay_const)
        approx_nkron.semilogy(opt['dict_params']['n_kron'][:21],np.mean(normE_all,1)[:21],c=next(color))
        print '5 : ' + str(np.mean(normE_all,1)[4])
        print '10 : ' + str(np.mean(normE_all,1)[9])
        print '15 : ' + str(np.mean(normE_all,1)[14])
        print '20 : ' + str(np.mean(normE_all,1)[19])

    #approxRC.axis([0,1, 0, 1])
    #approxRC.grid(True) 
    approxRC.set_ylabel("mean $\epsilon_j$")
    approxRC.set_xlabel("Theoretical RC")
    approxRC.legend(fontsize=18,loc=3,frameon=False)

    #singVals.axis([0,1, 0, 1])
    #singVals.grid(True) 
    singVals.set_ylabel("singular value")
    singVals.set_xlabel(r'$n_{\mathrm{kron}}$')
    singVals.legend(fontsize=18,loc=3,frameon=False)
    
    approx_nkron.set_ylabel(r'mean $\epsilon_j$')
    approx_nkron.set_xlabel(r'$n_{\mathrm{kron}}$')    
    approx_nkron.legend(['Hard','Moderate','Easy'],fontsize=24,loc=3,frameon=False)# ['Harsh','Moderate','Mild']

    
    f.savefig('./ResSynthData/RC-approx_compromise_'+make_file_name(opt)+'.pdf',bbox_inches = 'tight',bbox_pad = 2 )
    f2.savefig('./ResSynthData/nkron-approx_compromise_'+make_file_name(opt)+'.pdf',bbox_inches = 'tight',bbox_pad = 2 )    
    f2.savefig('./ResSynthData/nkron-approx_compromise_'+make_file_name(opt)+'.eps',bbox_inches = 'tight',bbox_pad = 2 )
    
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
#        meg_data = sio.loadmat('./datasets/MEG/X_meg.mat') # DEBUG TO DELETE
#        D_dense = Dict(meg_data['X_fixed']) # unstructured MEG matrix # DEBUG TO DELETE
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
        if D.opType == 'sukro':
            struct_cost = 2*D.nkron*D.K1*D.N2*(D.K2 + D.N1)
        elif D.opType == 'low-rank':
            struct_cost = 2*D.nrank*(opt['K'] + opt['N'])
        elif D.opType == 'faust':
            struct_cost = dense_cost*D.faust.density()
        else:
            print "Theoretical RC not implemented for this type of structure, using RC = 0.5"
            RC = 0.5
            
        RC = float(struct_cost)/dense_cost
        print "RC = %1.3f"%(RC)
    
    return RC
    
def estimateRC_faust(D,opt,total_it=1000, verbose=False, experimental=True,dense_vec=False):
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
            if dense_vec:
                p = 1 # only dense vectors
            else:
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
            D*x
            toc = time.time()
            mean_time_struct = mean_time_struct + (toc-tic)/float(total_it)
            tic = time.time()    
            D.transpose()*xT
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
        RC = D.density()
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
#    for key, val in opt.items():
#        exec(key+'='+ repr(val))
   
    problem, opt = GP.generate(opt) # Generate the problem
        
    ########################################################################
    #### Evaluate RC (Relative Complexity) and Approximation errors 
    ########################################################################
    normE_all = []
    norm2E_all = []
    RC = []
    if opt['dict_type'] is 'sukro_approx':    
        opt['dict_params']['nkron_list'] = opt['dict_params']['n_kron'] 
        opt['dict_params']['nkron_max'] = max(opt['dict_params']['nkron_list'])   

        # Verify if already done before
        nkron_list_str = '-'.join(str(e) for e in opt['dict_params']['n_kron']) #converts list into string using '-' as a separator
        filename =  './ResSynthData/RC_normE_decay'+str(opt['dict_params']['svd_decay_const'])+'_'+opt['dict_type']+'-dict_'+opt['data_type']+'-data_N'+str(opt['N'])+'_K'+str(opt['K'])+'_nkron'+nkron_list_str+'.npz'
        if os.path.isfile(filename) and ('reuse' in opt['dict_params']): # Load previously generate factors
            Data = np.load(filename)
            RC = Data['RC'][()].tolist()
            norm2E_all = Data['norm2E_all'][()]
            normE_all = Data['normE_all'][()]
        else:
            D_sukro = np.zeros_like(problem.D_bis.data)      
            
            for nkron in range(1,opt['dict_params']['nkron_max']+1): # Reconstruct the dictionary until nkron to calculate approximation error
                D_sukro = D_sukro + np.kron(problem.D_bis.A[:,:,nkron-1],problem.D_bis.B[:,:,nkron-1])
                if nkron in opt['dict_params']['nkron_list']: #sum(ll,[]) flattens list of lists 'll'
                    D_sukro_norm = D_sukro/np.tile(problem.D_bis.normcoef,(D_sukro.shape[0],1)) # normalizing
                    # Calculate errors
                    E = D_sukro_norm - problem.D.data
                    norm2E_all.append(np.linalg.norm(E,2))
                    normE_all.append(np.linalg.norm(E,axis=0))
                
                    # Estimate RC
                    problem.D_bis.nkron = nkron
                    RC.append(estimateRC(problem.D_bis,opt,total_it=1000,verbose=True,experimental=True))
                    
            if 'reuse' in opt['dict_params']: np.savez(filename,normE_all=normE_all,norm2E_all=norm2E_all,RC=RC,total_it=1000,experimental=True)    

    elif opt['dict_type']=='MEG':
        meg_data = sio.loadmat('./datasets/MEG/X_meg.mat')
        D = meg_data['X_fixed'] # unstructured MEG matrix
        
        opt['K'] = D.shape[1]
        opt['N'] = D.shape[0]
        
        D = Dict(D)

        filenames = ['./datasets/MEG/faust_approx/M_25.mat',
                     './datasets/MEG/faust_approx/M_16.mat',
                     './datasets/MEG/faust_approx/M_8.mat',
                     './datasets/MEG/faust_approx/M_6.mat']
#        filenames = ['./datasets/MEG/faust_approx/M_16.mat'] # M6 seems to give best results
        alphas = [1e12, 1e11, 1e6, 1e7] # Scaling factors for FAuST (to avoid too discrepant values)

        # Load empirical RC , if already done
        RC_filename = './datasets/MEG/faust_approx/RC_M25_16_8_6.npz'
        RC_loaded = False
        if os.path.isfile(RC_filename):
            Data = np.load(RC_filename)
            RC = Data['RC'].tolist()
            RC_loaded = True
            print('Loaded RC: ['+', '.join(map(str, RC))+']')
        
        opt['dict_params'] = []
        for filename, alpha in zip(filenames,alphas):
            # Fast approximation of dictionary
            meg_data = sio.loadmat(filename)
            facts = meg_data['facts']
            del meg_data
                  
            # Develloping FAuST fators
    #        n_layers = facts.shape[1]
    #        D_bis = facts[:,0][0];
    #        for k in range(1,n_layers):
    #            assert(facts[:,k][0].shape[0] == D_bis.shape[1])
    #            D_bis = D_bis.dot(facts[:,k][0])

            # Convert factors in sparse matrices (doesn't change performance)
#            factors = []
#            for k_factor in range(len(facts[0])):
#                factors += [scipy.sparse.csr_matrix(facts[0][k_factor])]
            
            F = Faust(facts.squeeze().tolist(),alpha)            
            D_bis = Dict(F.todense().T,'faust',F.transpose())
            opt['dict_params'].append(D_bis)
            
            # Calculate errors
            E = D.data - D_bis.data
            norm2E_all.append(np.linalg.norm(E,2))
            normE_all.append(np.linalg.norm(E,axis=0))
            
            # Estimate RC - if not already loaded
            if not RC_loaded:
                RC.append(estimateRC_faust(F.transpose(),opt,total_it=10000,verbose=True,experimental=True))           
#                RC.append(estimateRC(D_bis,opt,total_it=10000,verbose=True,experimental=False))
                np.savez(RC_filename,RC=RC,total_it=10000,experimental=True)
            
        del F,E
        
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

    elif opt['dict_type']=='MEG_low-rank':
        
        meg_data = sio.loadmat('./datasets/MEG/X_meg.mat')
        D = meg_data['X_fixed'] # unstructured MEG matrix
        
        ranks_list_str = '-'.join(str(e) for e in opt['dict_params']['ranks']) #converts list into string using '-' as a separator        
        opt['K'] = D.shape[1]
        opt['N'] = D.shape[0]
        
        D = Dict(D)
        D_bis_list = []

        # Calculating approximations - if not already done before
        filename =  './ResSynthData/'+opt['dict_type']+'-dict_'+'_N'+str(opt['N'])+'_K'+str(opt['K'])+'_ranks'+ranks_list_str+'.npz'
        if os.path.isfile(filename): # Load previously generate factors
            Data = np.load(filename)
            D_bis_list =  Data['D_bis_list'][()].tolist()
        else:
            # SVD calculation
            L,S,R = np.linalg.svd(D.data, full_matrices=False)
            for n_rank in opt['dict_params']['ranks']:
                D_bis = np.dot(L[:,:n_rank]*S[:n_rank],R[:n_rank,:])
                D_bis = Dict(D_bis,'low-rank',dict(L=L[:,:n_rank]*S[:n_rank], R=R[:n_rank,:]))
#                normcoef = np.sqrt(np.sum(D_bis**2,0)); D_bis /= np.tile(normcoef,(D_bis.shape[0],1)) # Normalizing columns to unit-norm
                D_bis_list.append(D_bis)
                
            np.savez(filename,D_bis_list=D_bis_list)
            
        # Calculating RC and Approximation error for different ranks - if not done before
        filename =  './ResSynthData/RC_normE_'+opt['dict_type']+'-dict_'+opt['data_type']+'-data_N'+str(opt['N'])+'_K'+str(opt['K'])+'_ranks'+ranks_list_str+'.npz'
        if os.path.isfile(filename): # Load previously generate factors
            Data = np.load(filename)
            RC = Data['RC'][()].tolist()
            norm2E_all = Data['norm2E_all'][()]
            normE_all = Data['normE_all'][()]
            print('Loaded RC: ['+', '.join(map(str, RC))+']')
        else:
            for  rank_idx, n_rank in enumerate(opt['dict_params']['ranks']):
                D_bis = D_bis_list[rank_idx]
                # Calculate errors
                E = D_bis.data - D.data
                norm2E_all.append(np.linalg.norm(E,2))
                normE_all.append(np.linalg.norm(E,axis=0))
            
                # Estimate RC
                total_it = 10000; experimental = True
                RC.append(estimateRC(D_bis,opt,total_it=total_it,verbose=True,experimental=experimental))
                
            np.savez(filename,normE_all=normE_all,norm2E_all=norm2E_all,RC=RC,total_it=total_it,experimental=experimental)                

        opt['dict_params'] = D_bis_list
        
        # Load 'y' for benchmark with Matlab
#        meg_data = sio.loadmat('./datasets/MEG/faust_approx/Data.mat')
#        y = meg_data['Data'][:,0]
#        y = np.expand_dims(y, -1)
#        del meg_data
        
        problem, opt = GP.generate(opt,D=D,D_bis=D_bis_list[0])
        
        if opt['L'] is not 'backtracking':
            opt['L'] = np.linalg.norm(D.data,2) ** 2 # fixed step size
            print('Using fixed step-size!')
    else:
        raise NotImplementedError('\n Implemented only for SuKro dictionaries')

    ########################################################################
    #### Run simulations
    ########################################################################
        
    timeRes, nbIteration, switchIt, sols, flops, dynamicRun, \
    dynamicRun_approx, noScreenRun  = \
        runAllversions(problem, RC, normE_all, norm2E_all, opt, warm_start = None)   
    
    #opt['dict_params'] = [] # this is necessary because np.savez is uncapable of saving this key (also there is an error in make_file_name)
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
    if opt['dict_type']=='MEG':
        del opt['dict_params'] # cannot be saved "no default __reduce__ due to non-trivial __cinit__"
    np.savez('./ResSynthData/'+make_file_name(opt)+'_lambda_'+str(opt['lasso'])+'.npz',\
        scrRate=dynamicRun['screenrate'],radius=dynamicRun['radius'],\
        ObjValue=dynamicRun['objective'], opt=opt, RC=RC,\
        nbIter = dynamicRun['nbIter'],\
        zeros = dynamicRun['zeros'], dGaps = dynamicRun['dGaps'],\
        time = dynamicRun['time'], time_per_it = dynamicRun['time_per_it'],\
        scrRate_approx=dynamicRun_approx['screenrate'],scrRate_est_approx=dynamicRun_approx['screenrate_est'],
        radius_approx=dynamicRun_approx['radius'], ObjValue_approx=dynamicRun_approx['objective'], \
        nbIter_approx = dynamicRun_approx['nbIter'], switch_it = dynamicRun_approx['switch_it'],\
        zeros_approx = dynamicRun_approx['zeros'], dGaps_approx = dynamicRun_approx['dGaps'],\
        dGaps_est_approx = dynamicRun_approx['dGaps_est'],\
        time_approx = dynamicRun_approx['time'], time_per_it_approx = dynamicRun_approx['time_per_it']) #TESTE dgap_est


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
    if not isinstance(dynamicRun_approx['switch_it'], list):
        dynamicRun_approx['switch_it'] = [dynamicRun_approx['switch_it']]
    markers_on1 = [x-1 for x in dynamicRun_approx['switch_it']]
        
    
    flops_ns = flop_calc_it("noScreen",K,N,[], noScreenRun['zeros'],[]) #float(flops["noScreen"])/nbIteration["noScreen"]
    flops_d = flop_calc_it("dynamic",K,N,dynamicRun['screenrate'], dynamicRun['zeros'],[])
    flops_d1 = flop_calc_it("dynamic",K,N,dynamicRun_approx['screenrate'], dynamicRun_approx['zeros'],[],RC,dynamicRun_approx['switch_it'])
           
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

    # Plot GAP evolution
    f  = plt.figure(figsize=1.4*plt.figaspect(0.7))  
    plt.semilogy(dynamicRun['dGaps'], '--k', label = r'$G$ dynamic')
    plt.semilogy(dynamicRun_approx['dGaps'], '-m',label = r'$G$ dynamic approx')
    #plt.semilogy(dynamicRun_approx['dGaps_est'], '-g',label = r'$\tilde{G}$ dynamic approx')
    
    plt.xlabel("Iteration t")
    plt.legend(fontsize=18,loc=3,frameon=False)
    
    f.savefig('./ResSynthData/GAP_evolution_' + make_file_name(opt)+\
        '_lasso'+str(opt['lasso'])+'.pdf',bbox_inches = 'tight',bbox_pad = 2 )
    
    return dynamicRun, dynamicRun_approx, noScreenRun, RC
    
    
def flop_calc_it(EmbedTest,K,N,screenrate,zeros,Gr,RC=1,switch_it=0): # Returns the iteration complexity
    nb_gr = len(Gr)

    # Adapted for Multiple dictionaries: inputs must now be lists (even if having one single element)
    if not isinstance(RC,list):
        RC = [RC]
    if not isinstance(switch_it,list):
        switch_it = [switch_it]
    
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
    if opt['dict_type']  in {'sukro','low-rank','MEG'}: 
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
                 label = 'A-D'+Gstr+opt['scr_type'], #+' '+str(opt['dict_params']['n_kron']), 
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
        
def traceGaps(dyn, dyn_approx, opt, RC):
    """
    Plot the duality gap evolution over the iterations and over time.
    """    
    
    ####  plot the results
    markersize = 12
 
    # config plot
    matplotlib.rc('axes', labelsize = 24) #same as matplotlib.rcParams['aves.labelsize'] = 24
    matplotlib.rc('xtick', labelsize = 24)
    matplotlib.rc('ytick', labelsize = 24)
    matplotlib.rc('axes', titlesize = 24)
    matplotlib.rc('lines', linewidth = 3)    
#    matplotlib.rc('mathtext', fontset='cm') # stix: designed to blend well with times 
    matplotlib.rc('text', usetex=True)
    matplotlib.rc('font',**{'family':'serif','serif':['Times']})
    
    #last_it = 200
    last_it = dyn_approx['nbIter']
    
    # Figure
    f , ((plt1, plt3), (plt2, axFlops_it)) = plt.subplots(2,2,figsize=(13.5,9));  plt.subplots_adjust(wspace=0.5, hspace=0.5) #figsize=(15,10)

    
    A0_idx = range(dyn_approx['switch_it'][0]+1)
    A1_idx = range(dyn_approx['switch_it'][0],dyn_approx['switch_it'][1]+1)
    A2_idx = range(dyn_approx['switch_it'][1],dyn_approx['switch_it'][2]+1)    
    A_idx = range(dyn_approx['switch_it'][2],last_it)
    
    ## GAP - ITERATIONS
#    f , (plt1) = plt.subplots(1,1,figsize=(7.5,5));  plt.subplots_adjust(wspace=0, hspace=0) # figsize = [width, height],(8.5, 5.6), 1.4*plt.figaspect(37./50)
    plt1.semilogy(A0_idx, dyn_approx['dGaps'][A0_idx], '-m')
    plt1.semilogy(A1_idx, dyn_approx['dGaps'][A1_idx], '-g')
    plt1.semilogy(A2_idx, dyn_approx['dGaps'][A2_idx], '-r')
    plt1.semilogy(A_idx, dyn_approx['dGaps'][A_idx], '-k')
    line, = plt1.semilogy(dyn['dGaps'], '--k')
    # plt1.semilogy(Data['dGaps_est_approx3'][:last_it], '--k', label = r'$\tilde{\tilde{G}}$') #Coincides with G    
    
    plt1.set_xlabel("Iteration t")
    plt1.set_ylabel("Duality gap")
#    plt1.set_ylim((4e-6,0.7))
    
    
    # Multicolored line in legend
    xy = np.array([range(last_it), dyn_approx['dGaps']]).T.reshape(-1, 1, 2)
    segments = np.hstack([xy[:-1], xy[1:]])
    lc = LineCollection(segments, cmap= matplotlib.colors.ListedColormap(['m','g','r','k']),
                        norm=plt.Normalize(0, 10), linewidth=3)

    plt1.legend([lc,line], [r'$\{\tilde{\mathbf{A}}^i\}_{i=0}^{4}$',r'$\mathbf{A}$'],\
                handler_map={lc: HandlerColorLineCollection(numpoints=4)}, fontsize=22,loc=3,frameon=False)
                
#    plt1.legend(fontsize=18,loc=3,frameon=False)

    extent = plt1.get_tightbbox(f.canvas.get_renderer()).transformed(f.dpi_scale_trans.inverted()) # gets the extent of the desired axis (plus ticks and labels) within the figure
#    extent = 'tight'
    
    f.savefig('./ResSynthData/GAP_evolution_' + make_file_name(opt)+\
        '_lasso'+str(opt['lasso'])+'.pdf',bbox_inches = extent) #,bbox_pad = 2
    f.savefig('./ResSynthData/GAP_evolution_' + make_file_name(opt)+\
        '_lasso'+str(opt['lasso'])+'.eps',bbox_inches = extent) #,bbox_pad = 2 

        
    ## GAP RATIO
#    f , (plt2) = plt.subplots(1,1,figsize=(7.5,5));  plt.subplots_adjust(wspace=0.12, hspace=0) # figaspect = height/width
    plt2.plot(A0_idx,(dyn_approx['dGaps_est'][A0_idx]/dyn_approx['dGaps'][A0_idx]), '-m')
    plt2.plot(A1_idx,(dyn_approx['dGaps_est'][A1_idx]/dyn_approx['dGaps'][A1_idx]), '-g')
    plt2.plot(A2_idx,(dyn_approx['dGaps_est'][A2_idx]/dyn_approx['dGaps'][A2_idx]), '-r')
#    plt2.legend(fontsize=18,loc=1,frameon=False)
    
    plt2.set_xlabel("Iteration t")
    plt2.set_ylabel(r'Gap ratio $\gamma_t$')
    plt2.set_ylim((0,1.05))
    plt2.set_xlim(plt1.get_xlim())
    plt2.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))

    # Add tick for Gamma - not working
    f.canvas.draw()
    labels = [ w.get_text() for w in plt2.get_yticklabels()]
    labels+=[r'$\Gamma$']
    plt2.set_yticklabels(labels)
    plt2.set_yticks(list(plt2.get_yticks()) + [opt['switching_gamma']])
    plt2.set_ylim((0,1.05))
    plt2.plot([-50, last_it+50], [opt['switching_gamma'], opt['switching_gamma']],':',lw=1, color="0.3")


    extent = plt2.get_tightbbox(f.canvas.get_renderer()).transformed(f.dpi_scale_trans.inverted()) # gets the extent of the desired axis (plus ticks and labels) within the figure
#    extent = 'tight'
    
    f.savefig('./ResSynthData/GAP_ratio_' + make_file_name(opt)+\
        '_lasso'+str(opt['lasso'])+'.pdf',bbox_inches = extent,bbox_pad = 2 )
    f.savefig('./ResSynthData/GAP_ratio_' + make_file_name(opt)+\
        '_lasso'+str(opt['lasso'])+'.eps',bbox_inches = extent,bbox_pad = 2 )
        
    ## GAP - TIME
#    f , (plt3) = plt.subplots(1,1,figsize=(7.5,5));  plt.subplots_adjust(wspace=0.12, hspace=0) # figaspect = height/width
    time_elapsed = np.cumsum(dyn_approx['time_per_it'])
    time_elapsed_conv = np.cumsum(dyn['time_per_it'])
    plt3.semilogy(time_elapsed[A0_idx], dyn_approx['dGaps'][A0_idx], '-m',label = r'$\tilde{\mathbf{A}}^{(0)}$')
    plt3.semilogy(time_elapsed[A1_idx], dyn_approx['dGaps'][A1_idx], '-g',label = r'$\tilde{\mathbf{A}}^{(1)}$')
    plt3.semilogy(time_elapsed[A2_idx], dyn_approx['dGaps'][A2_idx], '-r',label = r'$\tilde{\mathbf{A}}^{(2)}$')
    plt3.semilogy(time_elapsed[A_idx], dyn_approx['dGaps'][A_idx], '-k',label = r'$\mathbf{A}$')
    plt3.semilogy(time_elapsed_conv, dyn['dGaps'][:-1], '--k', label = r'$\mathbf{A} from the beginning$')
    
    plt3.set_xlabel("Time (s)")
    plt3.set_ylabel("Duality gap")
#    plt3.set_ylim((4e-6,0.7))
#    plt3.legend(fontsize=18,loc=3,frameon=False)

    extent = plt3.get_tightbbox(f.canvas.get_renderer()).transformed(f.dpi_scale_trans.inverted()) # gets the extent of the desired axis (plus ticks and labels) within the figure
#    extent = 'tight'
    
    f.savefig('./ResSynthData/GAP_evolution_time_' + make_file_name(opt)+\
        '_lasso'+str(opt['lasso'])+'.pdf',bbox_inches = extent,bbox_pad = 2 )
    f.savefig('./ResSynthData/GAP_evolution_time_' + make_file_name(opt)+\
        '_lasso'+str(opt['lasso'])+'.eps',bbox_inches = extent,bbox_pad = 2 )

    ## Nb Flops
#    f , (axFlops_it) = plt.subplots(1,1,figsize=(7.5,5));  plt.subplots_adjust(wspace=0.12, hspace=0) # figaspect = height/width
    K = opt['K']
    N = opt['N']
#    length = dyn['nbIter']+1
#    length_approx = dyn_approx['nbIter']+1
    if not isinstance(dyn_approx['switch_it'], list):
        dyn_approx['switch_it'] = [dyn_approx['switch_it']]
#    markers_on1 = [x-1 for x in dyn_approx['switch_it']]
    flops_d = flop_calc_it("dynamic",K,N,dyn['screenrate'], dyn['zeros'],[])
    flops_d1 = flop_calc_it("dynamic",K,N,dyn_approx['screenrate'], dyn_approx['zeros'],[],RC,dyn_approx['switch_it'])

    axFlops_it.plot(A0_idx,flops_d1[A0_idx], '-m')
    axFlops_it.plot(A1_idx,flops_d1[A1_idx], '-g')
    axFlops_it.plot(A2_idx,flops_d1[A2_idx], '-r')
    axFlops_it.plot(A_idx,flops_d1[A_idx], '-k')
    axFlops_it.plot(flops_d, '--k') #linewidth = 6, markevery=[length-1])
#    axFlops_it.grid(True)         
    axFlops_it.set_ylim((0,1.15*max(max(flops_d),max(flops_d1))))
    axFlops_it.set_ylabel("Iteration cost (\# flops)") #,fontsize = 24)
    axFlops_it.set_xlabel("Iteration t")
    #axFlops_it.legend(fontsize=22,loc=3,frameon=False)
    axFlops_it.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(MyFormatter))#(plt.LogFormatter(10,  labelOnlyBase=False))
#    axFlops_it.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.0e'))
#    axFlops_it.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True))
    
    extent = axFlops_it.get_tightbbox(f.canvas.get_renderer()).transformed(f.dpi_scale_trans.inverted()) # gets the extent of the desired axis (plus ticks and labels) within the figure
#    extent = 'tight'
    
    f.savefig('./ResSynthData/Simu_screenRate_'+make_file_name(opt)+\
        '_lasso'+str(opt['lasso'])+'approx.pdf',bbox_inches = extent,bbox_pad = 2 )
    f.savefig('./ResSynthData/Simu_screenRate_'+make_file_name(opt)+\
        '_lasso'+str(opt['lasso'])+'approx.eps',bbox_inches = extent,bbox_pad = 2 )

def MyFormatter(x,lim):
    if x == 0:
        return 0
#    return '{:1.0f}e+{:1.0f}'.format(x/10**np.floor(np.log10(x)),int(np.log10(x)))
    return r'${:1.0f}\!\cdot\!\!10^{:1.0f}$'.format(x/10**np.floor(np.log10(x)),int(np.log10(x)))

def runAllversions(problem=None, RC=1, normE_all=[], norm2E_all=[], opt={}, warm_start = None, **keywords):
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
                                warm_start = warm_start, switching_gamma = opt.get('switching_gamma',2e-1))                                                              
    elif (opt['dict_type']=='MEG') or (opt['dict_type']=='MEG_low-rank'):
        # At this point: D = slow (=fast+E), D_bis = fast
        problem.D, problem.D_bis = problem.D_bis, problem.D # start with the approximate (fast) dictionary 
        
#        res_dyn_approx = solver_approx( problem=problem, normE=normE_all[0], norm2E = norm2E_all[0], RC=RC[0], L=opt['L'], stop=opt['stop'], switching=opt['switching'], \
#                                        scr_type=opt['scr_type'], \
#                                        EmbedTest='dynamic', algo_type=opt['algo_type'], \
#                                        warm_start = warm_start)
        
#        import cProfile
#        pr = cProfile.Profile()
#        pr.enable()
 
        # gives same result as before
        res_dyn_approx = \
            solver_multiple(problem=problem, normE_all=normE_all, norm2E_all = norm2E_all, RC=RC, L=opt['L'], stop=opt['stop'], switching=opt['switching'], \
                                scr_type=opt['scr_type'], \
                                dict_specs = opt['dict_params'], \
                                EmbedTest='dynamic', algo_type=opt['algo_type'], \
                                warm_start = warm_start, switching_gamma = opt.get('switching_gamma',2e-1))

#        pr.disable()         
#        pr.print_stats(sort='time')

    else:
        raise NotImplementedError('\n Not implemented for this type of dictionary')

#    pr = cProfile.Profile()
#    pr.enable()

    # Exact Screening - Dynamic
    res_dyn = solver(problem=problem, L=opt['L'], stop=opt['stop'],\
                            scr_type=opt['scr_type'], \
                            EmbedTest='dynamic', algo_type=opt['algo_type'], \
                            warm_start = warm_start) 
#    pr.disable()         
#    pr.print_stats(sort='time')

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
                   
    return timeRes,nbIteration,switchIt,Solution, flops,res_dyn, res_dyn_approx, res_noScreen
    

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
    elif opt['dict_type']=='MEG':
        meg_data = sio.loadmat('./datasets/MEG/X_meg.mat')
        D = meg_data['X_fixed'] # unstructured MEG matrix
        
        opt['K'] = D.shape[1]
        opt['N'] = D.shape[0]
        
        D = Dict(D)

        filenames = ['./datasets/MEG/faust_approx/M_25.mat',
                     './datasets/MEG/faust_approx/M_16.mat',
                     './datasets/MEG/faust_approx/M_8.mat',
                     './datasets/MEG/faust_approx/M_6.mat']
#        filenames = ['./datasets/MEG/faust_approx/M_16.mat'] # M6 seems to give best results
        alphas = [1e12, 1e11, 1e6, 1e7] # Scaling factors for FAuST (to avoid too discrepant values)
        
        # Load empirical RC , if already done
        RC_filename = './datasets/MEG/faust_approx/RC_M25_16_8_6.npz'
        RC_loaded = False
        if os.path.isfile(RC_filename):
            Data = np.load(RC_filename)
            RC = Data['RC'].tolist()
            RC_loaded = True
            print('Loaded RC: ['+', '.join(map(str, RC))+']')

        
        opt['dict_params'] = []
        for filename, alpha in zip(filenames,alphas):
            # Fast approximation of dictionary
            meg_data = sio.loadmat(filename)
            facts = meg_data['facts']
            del meg_data
                  
            # Develloping FAuST fators
    #        n_layers = facts.shape[1]
    #        D_bis = facts[:,0][0];
    #        for k in range(1,n_layers):
    #            assert(facts[:,k][0].shape[0] == D_bis.shape[1])
    #            D_bis = D_bis.dot(facts[:,k][0])

            # Convert factors in sparse matrices (doesn't change performance)
#            factors = []
#            for k_factor in range(len(facts[0])):
#                factors += [scipy.sparse.csr_matrix(facts[0][k_factor])]
            
            F = Faust(facts.squeeze().tolist(),alpha)            
            D_bis = Dict(F.todense().T,'faust',F.transpose())
            opt['dict_params'].append(D_bis)
            
            # Calculate errors
            E = D.data - D_bis.data
            norm2E_all.append(np.linalg.norm(E,2))
            normE_all.append(np.linalg.norm(E,axis=0))
            
            # Estimate RC - if not already loaded
            if not RC_loaded:
                RC.append(estimateRC_faust(F.transpose(),opt,total_it=10000,verbose=True,experimental=True))           
#                RC.append(estimateRC(D_bis,opt,total_it=10000,verbose=True,experimental=False))
                np.savez(RC_filename,RC=RC,total_it=10000,experimental=True)
            
        del F,E
        
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
    else:
        raise NotImplementedError('\n Implemented only for SuKro dictionaries')

    
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

    switchIter = dict(noScreen=np.zeros((nblambdas,avg,len(RC))),\
                static=np.zeros((nblambdas,avg,len(RC))),\
                dynamic=np.zeros((nblambdas,avg,len(RC))), \
                dynamic_approx=np.zeros((nblambdas,avg,len(RC))))
                
#    switchIter = dict(noScreen=np.zeros((nblambdas,avg,len(opt['dict_params']['n_kron']))),\
#                static=np.zeros((nblambdas,avg,len(opt['dict_params']['n_kron']))),\
#                dynamic=np.zeros((nblambdas,avg,len(opt['dict_params']['n_kron']))), \
#                dynamic_approx=np.zeros((nblambdas,avg,len(opt['dict_params']['n_kron']))))
                
    nbFlops = dict(noScreen=np.zeros((nblambdas,avg), dtype = float),\
                static=np.zeros((nblambdas,avg), dtype = float),\
                dynamic=np.zeros((nblambdas,avg), dtype = float), \
                dynamic_approx=np.zeros((nblambdas,avg), dtype = float))
                
    xFinal = dict(noScreen=np.zeros((nblambdas,avg,opt['K'])),\
                static=np.zeros((nblambdas,avg,opt['K'])),\
                dynamic=np.zeros((nblambdas,avg,opt['K'])),\
                dynamic_approx=np.zeros((nblambdas,avg,opt['K'])))
                
    sig = np.zeros((opt['nbRuns'],opt['N']))
    
#    # Evaluate RC (Relative Complexity) and Approximation errors
#    normE_all = []
#    norm2E_all = []
#    RC = []
#    if opt['dict_type'] is 'sukro_approx':
#        D_sukro = np.zeros_like(problem.D_bis.data)            
#        for nkron in range(1,max(opt['dict_params']['n_kron'])+1): # Reconstruct the dictionary until nkron to calculate approximation error
#            D_sukro = D_sukro + np.kron(problem.D_bis.A[:,:,nkron-1],problem.D_bis.B[:,:,nkron-1])
#            if nkron in opt['dict_params']['n_kron']:
#                D_sukro_norm = D_sukro/np.tile(problem.D_bis.normcoef,(D_sukro.shape[0],1)) # normalizing
#                # Calculate errors
#                E = D_sukro_norm - problem.D.data
#                norm2E_all.append(np.linalg.norm(E,2))
#                normE_all.append(np.linalg.norm(E,axis=0))
#            
#                # Estimate RC
#                problem.D_bis.nkron = nkron
#                RC.append(estimateRC(problem.D_bis,opt,total_it=1000,verbose=True,experimental=False))
#    elif opt['dict_type']=='MEG':
#        if opt['wstart']: raise ValueError('Warm start is not available for this configuration, since the input vector y is regenerated every time.')
#
#        meg_data = sio.loadmat('./datasets/MEG/X_meg.mat')
#        D = meg_data['X_fixed'] # unstructured MEG matrix
#        
#        opt['K'] = D.shape[1]
#        opt['N'] = D.shape[0]
#        
#        D = Dict(D)
#
##        filenames = ['./datasets/MEG/faust_approx/M_6.mat',
##                     './datasets/MEG/faust_approx/M_8.mat',
##                     './datasets/MEG/faust_approx/M_16.mat',
##                     './datasets/MEG/faust_approx/M_25.mat']
#        filenames = ['./datasets/MEG/faust_approx/M_6.mat']
#        
#        opt['dict_params'] = []
#        for filename in filenames:
#            # Fast approximation of dictionary
#            meg_data = sio.loadmat(filename)
#            facts = meg_data['facts']
#            del meg_data
#                  
#            # Develloping FAuST fators
#    #        n_layers = facts.shape[1]
#    #        D_bis = facts[:,0][0];
#    #        for k in range(1,n_layers):
#    #            assert(facts[:,k][0].shape[0] == D_bis.shape[1])
#    #            D_bis = D_bis.dot(facts[:,k][0])
#    #                
#    #        D_bis = Dict(D_bis.T) #TODO create opType="faust"
#            F = Faust(facts.squeeze().tolist())
#            opt['dict_params'].append(F)
#            D_bis = Dict(F.todense().T,'faust',F.transpose())
#            
#            # Calculate errors
#            E = D.data - D_bis.data
#            norm2E_all.append(np.linalg.norm(E,2))
#            normE_all.append(np.linalg.norm(E,axis=0))
#            
#            # Estimate RC
#            RC.append(estimateRC(D_bis,opt,total_it=1000,verbose=True,experimental=False))
#            
#        del F,E
#        
#        # Test synthetic error matrix
##        E = np.sqrt(1./opt['N'])*np.random.randn(opt['N'],opt['K'])
##        normE = 1e-3
##        D_bis = D.data + normE*E
##        D_bis = Dict(D_bis)
#
#        # Load 'y' for benchmark with Matlab
##        meg_data = sio.loadmat('./datasets/MEG/faust_approx/Data.mat')
##        y = meg_data['Data'][:,0]
##        y = np.expand_dims(y, -1)
##        del meg_data
#        
#        problem, opt = GP.generate(opt,D=D,D_bis=D_bis)
#    else:
#        raise NotImplementedError('\n Implemented only for SuKro dictionaries')

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
                runAllversions(problem,RC,normE_all,norm2E_all,opt,start)
            for key in timeRes.iterkeys(): 
                timeRes[key][nblambdas-i-1,j] = timeIt[key]
                nbIter[key][nblambdas-i-1,j] = nbIt[key]
                switchIter[key][nblambdas-i-1,j] = switchIt[key]
                nbFlops[key][nblambdas-i-1,j] = flops[key]
                if not opt['dict_type'] =="MNIST"  :
                    xFinal[key][nblambdas-i-1,j] = res[key].flatten()
        print "problem %d over %d"%(j+1,avg)           
                 
    print('Done') 

    opt['dict_params'] = [] # this is necessary because np.savez is uncapable of saving this key (also there is an error in make_file_name)
    
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
                runAllversions(problem,RC,opt,start)
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
    
################################################################
###  Redefining functions: allow running many variations once
################################################################
   
def runProtocol(opt={},**keywords):
    """
    For a given number of iterations, generates a problem and solves it for
    all lambda/lambda*, optim algo and screening rule provided by the user
    
    Return average results for opt['nbRuns'] problems
    """
    #### handles options
    default = default_expe()
                
    opt = mergeopt(opt, default, keywords)

    pen_param_list = make_pen_param_list(opt['samp'],opt['min_reg'],opt['samp_type'])
    opt['lasso'] = 0.99


    # Evaluate RC (Relative Complexity) and Approximation errors
    normE_all = []
    norm2E_all = []
    RC = []
    if opt['dict_type'] is 'sukro_approx':
        opt['dict_params']['nkron_list'] = sorted(set(sum(opt['dict_params']['n_kron'], [])))  #sum(ll,[]) flattens list of lists 'll' and set() removes redundancy
        opt['dict_params']['nkron_max'] = max(opt['dict_params']['nkron_list'])    
#
#        # Generate problem
        problem, opt = GP.generate(opt)

        # Verify if already done before
        nkron_list_str = '-'.join(str(e) for e in opt['dict_params']['nkron_list']) #converts list into string using '-' as a separator
        filename =  './ResSynthData/RC_normE_decay'+str(opt['dict_params']['svd_decay_const'])+'_'+opt['dict_type']+'-dict_'+opt['data_type']+'-data_N'+str(opt['N'])+'_K'+str(opt['K'])+'_nkron'+nkron_list_str+'.npz'
        if os.path.isfile(filename) and ('reuse' in opt['dict_params']): # Load previously generate factors
            Data = np.load(filename)
            RC = Data['RC'][()]
            norm2E_all = Data['norm2E_all'][()]
            normE_all = Data['normE_all'][()]
        else:
            D_sukro = np.zeros_like(problem.D_bis.data)      
            
            for nkron in range(1,opt['dict_params']['nkron_max']+1): # Reconstruct the dictionary until nkron to calculate approximation error
                D_sukro = D_sukro + np.kron(problem.D_bis.A[:,:,nkron-1],problem.D_bis.B[:,:,nkron-1])
                if nkron in opt['dict_params']['nkron_list']: #sum(ll,[]) flattens list of lists 'll'
                    D_sukro_norm = D_sukro/np.tile(problem.D_bis.normcoef,(D_sukro.shape[0],1)) # normalizing
                    # Calculate errors
                    E = D_sukro_norm - problem.D.data
                    norm2E_all.append(np.linalg.norm(E,2))
                    normE_all.append(np.linalg.norm(E,axis=0))
                
                    # Estimate RC
                    problem.D_bis.nkron = nkron
                    RC.append(estimateRC(problem.D_bis,opt,total_it=1000,verbose=True,experimental=True))
                    
            np.savez(filename,normE_all=normE_all,norm2E_all=norm2E_all,RC=RC,total_it=1000,experimental=True)

        # Generate and save N sukro problems
#        if 'ResSynthData' not in os.listdir('./'):
#            os.mkdir('ResSynthData')
#        for k_run in range(opt['nbRuns']):
#            filename =  './ResSynthData/'+str(k_run)+opt['dict_type']+'-dict_'+opt['data_type']+\
#                        '-data_N'+str(opt['N'])+'_K'+str(opt['K'])+'_decay'+ str(opt['dict_params']['svd_decay_const']) +'.npz'
#            if not os.path.isfile(filename):
#                # Generate problem
#                problem, opt = GP.generate(opt)
#                
#                # Save the generated problem and normE/RC vectors
#                # Filename e.g. '1_sukro_approx-dict_bernoulli-gaussian-data_N2500_K10000_decay0.5'
#                np.savez('./ResSynthData/'+str(k_run)+opt['dict_type']+'-dict_'+opt['data_type']+
#                         '-data_N'+str(opt['N'])+'_K'+str(opt['K'])+'_decay'+ str(opt['dict_params']['svd_decay_const']) +'.npz',\
#                         problem=problem)        
#        
#        D_sukro = np.zeros_like(problem.D_bis.data)      
#        
#        for nkron in range(1,opt['dict_params']['nkron_max']+1): # Reconstruct the dictionary until nkron to calculate approximation error
#            D_sukro = D_sukro + np.kron(problem.D_bis.A[:,:,nkron-1],problem.D_bis.B[:,:,nkron-1])
#            if nkron in opt['dict_params']['nkron_list']: #sum(ll,[]) flattens list of lists 'll'
#                D_sukro_norm = D_sukro/np.tile(problem.D_bis.normcoef,(D_sukro.shape[0],1)) # normalizing
#                # Calculate errors
#                E = D_sukro_norm - problem.D.data
#                norm2E_all.append(np.linalg.norm(E,2))
#                normE_all.append(np.linalg.norm(E,axis=0))
#            
#                # Estimate RC
#                problem.D_bis.nkron = nkron
#                RC.append(estimateRC(problem.D_bis,opt,total_it=1000,verbose=True,experimental=False))
    elif opt['dict_type']=='MEG':
        meg_data = sio.loadmat('./datasets/MEG/X_meg.mat')
        D = meg_data['X_fixed'] # unstructured MEG matrix
        
        opt['K'] = D.shape[1]
        opt['N'] = D.shape[0]
        
        D = Dict(D)

        filenames = ['./datasets/MEG/faust_approx/M_25.mat',
                     './datasets/MEG/faust_approx/M_16.mat',
                     './datasets/MEG/faust_approx/M_8.mat',
                     './datasets/MEG/faust_approx/M_6.mat']
#        filenames = ['./datasets/MEG/faust_approx/M_16.mat'] # M6 seems to give best results
        alphas = [1e12, 1e11, 1e6, 1e7] # Scaling factors for FAuST (to avoid too discrepant values)
        
        opt['dict_params'] = []
        for filename, alpha in zip(filenames,alphas):
            # Fast approximation of dictionary
            meg_data = sio.loadmat(filename)
            facts = meg_data['facts']
            del meg_data

            # Convert factors in sparse matrices (doesn't change performance)
#            factors = []
#            for k_factor in range(len(facts[0])):
#                factors += [scipy.sparse.csr_matrix(facts[0][k_factor])]
            
            F = Faust(facts.squeeze().tolist(),alpha)            
            D_bis = Dict(F.todense().T,'faust',F.transpose())
            opt['dict_params'].append(D_bis)
            
            # Calculate errors
            E = D.data - D_bis.data
            norm2E_all.append(np.linalg.norm(E,2))
            normE_all.append(np.linalg.norm(E,axis=0))
            
            # Estimate RC
            RC.append(estimateRC(D_bis,opt,total_it=10000,verbose=True,experimental=False))
            
        del F,E
       
        problem, opt = GP.generate(opt,D=D,D_bis=D_bis)
    else:
        raise NotImplementedError('\n Implemented only for SuKro dictionaries')

    
    avg = opt['nbRuns']
    nblambdas = len(pen_param_list)
    
    # Number of dynamic approx simulations
    if opt['dict_type'] is 'sukro_approx':
        n_approx = len(opt['dict_params']['n_kron'])*len(opt['switching_gamma'])
    else:
        n_approx = 1
    
    timeRes = dict(noScreen=np.zeros((nblambdas,avg), dtype = float),\
                static=np.zeros((nblambdas,avg), dtype = float),\
                dynamic=np.zeros((nblambdas,avg), dtype = float), \
                dynamic_approx=np.zeros((n_approx,nblambdas,avg), dtype = float))
                
    nbIter = dict(noScreen=np.zeros((nblambdas,avg)),\
                static=np.zeros((nblambdas,avg)),\
                dynamic=np.zeros((nblambdas,avg)), \
                dynamic_approx=np.zeros((n_approx,nblambdas,avg)))

#    switchIter = dict(noScreen=np.zeros((nblambdas,avg,len(RC))),\
#                static=np.zeros((nblambdas,avg,len(RC))),\
#                dynamic=np.zeros((nblambdas,avg,len(RC))), \
#                dynamic_approx=np.zeros((n_approx,nblambdas,avg,len(RC))))
                
                
    nbFlops = dict(noScreen=np.zeros((nblambdas,avg), dtype = float),\
                static=np.zeros((nblambdas,avg), dtype = float),\
                dynamic=np.zeros((nblambdas,avg), dtype = float), \
                dynamic_approx=np.zeros((n_approx,nblambdas,avg), dtype = float))
                
#    xFinal = dict(noScreen=np.zeros((nblambdas,avg,opt['K'])),\
#                static=np.zeros((nblambdas,avg,opt['K'])),\
#                dynamic=np.zeros((nblambdas,avg,opt['K'])),\
#                dynamic_approx=np.zeros((n_approx,nblambdas,avg,opt['K'])))
    
                
    sig = np.zeros((opt['nbRuns'],opt['N']))
    
    opt_j = opt.copy()
    
    for j_rule in opt['scr_type']:
        opt_j['scr_type'] = j_rule
        for j_algo in opt['algo_type']:
            opt_j['algo_type'] = j_algo
            for j_stop in opt['stop']:
                opt_j['stop'] = j_stop
                
                for j in range(avg):
                    start = None
                    res = None
                    problem, opt_j = GP.generate(opt_j, D = problem.D, D_bis = problem.D_bis)
            
                    star,lstar = problem.getStar() 
                    sig[j,:] = problem.y.flatten()
                    for i,lasso_ in enumerate(pen_param_list[::-1]):
                        if not opt_j['wstart']:
                            start = None
                        elif res!=None:
                            start = res['noScreen']
                        opt_j['lasso'] = lasso_
                        problem.pen_param = opt_j['lasso']*lstar
                        # Run
                        timeIt, nbIt, switchIt, res, flops = \
                            runVersions(problem,RC,normE_all,norm2E_all,opt_j,start)
                        # Store results
                        for key in {'noScreen','static','dynamic'}:
                            timeRes[key][nblambdas-i-1,j] = timeIt[key]
                            nbIter[key][nblambdas-i-1,j] = nbIt[key]
                            nbFlops[key][nblambdas-i-1,j] = flops[key]
                        for k_approx in range(n_approx):
                            timeRes['dynamic_approx'][k_approx,nblambdas-i-1,j] = timeIt['dynamic_approx'][k_approx]
                            nbIter['dynamic_approx'][k_approx,nblambdas-i-1,j] = nbIt['dynamic_approx'][k_approx]
                            nbFlops['dynamic_approx'][k_approx,nblambdas-i-1,j] = flops['dynamic_approx'][k_approx]
                    print "problem %d over %d"%(j+1,avg)           
                     
                    print('Done') 
                
                ## Save results
                if 'ResSynthData' not in os.listdir('./'):
                    os.mkdir('ResSynthData')
                # Light save - xFinal (solution vector) is not saved
                np.savez('./ResSynthData/'+make_file_name(opt_j)+'_light_Protocol.npz',\
                    timeRes=timeRes, nbIter=nbIter, opt=opt_j,\
                    nbFlops=nbFlops,sig=sig,RC=RC)
                # Draw results
                traceProtocol(timeRes, nbIter, nbFlops ,opt_j)

def runVersions(problem=None, RC=1, normE_all=[], norm2E_all=[], opt={}, warm_start = None, **keywords):
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
    # No Screening                        
    res_noScreen = solver(problem=problem, L=opt['L'], stop=opt['stop'],\
                            scr_type=opt['scr_type'], \
                            EmbedTest='noScreen', algo_type=opt['algo_type'], \
                            warm_start = warm_start)  
                                
    # Exact Screening - Static                  
    res_static = solver(problem=problem, L=opt['L'], stop=opt['stop'],\
                            scr_type=opt['scr_type'], \
                            EmbedTest='static', algo_type=opt['algo_type'], \
                            warm_start = warm_start)

    # Exact Screening - Dynamic
    res_dyn = solver(problem=problem, L=opt['L'], stop=opt['stop'],\
                            scr_type=opt['scr_type'], \
                            EmbedTest='dynamic', algo_type=opt['algo_type'], \
                            warm_start = warm_start) 
  
    
    # Stable Screening
                                
    if opt['dict_type'] == 'sukro_approx':
        res_dyn_approx_list = []
        for n_kron in opt['dict_params']['n_kron']:
            # Find indexes in normE_all corresponding to current n_kron
            idx_list = [opt['dict_params']['nkron_list'].index(x) for x in n_kron]
            
            for gamma in opt['switching_gamma']:
                # At this point: D = slow (=fast+E), D_bis = fast
                problem.D, problem.D_bis = problem.D_bis, problem.D # start with the approximate (fast) dictionary
                
                res_dyn_approx = \
                    solver_multiple(problem=problem, normE_all=[normE_all[i] for i in idx_list], norm2E_all=[norm2E_all[i] for i in idx_list], RC=[RC[i] for i in idx_list], L=opt['L'], stop=opt['stop'], switching=opt['switching'], \
                                        scr_type=opt['scr_type'], \
                                        dict_specs = n_kron, \
                                        EmbedTest='dynamic', algo_type=opt['algo_type'], \
                                        warm_start = warm_start, switching_gamma = gamma)
                #res_dyn_approx.pop('sol', None) # removing heavy results
                res_dyn_approx_list.append(res_dyn_approx)
                                
    elif opt['dict_type']=='MEG':
        # At this point: D = slow (=fast+E), D_bis = fast
        problem.D, problem.D_bis = problem.D_bis, problem.D # start with the approximate (fast) dictionary 
        
        res_dyn_approx_list = \
            solver_multiple(problem=problem, normE_all=normE_all, norm2E_all = norm2E_all, RC=RC, L=opt['L'], stop=opt['stop'], switching=opt['switching'], \
                                scr_type=opt['scr_type'], \
                                dict_specs = opt['dict_params'], \
                                EmbedTest='dynamic', algo_type=opt['algo_type'], \
                                warm_start = warm_start, switching_gamma = gamma)

    else:
        raise NotImplementedError('\n Not implemented for this type of dictionary')


    timeRes     = { 'noScreen': res_noScreen['time'],
                    'static':   res_static['time'],
                    'dynamic':  res_dyn['time'],
                    'dynamic_approx':  [x['time'] for x in res_dyn_approx_list]}
                
    nbIteration = { 'noScreen': res_noScreen['nbIter'],
                    'static':   res_static['nbIter'],
                    'dynamic':  res_dyn['nbIter'],
                    'dynamic_approx':  [x['nbIter'] for x in res_dyn_approx_list]}
                    
    switchIt    = { 'noScreen': 0,
                    'static':   0,
                    'dynamic':  0,
                    'dynamic_approx':  [x['switch_it'] for x in res_dyn_approx_list]} #still a problem since each element might have different lenght
                    
    flops       = { 'noScreen': res_noScreen['flops'],
                    'static':   res_static['flops'],
                    'dynamic':  res_dyn['flops'],
                    'dynamic_approx':  [x['flops'] for x in res_dyn_approx_list]}            
    
    Solution    = { 'noScreen': res_noScreen['sol'],
                    'static':   res_static['sol'],
                    'dynamic':  res_dyn['sol'],
                    'dynamic_approx':  [x['sol'] for x in res_dyn_approx_list]}
                   
    return timeRes, nbIteration, switchIt, Solution, flops
    #return timeRes,nbIteration,switchIt,Solution, flops,res_dyn, res_dyn_approx, res_noScreen

    
def traceProtocol(timeRes, nbIter, nbFlops, opt):
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
    
    linestyles = ['-', '--', ':', '-.']
   
    q0_d, q1_d, q2_d, q3_d, q4_d = np.percentile(\
        timeRes['dynamic']/timeRes['noScreen'],[10,25,50,75,90],axis=1)
    q0_s, q1_s, q2_s, q3_s, q4_s = np.percentile(\
        timeRes['static']/timeRes['noScreen'],[10,25,50,75,90],axis=1)
    
    q0_d1_list=[]; q1_d1_list=[]; q2_d1_list=[]; q3_d1_list=[]; q4_d1_list=[]    
    for k_approx in range(len(timeRes['dynamic_approx'])):
        q0_d1, q1_d1, q2_d1, q3_d1, q4_d1 = np.percentile(\
            timeRes['dynamic_approx'][k_approx]/timeRes['noScreen'],[10,25,50,75,90],axis=1)
        q0_d1_list.append(q0_d1); q1_d1_list.append(q1_d1); q2_d1_list.append(q2_d1); q3_d1_list.append(q3_d1); q4_d1_list.append(q4_d1) 
    
    
    flop_q0_d, flop_q1_d, flop_q2_d, flop_q3_d, flop_q4_d =\
            np.percentile(nbFlops['dynamic'].astype(float)
            /nbFlops['noScreen'],[10,25,50,75,90],axis=1)
    flop_q0_s, flop_q1_s, flop_q2_s, flop_q3_s, flop_q4_s =\
            np.percentile(nbFlops['static'].astype(float)
            /nbFlops['noScreen'],[10,25,50,75,90],axis=1)
    
    flop_q0_d1_list=[]; flop_q1_d1_list=[]; flop_q2_d1_list=[]; flop_q3_d1_list=[]; flop_q4_d1_list=[]
    for k_approx in range(len(nbFlops['dynamic_approx'])):
        flop_q0_d1, flop_q1_d1, flop_q2_d1, flop_q3_d1, flop_q4_d1 =\
                np.percentile(nbFlops['dynamic_approx'][k_approx].astype(float)
                /nbFlops['noScreen'],[10,25,50,75,90],axis=1)
        flop_q0_d1_list.append(flop_q0_d1); flop_q1_d1_list.append(flop_q1_d1); flop_q2_d1_list.append(flop_q2_d1); flop_q3_d1_list.append(flop_q3_d1); flop_q4_d1_list.append(flop_q4_d1)

    itq1_d,it_median_d ,itq3_d= np.percentile(nbIter['dynamic'] ,[25,50,75],axis=1)
    itq1_s,it_median_s ,itq3_s= np.percentile(nbIter['noScreen'] ,[25,50,75],axis=1)            
       
    f , (axTime, axFlops) = \
        plt.subplots(2,1,sharex=True,figsize=2*plt.figaspect(1.3))     

    pen_param_list = make_pen_param_list(opt['samp'],opt['min_reg'],opt['samp_type'])  
    mkevry = max(1,len(pen_param_list)/10)
    
    n_gammas = len(opt['switching_gamma'])

    if opt['Gr']:
            Gstr = 'G'
    else:
            Gstr =''
                        
    ## Time plot
    if opt['dict_type']  in {'sukro','sukro_approx','low-rank','MEG'}: 
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
#        color=iter(plt.cm.rainbow(np.linspace(0,1,len(timeRes['dynamic_approx'])))) # one color per dynamic_approx simulation
        color=iter(plt.cm.rainbow(np.linspace(0,1,len(opt['dict_params']['n_kron'])))) # one color per n_kron
        for k_approx in range(len(timeRes['dynamic_approx'])):
            if k_approx%n_gammas==0: c = next(color) #'m^-',
            linestyle = linestyles[(k_approx%n_gammas) % len(linestyles)] # one different linestyle for each gamma
            plt.plot(pen_param_list, q2_d1_list[k_approx], c=c, linestyle = linestyle,
                     label = 'A-D'+Gstr+opt['scr_type']+str(opt['dict_params']['n_kron'][k_approx/n_gammas])+r', $\Gamma$='+str(opt['switching_gamma'][k_approx%n_gammas]), #+str(k_approx), 
                     markevery= mkevry, markersize = markersize)  
            plt.fill_between(pen_param_list, q1_d1_list[k_approx], q3_d1_list[k_approx],alpha = 0.2,
                             facecolor = 'm')   

        if opt['samp_type'] is not 'linear':
            plt.xscale('log')                
        
        plt.grid(True)         
        plt.ylim((0,1.15))
        plt.title("Normalized running times") 
        plt.legend(fontsize=12,loc=3,frameon=False)
        
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
#    color=iter(plt.cm.rainbow(np.linspace(0,1,len(timeRes['dynamic_approx'])))) # one color per dynamic_approx simulation
    color=iter(plt.cm.rainbow(np.linspace(0,1,len(opt['dict_params']['n_kron'])))) # one color per n_kron
    for k_approx in range(len(nbFlops['dynamic_approx'])):
        if k_approx%n_gammas==0: c = next(color) #'m^-',
        linestyle = linestyles[(k_approx%n_gammas) % len(linestyles)] # one different linestyle for each gamma
        plt.plot(pen_param_list,flop_q2_d1_list[k_approx],c=c, linestyle=linestyle,
                     label = 'A-D'+Gstr+opt['scr_type']+' nkron='+str(opt['dict_params']['n_kron'][k_approx/n_gammas])+r', $\Gamma$='+str(opt['switching_gamma'][k_approx%n_gammas]), #+str(k_approx),
#                     label = 'A-D'+Gstr+opt['scr_type']+' nkron='+str(opt['dict_params']['n_kron'][k_approx]), #+str(k_approx),
                     markevery= mkevry, markersize = markersize)  
        plt.fill_between(pen_param_list, flop_q1_d1_list[k_approx], flop_q3_d1_list[k_approx],alpha = 0.2,
                             facecolor = 'm')

    if opt['samp_type'] is not 'linear':
        plt.xscale('log')
       
    plt.grid(True)         
    plt.ylim((0,1.15))
    plt.ylabel("Normalized flops number",fontsize = 24)
    plt.xlabel(r"$\lambda/\lambda_*$")
    plt.legend(fontsize=12,loc=3,frameon=False) #fontsize=20

    f.suptitle(type2name(opt['dict_type'],opt) + ' + ' + opt['algo_type'],fontsize=26)


    if 'ResSynthData' not in os.listdir('./'):
        os.mkdir('ResSynthData')
    f.savefig('./ResSynthData/'+make_file_name(opt)+'_Simu_relNoScreen.pdf',bbox_inches = 'tight' )
    if not opt['disp_fig']:
        plt.close()