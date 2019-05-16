# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 17:04:01 2014

@author: antoinebonnefoy

Copyright (C) 2019 Cassio Fraga Dantas

SPDX-License-Identifier: AGPL-3.0-or-later
"""
import numpy as np
import os
import matplotlib
from matplotlib import pyplot as plt 
 
from dynascreen.solve import solver
from . import generate_problem as GP
from .misc import mergeopt, make_file_name, type2name
from .misc import testopt, default_expe, make_pen_param_list

      
def first(opt =dict(), **keywords):
    '''
    Plot one execution of the dynamic screening for given options
     
    to modify default option just do first(algo_type = 'FISTA')...
    '''
    default =  dict(dict_type = 'gnoise', data_type = 'gnoise', lasso=0.75, N=1000, K=5000, 
                    stop = dict(rel_tol = 1e-8))
    expe = mergeopt(opt, default, keywords)
    expeScrRate(opt=expe)

   
def second(opt=dict(), **keywords):
    '''
    Plot the normalized time and flops for a synthetic Lasso problem
    versus the penalization parameter \lambda/\lambda_*
    you can chosse the dictionary to be gaussian noise or pnoise
    '''        
    np.random.seed(0)
    default = dict(dict_type = 'gnoise',data_type = 'gnoise', N=1000,K=5000)
    expe = mergeopt(opt, default, keywords)
    res = runLambdas(opt=expe)
    traceLambdas(res['timeRes'], res['nbIter'], res['nbFlops'] ,expe )
    
    
def third(opt=dict(), **keywords):
    '''
    Plot the normalized time and flops for a synthetic Group-Lasso problem 
    versus the penalization parameter \lambda/\lambda_*
    '''             
    default = dict(dict_type = 'pnoise',data_type = 'pnoise', N=1000,K=5000, Gr = 1, grsize = 10, sparse= 0.05)
    expe = mergeopt(default,opt, keywords)
    res = runLambdas(opt=expe)
    traceLambdas(res['timeRes'], res['nbIter'], res['nbFlops'] ,expe )
    
    

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

    
    np.random.seed(0)
    problem, opt = GP.generate(opt) # Generate the problem

        
    timeRes, nbIteration, sols, flops, dynamicRun = \
        run3versions(problem, opt, warm_start = None)   
    
    if problem.__class__.__name__ == 'GroupLasso':
        print '\n time to compute group norms: %.3f ms'%(opt['matNormTime']*1000)
    print "\ntime to compute with no screening : %.3f ms in %d iterations"%(timeRes['noScreen']*1000, nbIteration['noScreen']) 
    print "time to compute with static screening : %.3f ms in %d iterations"%(timeRes['static']*1000, nbIteration['static'])
    print "time to compute with dynamic screening : %.3f ms in %d iterations"%(timeRes['dynamic']*1000,nbIteration['dynamic'] ) 

    
    print "wrt no screen: time gain : %.3f, theoretical gain %.3f"\
        %(timeRes['noScreen']/timeRes['dynamic'],\
        float(flops['noScreen'])/flops['dynamic'])
    print "wrt static: time gain : %.3f, theoretical gain %.3f"\
        %(timeRes['static']/timeRes['dynamic'],
        float(flops['static'])/flops['dynamic'])
        
    #### saving data and plotting result  
    if 'ResSynthData' not in os.listdir('./'):
        os.mkdir('ResSynthData')
    np.savez('./ResSynthData/'+make_file_name(opt)+'_Simu_ScreenRate.npz',\
        scrRate=dynamicRun['screenrate'],radius=dynamicRun['radius'],\
        ObjValue=dynamicRun['objective'], opt=opt)

    matplotlib.rc('axes', labelsize = 'xx-large')
    matplotlib.rc('xtick', labelsize = 20)
    matplotlib.rc('ytick', labelsize = 20)
    matplotlib.rc('axes', titlesize = 'xx-large')
    matplotlib.rc('lines', linewidth = 2)
    plt.figure()
    K = problem.D.shape[1]
    length = dynamicRun['nbIter']+1
    plt.plot(np.arange(length),(1 - dynamicRun['screenrate'])*K) 
    plt.axis([0,length, 0, K*1.1])
    plt.grid(True) 
    plt.ylabel("Size of the dictionary")
    plt.xlabel("Iteration t")
    plt.savefig('./ResSynthData/Simu_screenRate_'+make_file_name(opt)+\
        '_lasso'+str(opt['lasso'])+'.pdf',bbox_inches = 'tight',bbox_pad = 2 )
    
#    plt.figure()
#    plt.plot(dynamicRun['objective'].flatten())
    return dynamicRun
    
    
    
    
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
    markersize = 12
    
   
    q0_d, q1_d, q2_d, q3_d, q4_d = np.percentile(\
        timeRes['dynamic']/timeRes['noScreen'],[10,25,50,75,90],axis=1)
    q0_s, q1_s, q2_s, q3_s, q4_s = np.percentile(\
        timeRes['static']/timeRes['noScreen'],[10,25,50,75,90],axis=1)
    
    flop_q0_d, flop_q1_d, flop_q2_d, flop_q3_d, flop_q4_d =\
            np.percentile(nbFlops['dynamic'].astype(float)
            /nbFlops['noScreen'],[10,25,50,75,90],axis=1)
    flop_q0_s, flop_q1_s, flop_q2_s, flop_q3_s, flop_q4_s =\
            np.percentile(nbFlops['static'].astype(float)
            /nbFlops['noScreen'],[10,25,50,75,90],axis=1)

    itq1_d,it_median_d ,itq3_d= np.percentile(nbIter['dynamic'] ,[25,50,75],axis=1)
    itq1_s,it_median_s ,itq3_s= np.percentile(nbIter['noScreen'] ,[25,50,75],axis=1)            
       
    f , (axTime, axFlops) = \
        plt.subplots(2,1,sharex=True,figsize=2*plt.figaspect(1.3))     

    pen_param_list = make_pen_param_list(opt['samp'])  
    mkevry = max(1,len(pen_param_list)/10)

    axTime.plot(pen_param_list,q2_d,'ks-' ,markevery= mkevry,markersize = markersize)  
    axTime.fill_between(pen_param_list, q1_d, q3_d,alpha = 0.2,facecolor = 'k')       
    axTime.fill_between(pen_param_list, q1_d, q3_d,alpha = 0.1, 
                        color = 'none',edgecolor = 'k', hatch = '/')
             
    axTime.plot(pen_param_list,  q2_s, 'bo-' ,markevery= mkevry,
                markersize = markersize) 
    axTime.fill_between(pen_param_list, q1_s, q3_s,alpha = 0.2, facecolor = 'b' )
    axTime.fill_between(pen_param_list, q1_s, q3_s,alpha = 0.1, edgecolor = 'b', 
                        color = 'none', hatch = '\\')
       
    axTime.grid(True)         
    axTime.set_ylim((-0.19,1.15))
    axTime.set_title("Normalized running times") 
    axTime.legend(fontsize=22,loc=3,frameon=False)
    
    if opt['Gr']:
            Gstr = 'G'
    else:
            Gstr =''    
    
    axFlops.plot(pen_param_list,flop_q2_d,'ks-',
                 label = opt['algo_type'] + ' + D'+Gstr+opt['scr_type'],                 
                 markevery= mkevry, markersize = markersize)  
    axFlops.fill_between(pen_param_list, flop_q1_d, flop_q3_d,alpha = 0.2,
                         facecolor = 'k')
    axFlops.fill_between(pen_param_list,flop_q1_d, flop_q3_d,alpha = 0.1,
                         edgecolor = 'k', hatch = '/',color='none')
                         
    axFlops.plot(pen_param_list,flop_q2_s,'bo-' ,
                 label = opt['algo_type'] + ' + '+Gstr+opt['scr_type'],                    
                 markevery= mkevry, markersize = markersize) 
    axFlops.fill_between(pen_param_list,flop_q1_s, flop_q3_s,alpha = 0.2,
                         facecolor = 'b')
    axFlops.fill_between(pen_param_list,flop_q1_s, flop_q3_s,alpha = 0.1, 
                         edgecolor = 'b', color = 'none', hatch = '\\')
       
    axFlops.grid(True)         
    axFlops.set_ylim((-0.19,1.15))
    axFlops.set_ylabel("Normalized flops number",fontsize = 24)
    axFlops.set_xlabel(r"$\lambda/\lambda_*$")
    axFlops.legend(fontsize=22,loc=3,frameon=False)
           
    f.suptitle(type2name(opt['dict_type']),fontsize=28)

    if 'ResSynthData' not in os.listdir('./'):
        os.mkdir('ResSynthData')
    f.savefig('./ResSynthData/'+make_file_name(opt)+'_Simu_relNoScreen.pdf' )
    if not opt['disp_fig']:
        plt.close()
        


def run3versions(problem=None, opt={}, warm_start = None, **keywords):
    """
    Run the 3 versions of the Algorithm on the same problem recording the computation times
    """    
    ####  handles options
        
    default = default_expe()
    opt = mergeopt(opt, default, keywords)
    testopt(opt)

    print r'lambda/lambda* = '+str(opt['lasso'])    

    res_dyn = solver(problem=problem, L=opt['L'], stop=opt['stop'],\
                            scr_type=opt['scr_type'], \
                            EmbedTest='dynamic', algo_type=opt['algo_type'], \
                            warm_start = warm_start) 
                            
    res_static = solver(problem=problem, L=opt['L'], stop=opt['stop'],\
                            scr_type=opt['scr_type'], \
                            EmbedTest='static', algo_type=opt['algo_type'], \
                            warm_start = warm_start)
                            
    res_noScreen = solver(problem=problem, L=opt['L'], stop=opt['stop'],\
                            scr_type=opt['scr_type'], \
                            EmbedTest='noScreen', algo_type=opt['algo_type'], \
                            warm_start = warm_start)  

    timeRes     = { 'noScreen': res_noScreen['time'],
                    'static':   res_static['time'],
                    'dynamic':  res_dyn['time']}
                
    nbIteration = { 'noScreen': res_noScreen['nbIter'],
                    'static':   res_static['nbIter'],
                    'dynamic':  res_dyn['nbIter']}
                    
    flops       = { 'noScreen': res_noScreen['flops'],
                    'static':   res_static['flops'],
                    'dynamic':  res_dyn['flops']}            
    
    Solution    = { 'noScreen': res_noScreen['sol'],
                    'static':   res_static['sol'],
                    'dynamic':  res_dyn['sol']}       
    
    return timeRes,nbIteration,Solution,flops,res_dyn

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
        
    pen_param_list = make_pen_param_list(opt['samp'])
    #### run the algorithm if needed
    if (not opt['recalc']) and (make_file_name(opt)+'_done.npz') in os.listdir('./ResSynthData'):
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
                dynamic=np.zeros((nblambdas,avg), dtype = float))
                
    nbIter = dict(noScreen=np.zeros((nblambdas,avg)),\
                static=np.zeros((nblambdas,avg)),\
                dynamic=np.zeros((nblambdas,avg)))
                
    nbFlops = dict(noScreen=np.zeros((nblambdas,avg), dtype = float),\
                static=np.zeros((nblambdas,avg), dtype = float),\
                dynamic=np.zeros((nblambdas,avg), dtype = float))                
                
    xFinal = dict(noScreen=np.zeros((nblambdas,avg,opt['K'])),\
                static=np.zeros((nblambdas,avg,opt['K'])),\
                dynamic=np.zeros((nblambdas,avg,opt['K'])))
                
    sig = np.zeros((opt['nbRuns'],opt['N']))
    

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
            timeIt, nbIt, res, flops, junk = \
                run3versions(problem,opt,start)
            for key in timeRes.iterkeys(): 
                timeRes[key][nblambdas-i-1,j] = timeIt[key]
                nbIter[key][nblambdas-i-1,j] = nbIt[key]
                nbFlops[key][nblambdas-i-1,j] = flops[key]
                if not opt['dict_type'] =="MNIST"  :
                    xFinal[key][nblambdas-i-1,j] = res[key].flatten()
        print "problem %d over %d"%(j+1,avg)           
                 
    print('Done') 


    if 'ResSynthData' not in os.listdir('./'):
        os.mkdir('ResSynthData')
    np.savez('./ResSynthData/'+make_file_name(opt)+'_done.npz',\
        timeRes=timeRes, nbIter=nbIter, opt=opt, xFinal = xFinal,\
        nbFlops=nbFlops,sig=sig)
    
    ret = { 'timeRes' : timeRes, \
            'nbIter': nbIter, \
            'nbFlops': nbFlops, \
            'opt': opt }
    return ret
    
