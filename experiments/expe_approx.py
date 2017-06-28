# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 17:04:01 2014

@author: cassiofraga
"""
import numpy as np
import os
import matplotlib
from matplotlib import pyplot as plt 
 
from dynascreen.solve import solver
from dynascreen.solve import solver_approx
from . import generate_problem as GP
from .misc import mergeopt, make_file_name, type2name
from .misc import testopt, default_expe, make_pen_param_list

from dynascreen.dictionary import Dict

      
def first(opt =dict(), **keywords):
    '''
    Plot one execution of the dynamic screening for given options
     
    to modify default option just do first(algo_type = 'FISTA')...
    '''
    lasso_list = [0.6] #[0.5, 0.75, 0.85]
    
    for lasso in lasso_list:
        default =  dict(dict_type = 'gnoise', lasso=lasso, N=100, K=500,
                        stop=dict(rel_tol=1e-8, max_iter=10000), scr_type = "ST1")
        expe = mergeopt(default, opt, keywords)
        expeScrRate(opt=expe)

   
def second(opt=dict(), **keywords):
    '''
    Plot the normalized time and flops for a synthetic Lasso problem
    versus the penalization parameter \lambda/\lambda_*
    you can chosse the dictionary to be gaussian noise or pnoise
    '''        
    np.random.seed(0)
    default = dict(dict_type = 'gnoise', N=1000,K=5000,scr_type = "ST1")
    expe = mergeopt(opt, default, keywords)
    res = runLambdas(opt=expe)
    traceLambdas(res['timeRes'], res['nbIter'], res['nbFlops'] ,expe )
    
    
def third(opt=dict(), **keywords):
    '''
    Plot the normalized time and flops for a synthetic Group-Lasso problem 
    versus the penalization parameter \lambda/\lambda_*
    '''             
    default = dict(dict_type = 'pnoise', N=1000,K=5000, Gr = 1, grsize = 10, sparse= 0.05)
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

    
    #np.random.seed(0)
    np.random.seed(10) #used for figures with y=X\beta
    
    problem, opt = GP.generate(opt) # Generate the problem
    RC = 1./2
        
    timeRes, nbIteration, sols, flops, dynamicRun, \
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
        zeros = dynamicRun['zeros'],\
        scrRate_approx=dynamicRun_approx['screenrate'],scrRate_est_approx=dynamicRun_approx['screenrate_est'],
        radius_approx=dynamicRun_approx['radius'], ObjValue_approx=dynamicRun_approx['objective'], \
        nbIter_approx = dynamicRun_approx['nbIter'], switch_it = dynamicRun_approx['switch_it'],\
        zeros_approx = dynamicRun_approx['zeros'],\
        scrRate_approx2=dynamicRun_approx2['screenrate'],scrRate_est_approx2=dynamicRun_approx2['screenrate_est'],
        radius_approx2=dynamicRun_approx2['radius'], ObjValue_approx2=dynamicRun_approx2['objective'], \
        nbIter_approx2 = dynamicRun_approx2['nbIter'], switch_it2 = dynamicRun_approx2['switch_it'],\
        zeros_approx2 = dynamicRun_approx2['zeros'],\
        scrRate_approx3=dynamicRun_approx3['screenrate'],scrRate_est_approx3=dynamicRun_approx3['screenrate_est'],
        radius_approx3=dynamicRun_approx3['radius'], ObjValue_approx3=dynamicRun_approx3['objective'],\
        nbIter_approx3 = dynamicRun_approx3['nbIter'], switch_it3 = dynamicRun_approx3['switch_it'],\
        zeros_approx3 = dynamicRun_approx3['zeros'])

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

    pen_param_list = make_pen_param_list(opt['samp'])  
    mkevry = max(1,len(pen_param_list)/10)

    ## Time plot
#    axTime.plot(pen_param_list,q2_d,'ks-' ,markevery= mkevry,markersize = markersize)  
#    axTime.fill_between(pen_param_list, q1_d, q3_d,alpha = 0.2,facecolor = 'k')       
#    axTime.fill_between(pen_param_list, q1_d, q3_d,alpha = 0.1, 
#                        color = 'none',edgecolor = 'k', hatch = '/')
#             
#    axTime.plot(pen_param_list,  q2_s, 'bo-' ,markevery= mkevry,
#                markersize = markersize) 
#    axTime.fill_between(pen_param_list, q1_s, q3_s,alpha = 0.2, facecolor = 'b' )
#    axTime.fill_between(pen_param_list, q1_s, q3_s,alpha = 0.1, edgecolor = 'b', 
#                        color = 'none', hatch = '\\')
#       
#    axTime.grid(True)         
#    axTime.set_ylim((-0.19,1.15))
#    axTime.set_title("Normalized running times") 
#    axTime.legend(fontsize=22,loc=3,frameon=False)
    
    if opt['Gr']:
            Gstr = 'G'
    else:
            Gstr =''    
    
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

       
    plt.grid(True)         
    plt.ylim((0,1.15))
    plt.ylabel("Normalized flops number",fontsize = 24)
    plt.xlabel(r"$\lambda/\lambda_*$")
    plt.legend(fontsize=20,loc=3,frameon=False)

    f.suptitle(type2name(opt['dict_type']) + ' + ' + opt['algo_type'],fontsize=26)


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

    print r'lambda/lambda* = '+str(opt['lasso']) 
    
    normE = 1e-1*np.ones(1) #*np.ones((problem.D.shape[1],1))
    problem.D_bis =  Dict(problem.D.data + normE*np.sqrt(1./opt['N'])*np.random.randn(opt['N'],opt['K']))
    problem.D_bis.normalize()
    
    #normE = 0
    res_dyn_approx = solver_approx(problem=problem, normE=normE, RC=RC, L=opt['L'], stop=opt['stop'], \
                            scr_type=opt['scr_type'], \
                            EmbedTest='dynamic', algo_type=opt['algo_type'], \
                            warm_start = warm_start)

    normE = 1e-2*np.ones(1) #*np.ones((problem.D.shape[1],1))                            
    problem.D_bis =  Dict(problem.D.data + normE*np.sqrt(1./opt['N'])*np.random.randn(opt['N'],opt['K']))
    problem.D_bis.normalize()
    
    res_dyn_approx2 = solver_approx(problem=problem, normE=normE, RC=RC, L=opt['L'], stop=opt['stop'],\
                            scr_type=opt['scr_type'], \
                            EmbedTest='dynamic', algo_type=opt['algo_type'], \
                            warm_start = warm_start) 
                            
    normE = 1e-3*np.ones(1) #*np.ones((problem.D.shape[1],1))
    problem.D_bis =  Dict(problem.D.data + normE*np.sqrt(1./opt['N'])*np.random.randn(opt['N'],opt['K']))
    problem.D_bis.normalize()
    problem.D, problem.D_bis = problem.D_bis, problem.D # The approximate dictionary is used
    
    res_dyn_approx3 = solver_approx(problem=problem, normE=normE, RC=RC, L=opt['L'], stop=opt['stop'],\
                            scr_type=opt['scr_type'], \
                            EmbedTest='dynamic', algo_type=opt['algo_type'], \
                            warm_start = warm_start)                             
                            
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
    
    return timeRes,nbIteration,Solution,flops,res_dyn, res_dyn_approx, res_dyn_approx2,res_dyn_approx3,res_noScreen

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
            timeIt, nbIt, res, flops, junk, junk1, junk2, junk3, junk_ns = \
                run3versions(problem,0.5,opt,start)
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
    