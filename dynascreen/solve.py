# -*- coding: utf-8 -*-
"""

Solver for the LASSO and Group-Lasso problem with 4 algorithm (ISAT,FISTA,SPARSA,Chambolle-Pock)

Alow to choose between dynamic screening,  static screening and no screening versions.

Created on Wed Oct 2 10:22:35 2013

@author: Antoine Bonnefoy
"""

import numpy as np
from numpy import linalg as LA
import time
import sys

from .problem import Lasso, GroupLasso
from . import fast_mat_prod as fprod

selfmod = sys.modules[__name__]

##############################################
###    General Solver for Lasso and Group-Lasso
##############################################

def solver(y=None, D=None, lasso=None, Gr=[], problem=None, stop = dict(),
               L='backtracking',scr_type="Dome",EmbedTest='dynamic',mon=False, 
               algo_type ='FISTA', warm_start = None, verbose=0):
    """
    This function solve the Lasso and Group-Lasso problems for a dictionary D,
    an observation y and a regularization parameter :math:`\lambda`
    thanks to iterative first order algorithm (ISTA, FISTA, SpARSA, Chambolle-Pock) 
    with 3 possible screening strategy (none/'static'/'dynamic').
    
    The problem that are solvedd are the Lasso:
    
    .. math::    
         \min_{x \in {R}^N} \dfrac{1}{2} \| Dx - y \| + \lambda \|x\|_1
                  
                  
    and the Group-Lasso :
    
    .. math::
         \min_{x \in {R}^N} \dfrac{1}{2} \| Dx - y \| + \lambda \sum_{g \in \mathcal{G}}\|x_g\|_2
         
         
            
    Parameters
    -----------
    y : 1-column array, float
        the observation if not normalized we normalize it.
         
    D : Dict from BuildDict module or ndarray
        the dictionnary (each atoms is normalized if a ndarray is given)
     
    lasso : float
        the penalization parameter :math:`\lambda`
        
    Gr : list of tuple default ([]: no grouping)
        the groups of dictionary columns :math:`\mathcal{G}`
        
    problem : Problem class instance, optional
        the problem to solve, (alternative to the 4 previous entries)
         
    stop : dict
        - 'rel_tol' for relative variation of the functional stopping criterion
        - 'max_iter' for fixing maximum iteration number 

    L : ['backtracking'],float,None
        step size strategy ( when None: is computed as  L = ||D||^2 very slow)
        
    scr_type : 'ST1', 'ST3' or ['Dome']
        choose the screening test
        
    algo_type : string
        Name of the algo among ISTA, [FISTA], SPARSA, Chambolle-Pock
        
    EmbedTest : None, 'static' or ['dynamic']
        embedded screening is performed if 'dynamic', 
        static screening is performed if 'static' and 
        no screening when None
        
    mon : boolean
        True to monitor the iterates, and screening vectors
        
    warm_start : array_like
        Specify the initial value a x
        
    
    Returns
    ---------
    res : dict
        {'sol':         sparse code of y;
        'eval':         objective value;
        'objective':    array of objective values along iterations;
        'radius':       radius values along iterations;
        'screenrate':   screening rates along iterations;
        'zeros':        numbers of zeros in the iterates alogn the iterations;
        'dGaps':        dualGaps;
        'time':         duration;
        'nbIter':       number of iteration;
        'flops':        number of flops;
        'problem':      problem solved;
        'stopCrit':     the stoping criterion used;
        'monVar':       additional monitored variables}
        
    See Also
    --------
    BuildDict, Problem, 
     
    """      
    
    
    if not np.logical_xor( problem!=None, (D!=None and y!=None and lasso!=None)):
        raise NameError(" Problem must be specified, and only once")
        
    if not problem:
        if Gr:        
            problem = GroupLasso(D, y, Gr, lasso)
        else:
            problem = Lasso(D,y,lasso)

    if algo_type in [cls.__name__ for cls in OptimAlgo.__subclasses__()]:
        Algo = selfmod.__getattribute__(algo_type)(problem)     
    
    # Set the value of L if needed
    if L == None:
        print "computation of the square norm of D... might be long"
        L = LA.norm(problem.D.data,ord=2)**2 
    
    checkpoint1, checkpoint2, checkpoint3, checkpoint4 = list(),list(),list(),list()# DEBUG TIME
    
    startTime = time.time()  
    
    # initialize the variables 
    N,K = problem.D.shape
    Algo.initialization( L=L, warm_start = warm_start, stop = stop)
    Screen = ScreenTest(K,scr_type)
    
    # perform the static screening
    if EmbedTest=='static':
        app, dualpt ,grad = problem.gradient(Algo.x, Screen)
        if warm_start is None:
            Screen.Initialisation(problem, scalProd = -grad, lasso = problem.pen_param)
        else:
            Screen.Initialisation(problem,  lasso = problem.pen_param)
        
        Screen.RefineR(dualpt,grad,Algo.x)
        Screen.SetScreen()
        Rate = Screen.GetRate() 
        del Screen.feasDual #To ensure it will be recalculated for dgap (Algo.StopCrit)
    else:
        Rate = 0
                
    objective = [problem.objective(Algo.x, Screen)]
    rayons = [Screen.R]
    screenrate = [Screen.GetRate()]
    zeros = [K - np.count_nonzero(Algo.x)]
    dGaps = [problem.dualGap(Algo.x,Screen = Screen)]
    if mon: # monitoring data
        xmon = np.array(Algo.x)
        screenmon = np.array(Screen.screen[np.newaxis].T)
        
    ## Enter the Loop
    Algo.stopCrit = ''
    while not Algo.stopCrit:
        checkpoint1.append(time.time()) # DEBUG TIME
        #####    One Iteration step    #############
        Algo.Iterate(Screen)
        checkpoint2.append(time.time()) # DEBUG TIME
        
        #####    Dynamic Screening    ##############
        # dynamic screening
        if EmbedTest=='dynamic':               
            if Screen.init==0: # at the first iteration need to compute the test vector
                if warm_start is None:
                    if Algo.__class__.__name__ == 'Chambolle-Pock':
                        Screen.Initialisation(problem, \
                            -Algo.grad*(1+Algo.stepsigma)/Algo.stepsigma)                    
                    else:
                        Screen.Initialisation(problem, -Algo.grad,
                                              lasso=problem.pen_param)
                else:
                    if Algo.__class__.__name__ == 'Chambolle-Pock':
                        Screen.Initialisation(problem)                    
                    else:
                        Screen.Initialisation(problem, lasso = problem.pen_param)
                    
            Screen.RefineR(Algo.dualpt,Algo.grad,Algo.x)
            # screen with the new test
            Screen.SetScreen()                 
            Rate = Screen.GetRate()
        
        checkpoint3.append(time.time()) # DEBUG TIME                        
        Algo.itCount += 1 
        Algo.StopCrit(Screen,EmbedTest)      
        
        rayons.append(Screen.newR)
        objective.append(Algo.lastErrs[-1])
        screenrate.append(Rate)
        zeros.append( K - np.count_nonzero(Algo.x))
        dGaps.append(Algo.dgap)
        if mon: # monitoring data
            screenmon = np.append(screenmon,Screen.screen[np.newaxis].T,axis=1)
            xmon = np.append(xmon,Algo.x,axis=1)
        checkpoint4.append(time.time()) # DEBUG TIME
       
    duration = time.time() - startTime
    time_per_it = checkpoint4 - np.append(startTime,checkpoint4[:-1])
    if not(mon):
        monvar = dict()
    else:
        monvar={'xmon':         xmon,
                'screenmon':    screenmon}
 
#    print([t2-t1 for t2,t1 in zip(checkpoint2,checkpoint1)]) # DEBUG TIME
#    print([t2-t1 for t2,t1 in zip(checkpoint3,checkpoint2)])
#    print([t2-t1 for t2,t1 in zip(checkpoint4,checkpoint3)])
#    print ""
    
    if verbose >=1:
        print "solved in %d"%(Algo.itCount)
        
    Result =   {'sol':          Algo.x,
                'eval':         objective[-1],
                'objective':    np.asarray(objective).flatten(),
                'radius':       np.asarray(rayons),
                'screenrate':   np.asarray(screenrate, dtype=float),
                'zeros':        np.asarray(zeros, dtype=float),
                'dGaps':        np.asarray(dGaps, dtype=float),
                'time':         duration,
                'time_per_it':  time_per_it,
                'nbIter':       Algo.itCount,
                'flops':        flop_calc(EmbedTest,K,N,screenrate,zeros,Gr),
                'problem':      problem,
                'stopCrit':     Algo.stopCrit,
                'monVar':       monvar}
                   
    return Result
   
def flop_calc(EmbedTest,K,N,screenrate,zeros,Gr,RC=1,switch_it=0): 
    nbit = len(zeros)-1
    nb_gr = len(Gr)
    if EmbedTest == 'dynamic':
        if RC == 1:
            flops = ( N*K*(2 - np.asarray(screenrate).mean() -
                np.asarray(zeros).mean()/K) + \
                    6*(1-np.asarray(screenrate).mean()) + 5*N + 5*nb_gr) * nbit
                    #6*K + 5*N + 5*nb_gr) * nbit
                    
        else: 
            # RC and switch_it are lists of same size
            if not isinstance(RC,list):
                RC = [RC]
            if not isinstance(switch_it,list):
                switch_it = [switch_it]
                
            # Final iterations: original dictionary
            flops = ( N*K*(2 - np.asarray(screenrate[switch_it[-1]:]).mean() -
                        np.asarray(zeros[switch_it[-1]:]).mean()/K) + \
                        6*(1-np.asarray(screenrate[switch_it[-1]:]).mean()) + 5*N + 5*nb_gr)* (nbit - switch_it[-1]) + N*K
                        #6*K + 5*N + 5*nb_gr) * (nbit - switch_it)
                        
            # Iterations with approximate dictionaries
            switch_it = [0] + switch_it
            for k in range(len(RC)):
                #N*K*(RC[k] + 1 - #TODO verify which is right
                if switch_it[k+1] > switch_it[k]:
                    flops += ( N*K*RC[k]*(1 + 1 - 
                              np.asarray(zeros[switch_it[k]:switch_it[k+1]]).mean()/K) + \
                              7*(1-np.asarray(screenrate[switch_it[k]:switch_it[k+1]]).mean()) + 5*N + 5*nb_gr)* (switch_it[k+1] - switch_it[k])                      

    elif EmbedTest == 'static':
        flops = ( N*K*(2 - np.asarray(zeros).mean()/K - 
            np.asarray(screenrate).mean() ) + 4*K + N + 3*nb_gr) \
                * nbit + N*K
    else:
        flops = ( N*K*(2- np.asarray(zeros).mean()/K) + 4*K + N + 3*nb_gr) * nbit
        
    return flops
    
    
    
##############################################
###    General Solver for Lasso and Group-Lasso
##############################################

def switching_criterion(N, K, RC=1, Rate=0, Rate_old=0, Rate_est=0):
    # RC criterion
    crit_RC = (1-Rate_est < RC*float(N)/(N-1))  # complexity gain of approximate dicitonary doesn't pay off
#    crit_RC = (1-Rate < RC*float(N)/(N-1))  # complexity gain of approximate dicitonary doesn't pay off
    
    # Screening saturation criterion
#        crit_Rate = (Rate == Rate_old) and (Rate != 0) # Screening has saturated  
    
    return crit_RC

def solver_approx(y=None, D=None, RC=1, normE=np.zeros(1), norm2E=0, lasso=None, Gr=[], problem=None, stop = dict(), switching = '',
               L='backtracking',scr_type="Dome",EmbedTest='dynamic',mon=False, 
               algo_type ='FISTA', warm_start = None, verbose=0, switching_gamma=2e-1):
    """
    This function solve the Lasso and Group-Lasso problems for a dictionary D,
    an observation y and a regularization parameter :math:`\lambda`
    thanks to iterative first order algorithm (ISTA, FISTA, SpARSA, Chambolle-Pock) 
    with 3 possible screening strategy (none/'static'/'dynamic').
    
    The problem that are solvedd are the Lasso:
    
    .. math::    
         \min_{x \in {R}^N} \dfrac{1}{2} \| Dx - y \| + \lambda \|x\|_1
                  
                  
    and the Group-Lasso :
    
    .. math::
         \min_{x \in {R}^N} \dfrac{1}{2} \| Dx - y \| + \lambda \sum_{g \in \mathcal{G}}\|x_g\|_2
         
         
            
    Parameters
    -----------
    y : 1-column array, float
        the observation if not normalized we normalize it.
         
    D : Dict from BuildDict module or ndarray
        the dictionnary (each atoms is normalized if a ndarray is given)
     
    lasso : float
        the penalization parameter :math:`\lambda`
        
    Gr : list of tuple default ([]: no grouping)
        the groups of dictionary columns :math:`\mathcal{G}`
        
    problem : Problem class instance, optional
        the problem to solve, (alternative to the 4 previous entries)
         
    stop : dict
        - 'rel_tol' for relative variation of the functional stopping criterion
        - 'max_iter' for fixing maximum iteration number 

    L : ['backtracking'],float,None
        step size strategy ( when None: is computed as  L = ||D||^2 very slow)
        
    scr_type : 'ST1', 'ST3' or ['Dome']
        choose the screening test
        
    algo_type : string
        Name of the algo among ISTA, [FISTA], SPARSA, Chambolle-Pock
        
    EmbedTest : None, 'static' or ['dynamic']
        embedded screening is performed if 'dynamic', 
        static screening is performed if 'static' and 
        no screening when None
        
    mon : boolean
        True to monitor the iterates, and screening vectors
        
    warm_start : array_like
        Specify the initial value a x

    switching_gamma : parameter that controls the convergence-based switching
        criterion. The closest to zero switching_gamma is, the longer the 
        approximate dictionary is kept.        
    
    Returns
    ---------
    res : dict
        {'sol':         sparse code of y;
        'eval':         objective value;
        'objective':    array of objective values along iterations;
        'radius':       radius values along iterations;
        'screenrate':   screening rates along iterations;
        'zeros':        numbers of zeros in the iterates alogn the iterations;
        'dGaps':        dualGaps;
        'time':         duration;
        'nbIter':       number of iteration;
        'flops':        number of flops;
        'problem':      problem solved;
        'stopCrit':     the stoping criterion used;
        'monVar':       additional monitored variables}
        
    See Also
    --------
    BuildDict, Problem, 
     
    """      
    
    
    if not np.logical_xor( problem!=None, (D!=None and y!=None and lasso!=None)):
        raise NameError(" Problem must be specified, and only once")
        
    if not problem:
        if Gr:        
            problem = GroupLasso(D, y, Gr, lasso)
        else:
            problem = Lasso(D,y,lasso)
            
    if algo_type in [cls.__name__ for cls in OptimAlgo.__subclasses__()]:
        Algo_approx = selfmod.__getattribute__(algo_type)(problem)
    
    # Set the value of L if needed
    if L == None:
        print "computation of the square norm of D... might be long"
        L = LA.norm(problem.D.data,ord=2)**2 

    # Convergence switching criterion
    stop_approx = stop.copy()
#    if switching is not 'screening_only': # No convergence criterion for switching
    if switching not in {'screening_only','off'}: # No convergence criterion for switching
        if ('dgap_tol' in  stop.keys()) or ('dgap_rel_tol'  in  stop.keys()):
            # gap ratio - switching criterion
            stop_approx["dgap_ratio"] = switching_gamma # 2e-1 gives very close results to stop_approx["dgap_rel_tol"] = 5e-3. 5e-2 seems to give better results for MEG
            # gap relative variation - switching criterion
#            stop_approx["dgap_rel_tol"] = 5e-3
            # gap absolute variation - switching criterion
#            stop_approx["dgap_tol"] = np.mean(normE)
#            stop_approx["dgap_tol"] = 1e-1*np.mean(normE); print "MODIF! stop_approx" # Never switches
        elif 'rel_tol' in  stop.keys():
            stop_approx["rel_tol"] = stop["rel_tol"]*1e8*(np.mean(normE)**2)
        elif 'conv_speed' in  stop.keys():
            stop_approx["conv_speed"] = np.mean(normE)**2 # not calibrated by experiments
        else:
            raise NotImplementedError('Convergence-based switching criterion not defined for this particular convergence criterion')

    checkpoint1, checkpoint2, checkpoint3 = list(),list(),list() # DEBUG TIME
    checkpoint4, checkpoint5, checkpoint6 = list(),list(),list() # DEBUG TIME
    checkpoint7, checkpoint8 = list(),list() # DEBUG TIME
    
    startTime = time.time()  
    
    # initialize the variables 
    N,K = problem.D.shape
    Algo_approx.initialization( L=L, warm_start = warm_start, stop = stop_approx)
         

    Screen = ScreenTestApprox(K,scr_type + "_approx") #"ST1_approx")
    
    Rate, Rate_old, Rate_est = 0, 0, 0
                
    objective = [problem.objective(Algo_approx.x, Screen)]
    rayons = [Screen.R]
    screenrate = [Screen.GetRate()]
    screenrate_est = [Screen.GetRateEst()] # Overhead
    zeros = [K - np.count_nonzero(Algo_approx.x)]
    dGaps = [problem.dualGap(Algo_approx.x,Screen = Screen)]
    dGaps_est = list(dGaps) # It contains the dgap_est, calculated with feasDual_est (unsafe!). It doesn't saturate before switching as does the real gap.
    
    if mon: # monitoring data
        xmon = np.array(Algo_approx.x)
        screenmon = np.array(Screen.screen[np.newaxis].T)
    
    ## Enter the Loop of approximate problem (before switching)
#    while not  switching_criterion(N,K,RC,Rate,Rate_old,Rate_est) and not Algo_approx.stopCrit:
    while (not  switching_criterion(N,K,RC,Rate,Rate_old,Rate_est) or switching=='off' ) and not Algo_approx.stopCrit:
        checkpoint1.append(time.time()) # DEBUG TIME
        #####    One Iteration step    #############
        Algo_approx.Iterate(Screen)
        checkpoint2.append(time.time()) # DEBUG TIME        
        
        #####    Dynamic Screening    ##############
        # dynamic screening
        if EmbedTest=='dynamic':               
            if Screen.init==0: # at the first iteration need to compute the test vector
                # Using original atoms for the term |d^T c|. Internal product needs to be computed anyway
                scalProd = problem.D_bis.ApplyTranspose(problem.y) #TODO: GAP doesn't use it. So this calculation could be avoided.
                # Using atoms from the approximate dictionary. Can be taken from the algorithm iteration in case of no warm_start
#                scalProd = None
#                if warm_start is None:
#                    if Algo_approx.__class__.__name__ == 'Chambolle-Pock':
#                        scalProd = -Algo_approx.grad*(1+Algo_approx.stepsigma)/Algo_approx.stepsigma
#                    else:
#                        scalProd = -Algo_approx.grad
                Screen.Initialisation(problem, scalProd, \
                                              lasso=problem.pen_param, normE = normE, norm2E = norm2E)

            Screen.RefineR(Algo_approx.dualpt,Algo_approx.grad,Algo_approx.x)
            # screen with the new test
            Screen.SetScreen()
            Rate_old = Rate         # the swtiching criterion need the previous rate
            Rate = Screen.GetRate()
            Rate_est = Screen.GetRateEst() # Overhead

        checkpoint3.append(time.time()) # DEBUG TIME
        Algo_approx.itCount += 1 
        Algo_approx.StopCrit(Screen,EmbedTest)
        
        rayons.append(Screen.newR)
        objective.append(Algo_approx.lastErrs[-1])
        screenrate.append(Rate)
        screenrate_est.append(Rate_est) # Overhead
        zeros.append( K - np.count_nonzero(Algo_approx.x))
        dGaps.append(Algo_approx.dgap)
        dGaps_est.append(Algo_approx.dgap_est)

        if mon: # monitoring data
            screenmon = np.append(screenmon,Screen.screen[np.newaxis].T,axis=1)
            xmon = np.append(xmon,Algo_approx.x,axis=1)
        checkpoint4.append(time.time()) # DEBUG TIME

    duration1 = time.time() - startTime #DEBUG TIME
    time_per_it = checkpoint4 - np.append(startTime,checkpoint4[:-1])
    
    ## Enter the Loop of original problem
    # Reinitialisations - Overhead
    switch_it = Algo_approx.itCount
    Screen.TestType = scr_type #'ST1'
    Screen.init = 0 #TODO is it really necessary to reinitialize
    
    Algo_approx.stopCrit = ''
    #Algo_approx.D = problem.D_bis # artigo
    if 'dgap_rel_tol' in  stop.keys():
        Algo_approx.stopParams["dgap_rel_tol"] = stop["dgap_rel_tol"]
    elif 'dgap_tol' in  stop.keys():
        Algo_approx.stopParams["dgap_tol"] = stop["dgap_tol"]
        Algo_approx.stopParams["dgap_rel_tol"] = -np.inf
    elif 'rel_tol' in  stop.keys():
        Algo_approx.stopParams["rel_tol"] = stop["rel_tol"]
    elif 'conv_speed' in  stop.keys():
        Algo_approx.stopParams["conv_speed"] = stop["conv_speed"]
    elif 'max_iter' in  stop.keys():
        Algo_approx.stopParams["max_iter"] = stop["max_iter"]
    else:
        raise NotImplementedError('Convergence-based switching criterion not defined for this particular convergence criterion')
        
    problem.D, problem.D_bis = problem.D_bis, problem.D
    #Algo_approx.D = problem.D
    
    duration2 = time.time() - startTime  - duration1 #DEBUG TIME
    # Avoiding complexity peak at switching point
    # screen_est is used on the first iteration instead of screen - Not safe!
#    screenrate[-1] = Rate_est
#    rayons[-1] = Screen.newR_est
#    Screen.screen, Screen.screen_est = Screen.screen_est, Screen.screen


    while not Algo_approx.stopCrit:
        checkpoint5.append(time.time()) # DEBUG TIME
        #####    One Iteration step    #############
        Algo_approx.Iterate(Screen)
        checkpoint6.append(time.time()) # DEBUG TIME
        
        #####    Dynamic Screening    ##############
        # dynamic screening
        if EmbedTest=='dynamic':
            if Screen.init==0: # at the first iteration need to compute the test vector
                if warm_start is None:
                    if Algo_approx.__class__.__name__ == 'Chambolle-Pock':
                        Screen.Initialisation(problem, \
                            -Algo_approx.grad*(1+Algo_approx.stepsigma)/Algo_approx.stepsigma)                    
                    else:
                        Screen.Initialisation(problem, -Algo_approx.grad,
                                              lasso=problem.pen_param)
                else:
                    if Algo_approx.__class__.__name__ == 'Chambolle-Pock':
                        Screen.Initialisation(problem)                    
                    else:
                        Screen.Initialisation(problem, lasso = problem.pen_param)

            Screen.RefineR(Algo_approx.dualpt,Algo_approx.grad,Algo_approx.x)
            # screen with the new test
            Screen.SetScreen()                 
            Rate = Screen.GetRate()

        checkpoint7.append(time.time()) # DEBUG TIME                                
        Algo_approx.itCount += 1 
        Algo_approx.StopCrit(Screen,EmbedTest)      
        
        rayons.append(Screen.newR)
        objective.append(Algo_approx.lastErrs[-1])
        screenrate.append(Rate)
        screenrate_est.append(Rate)
        zeros.append( K - np.count_nonzero(Algo_approx.x))
        dGaps.append(Algo_approx.dgap)
        dGaps_est.append(Algo_approx.dgap_est)
        
        if mon: # monitoring data
            screenmon = np.append(screenmon,Screen.screen[np.newaxis].T,axis=1)
            xmon = np.append(xmon,Algo_approx.x,axis=1)
        checkpoint8.append(time.time()) # DEBUG TIME
    
    duration3 = time.time() - startTime  - duration2 - duration1 #DEBUG TIME
    duration = time.time() - startTime
    time_per_it = np.append(time_per_it, checkpoint8 - np.append(checkpoint4[-1],checkpoint8[:-1]))

    if not(mon):
        monvar = dict()
    else:
        monvar={'xmon':         xmon,
                'screenmon':    screenmon}
 
#    print "DURATION1: approx dict loop  %.3f ms in %d iterations"%(duration1*1000, switch_it) #DEBUG TIME
#    print([t2-t1 for t2,t1 in zip(checkpoint2,checkpoint1)])
#    print([t2-t1 for t2,t1 in zip(checkpoint3,checkpoint2)])
#    print([t2-t1 for t2,t1 in zip(checkpoint4,checkpoint3)])
#    print "DURATION2  %.3f ms"%(duration2*1000) #DEBUG TIME
#    print "DURATION3  %.3f ms in %d iterations"%(duration3*1000,Algo_approx.itCount-switch_it) #DEBUG TIME
#    print([t2-t1 for t2,t1 in zip(checkpoint6,checkpoint5)])
#    print([t2-t1 for t2,t1 in zip(checkpoint7,checkpoint6)])
#    print([t2-t1 for t2,t1 in zip(checkpoint8,checkpoint7)])
#    print ""    
    
    if verbose >=1:
        print "solved in %d"%(Algo_approx.itCount)
        
    Result =   {'sol':              Algo_approx.x,
                'eval':             objective[-1],
                'objective':        np.asarray(objective).flatten(),
                'radius':           np.asarray(rayons),
                'screenrate':       np.asarray(screenrate, dtype=float),
                'screenrate_est':   np.asarray(screenrate_est, dtype=float),
                'zeros':            np.asarray(zeros, dtype=float),
                'dGaps':            np.asarray(dGaps, dtype=float),
                'dGaps_est':        np.asarray(dGaps_est, dtype=float),
                'time':             duration,
                'time1':            duration1, #DEBUG TIME
                'time2':            duration2, #DEBUG TIME
                'time3':            duration3, #DEBUG TIME
                'nbIter':           Algo_approx.itCount,
                'flops':            flop_calc(EmbedTest,K,N,screenrate,zeros,Gr,RC,switch_it),
                'time_per_it':      time_per_it,
                'problem':          problem,
                'stopCrit':         Algo_approx.stopCrit,
                'monVar':           monvar,
                'switch_it':        switch_it}
                   
    return Result    


def solver_multiple(y=None, D=None, RC=1, normE_all=np.zeros(1), norm2E_all=0, dict_specs=None, lasso=None, Gr=[], problem=None, stop = dict(), switching = '',
               L='backtracking',scr_type="Dome",EmbedTest='dynamic',mon=False,
               algo_type ='FISTA', warm_start = None, verbose=0, switching_gamma=2e-1):
    """
    This function solve the Lasso and Group-Lasso problems for a dictionary D,
    an observation y and a regularization parameter :math:`\lambda`
    thanks to iterative first order algorithm (ISTA, FISTA, SpARSA, Chambolle-Pock) 
    with 3 possible screening strategy (none/'static'/'dynamic').
    
    The problem that are solvedd are the Lasso:
    
    .. math::    
         \min_{x \in {R}^N} \dfrac{1}{2} \| Dx - y \| + \lambda \|x\|_1
                  
                  
    and the Group-Lasso :
    
    .. math::
         \min_{x \in {R}^N} \dfrac{1}{2} \| Dx - y \| + \lambda \sum_{g \in \mathcal{G}}\|x_g\|_2
         
         
            
    Parameters
    -----------
    y : 1-column array, float
        the observation if not normalized we normalize it.
         
    D : Dict from BuildDict module or ndarray
        the dictionnary (each atoms is normalized if a ndarray is given)
     
    lasso : float
        the penalization parameter :math:`\lambda`
        
    Gr : list of tuple default ([]: no grouping)
        the groups of dictionary columns :math:`\mathcal{G}`
        
    problem : Problem class instance, optional
        the problem to solve, (alternative to the 4 previous entries)
         
    stop : dict
        - 'rel_tol' for relative variation of the functional stopping criterion
        - 'max_iter' for fixing maximum iteration number 

    L : ['backtracking'],float,None
        step size strategy ( when None: is computed as  L = ||D||^2 very slow)
        
    scr_type : 'ST1', 'ST3' or ['Dome']
        choose the screening test
        
    algo_type : string
        Name of the algo among ISTA, [FISTA], SPARSA, Chambolle-Pock
        
    EmbedTest : None, 'static' or ['dynamic']
        embedded screening is performed if 'dynamic', 
        static screening is performed if 'static' and 
        no screening when None
        
    mon : boolean
        True to monitor the iterates, and screening vectors
        
    warm_start : array_like
        Specify the initial value a x
    
    switching_gamma : parameter that controls the convergence-based switching
        criterion. The closest to zero switching_gamma is, the longer the 
        approximate dictionary is kept.
    
    Returns
    ---------
    res : dict
        {'sol':         sparse code of y;
        'eval':         objective value;
        'objective':    array of objective values along iterations;
        'radius':       radius values along iterations;
        'screenrate':   screening rates along iterations;
        'zeros':        numbers of zeros in the iterates alogn the iterations;
        'dGaps':        dualGaps;
        'time':         duration;
        'nbIter':       number of iteration;
        'flops':        number of flops;
        'problem':      problem solved;
        'stopCrit':     the stoping criterion used;
        'monVar':       additional monitored variables}
        
    See Also
    --------
    BuildDict, Problem, 
     
    """      
    
    
    if not np.logical_xor( problem!=None, (D!=None and y!=None and lasso!=None)):
        raise NameError(" Problem must be specified, and only once")
        
    if not problem:
        if Gr:        
            problem = GroupLasso(D, y, Gr, lasso)
        else:
            problem = Lasso(D,y,lasso)
            
    if algo_type in [cls.__name__ for cls in OptimAlgo.__subclasses__()]:
        Algo_approx = selfmod.__getattribute__(algo_type)(problem)
    
    # Set the value of L if needed
    if L == None:
        print "computation of the square norm of D... might be long"
        L = LA.norm(problem.D.data,ord=2)**2 

    # Convergence switching criterion
    stop_approx = stop.copy()
    if switching is not 'screening_only': # No convergence criterion for switching
        if ('dgap_tol' in  stop.keys()) or ('dgap_rel_tol'  in  stop.keys()):
            # gap ratio - switching criterion
            stop_approx["dgap_ratio"] = switching_gamma # 2e-1 gives very close results to stop_approx["dgap_rel_tol"] = 5e-3. 5e-2 seems to give better results for MEG
            # gap relative variation - switching criterion
#            stop_approx["dgap_rel_tol"] = 5e-3
            # gap absolute variation - switching criterion
#            stop_approx["dgap_tol"] = np.mean(normE)
#            stop_approx["dgap_tol"] = 1e-1*np.mean(normE); print "MODIF! stop_approx" # Never switches
        else:
            #TODO - rel_tol needs to be redefin for each dict. approx.
            raise ValueError('convergence switching criterion rel_tol not implemented')
            #stop_approx["rel_tol"] = stop["rel_tol"]*1e8*(np.mean(normE)**2)

    checkpoint1, checkpoint2, checkpoint3 = list(),list(),list() # DEBUG TIME
    checkpoint4, checkpoint5, checkpoint6 = list(),list(),list() # DEBUG TIME
    checkpoint7, checkpoint8 = list(),list() # DEBUG TIME
    
    startTime = time.time()  
    
    # initialize the variables 
    N,K = problem.D.shape
    Algo_approx.initialization( L=L, warm_start = warm_start, stop = stop_approx)
         

    Screen = ScreenTestApprox(K,scr_type + "_approx") #"ST1_approx")
    
    Rate, Rate_old, Rate_est = 0, 0, 0
                
    objective = [problem.objective(Algo_approx.x, Screen)]
    rayons = [Screen.R]
    screenrate = [Screen.GetRate()]
    screenrate_est = [Screen.GetRateEst()] # Overhead
    zeros = [K - np.count_nonzero(Algo_approx.x)]
    dGaps = [problem.dualGap(Algo_approx.x,Screen = Screen)]
#    dual = [] #MODIF
#    primal = [] #MODIF
    dGaps_est = list(dGaps) # It contains the dgap_est, calculated with feasDual_est (unsafe!). It doesn't saturate before switching as does the real gap.
    
    if mon: # monitoring data
        xmon = np.array(Algo_approx.x)
        screenmon = np.array(Screen.screen[np.newaxis].T)
    
    ## Enter the Loop of approximate problem (before switching)
    switch_it = []    
    nb_approx = len(norm2E_all)
    k_approx = 0
    #while not  switching_criterion(N,K,RC[k_approx],Rate,Rate_old,Rate_est) and k_approx < nb_approx: # switches to next approx. dictionary or directly to original dictionary (if transition is due to switching level)
    while k_approx < nb_approx: # switches to next approx. dictionary or directly to original dictionary (if transition is due to switching level)
        normE = normE_all[k_approx]
        norm2E = norm2E_all[k_approx]
        # Updating current dict
        if problem.D.opType is 'sukro':       
            problem.D.nkron = dict_specs[k_approx]
        elif problem.D.opType in {'faust','low-rank'}:
            problem.D = dict_specs[k_approx]
        else:
            raise NameError('Multiple dictionaries solver not defined for this type of dictionary')
        
        while not  switching_criterion(N,K,RC[k_approx],Rate,Rate_old,Rate_est) and not Algo_approx.stopCrit:
            checkpoint1.append(time.time()) # DEBUG TIME
            #####    One Iteration step    #############
            Algo_approx.Iterate(Screen)
            checkpoint2.append(time.time()) # DEBUG TIME        
            
            #####    Dynamic Screening    ##############
            # dynamic screening
            if EmbedTest=='dynamic':               
                if Screen.init==0: # at the first iteration need to compute the test vector
                    scalProd = None
                    if not 'scalProd' in dir(Screen):
                        # Using original atoms for the term |d^T c|. Internal product needs to be computed anyway
                        scalProd = problem.D_bis.ApplyTranspose(problem.y) #TODO: GAP doesn't use it. So this calculation could be avoided.
                        # Using atoms from the approximate dictionary. Can be taken from the algorithm iteration in case of no warm_start
#                        if warm_start is None:
#                            if Algo_approx.__class__.__name__ == 'Chambolle-Pock':
#                                scalProd = -Algo_approx.grad*(1+Algo_approx.stepsigma)/Algo_approx.stepsigma
#                            else:
#                                scalProd = -Algo_approx.grad
                    Screen.Initialisation(problem, scalProd, \
                                                  lasso=problem.pen_param, normE = normE, norm2E = norm2E)
    
                Screen.RefineR(Algo_approx.dualpt,Algo_approx.grad,Algo_approx.x)
                # screen with the new test
                Screen.SetScreen()
                Rate_old = Rate         # the swtiching criterion need the previous rate
                Rate = Screen.GetRate()
                Rate_est = Screen.GetRateEst() # Overhead
    
            checkpoint3.append(time.time()) # DEBUG TIME
            Algo_approx.itCount += 1 
            Algo_approx.StopCrit(Screen,EmbedTest)
            
            rayons.append(Screen.newR)
            objective.append(Algo_approx.lastErrs[-1])
            screenrate.append(Rate)
            screenrate_est.append(Rate_est) # Overhead
            zeros.append( K - np.count_nonzero(Algo_approx.x))
            dGaps.append(Algo_approx.dgap)
#            dual.append(Algo_approx.dual) #MODIF
#            primal.append(Algo_approx.primal) #MODIF
            dGaps_est.append(Algo_approx.dgap_est)
    
            if mon: # monitoring data
                screenmon = np.append(screenmon,Screen.screen[np.newaxis].T,axis=1)
                xmon = np.append(xmon,Algo_approx.x,axis=1)
            checkpoint4.append(time.time()) # DEBUG TIME
    
        duration1 = time.time() - startTime #DEBUG TIME
        time_per_it = checkpoint4 - np.append(startTime,checkpoint4[:-1])

        ## Change dictionary
        # Reinitialisations - Overhead
        switch_it.append(Algo_approx.itCount)
        Screen.init = 0 #TODO is it really necessary to reinitialize ?
        Algo_approx.stopCrit = ''
        k_approx += 1 #index of next approximate dictionary

    
    ## Enter the Loop of original problem
    Screen.TestType = scr_type #'ST1'
    
    #Algo_approx.D = problem.D_bis # artigo
    if 'dgap_rel_tol' in  stop.keys():
        Algo_approx.stopParams["dgap_rel_tol"] = stop["dgap_rel_tol"]
        Algo_approx.stopParams["dgap_ratio"] = -np.inf
    elif 'dgap_tol' in  stop.keys():
        Algo_approx.stopParams["dgap_tol"] = stop["dgap_tol"]
        Algo_approx.stopParams["dgap_rel_tol"] = -np.inf
        Algo_approx.stopParams["dgap_ratio"] = -np.inf
    else:
        Algo_approx.stopParams["rel_tol"] = stop["rel_tol"]
        
    problem.D, problem.D_bis = problem.D_bis, problem.D
    #Algo_approx.D = problem.D
    
    duration2 = time.time() - startTime  - duration1 #DEBUG TIME
    # Avoiding complexity peak at switching point
    # screen_est is used on the first iteration instead of screen - Not safe!
#    screenrate[-1] = Rate_est
#    rayons[-1] = Screen.newR_est
#    Screen.screen, Screen.screen_est = Screen.screen_est, Screen.screen


    while not Algo_approx.stopCrit:
        checkpoint5.append(time.time()) # DEBUG TIME
        #####    One Iteration step    #############
        Algo_approx.Iterate(Screen)
        checkpoint6.append(time.time()) # DEBUG TIME
        
        #####    Dynamic Screening    ##############
        # dynamic screening
        if EmbedTest=='dynamic':
            if Screen.init==0: # at the first iteration need to compute the test vector
                if warm_start is None:
                    if Algo_approx.__class__.__name__ == 'Chambolle-Pock':
                        Screen.Initialisation(problem, \
                            -Algo_approx.grad*(1+Algo_approx.stepsigma)/Algo_approx.stepsigma)                    
                    else:
                        Screen.Initialisation(problem, -Algo_approx.grad,
                                              lasso=problem.pen_param)
                else:
                    if Algo_approx.__class__.__name__ == 'Chambolle-Pock':
                        Screen.Initialisation(problem)                    
                    else:
                        Screen.Initialisation(problem, lasso = problem.pen_param)

            Screen.RefineR(Algo_approx.dualpt,Algo_approx.grad,Algo_approx.x)
            # screen with the new test
            Screen.SetScreen()                 
            Rate = Screen.GetRate()

        checkpoint7.append(time.time()) # DEBUG TIME                                
        Algo_approx.itCount += 1 
        Algo_approx.StopCrit(Screen,EmbedTest)      
        
        rayons.append(Screen.newR)
        objective.append(Algo_approx.lastErrs[-1])
        screenrate.append(Rate)
        screenrate_est.append(Rate)
        zeros.append( K - np.count_nonzero(Algo_approx.x))
        dGaps.append(Algo_approx.dgap)
#        dual.append(Algo_approx.dual) #MODIF
#        primal.append(Algo_approx.primal) #MODIF
        dGaps_est.append(Algo_approx.dgap_est)
        
        if mon: # monitoring data
            screenmon = np.append(screenmon,Screen.screen[np.newaxis].T,axis=1)
            xmon = np.append(xmon,Algo_approx.x,axis=1)
        checkpoint8.append(time.time()) # DEBUG TIME
    
    duration3 = time.time() - startTime  - duration2 - duration1 #DEBUG TIME
    duration = time.time() - startTime
    time_per_it = np.append(time_per_it, checkpoint8 - np.append(checkpoint4[-1],checkpoint8[:-1]))

    if not(mon):
        monvar = dict()
    else:
        monvar={'xmon':         xmon,
                'screenmon':    screenmon}
 
#    print "DURATION1: approx dict loop  %.3f ms in %d iterations"%(duration1*1000, switch_it) #DEBUG TIME
#    print([t2-t1 for t2,t1 in zip(checkpoint2,checkpoint1)])
#    print([t2-t1 for t2,t1 in zip(checkpoint3,checkpoint2)])
#    print([t2-t1 for t2,t1 in zip(checkpoint4,checkpoint3)])
#    print "DURATION2  %.3f ms"%(duration2*1000) #DEBUG TIME
#    print "DURATION3  %.3f ms in %d iterations"%(duration3*1000,Algo_approx.itCount-switch_it) #DEBUG TIME
#    print([t2-t1 for t2,t1 in zip(checkpoint6,checkpoint5)])
#    print([t2-t1 for t2,t1 in zip(checkpoint7,checkpoint6)])
#    print([t2-t1 for t2,t1 in zip(checkpoint8,checkpoint7)])
#    print ""    
    
    if verbose >=1:
        print "solved in %d"%(Algo_approx.itCount)
        
    Result =   {'sol':              Algo_approx.x,
                'eval':             objective[-1],
                'objective':        np.asarray(objective).flatten(),
                'radius':           np.asarray(rayons),
                'screenrate':       np.asarray(screenrate, dtype=float),
                'screenrate_est':   np.asarray(screenrate_est, dtype=float),
                'zeros':            np.asarray(zeros, dtype=float),
                'dGaps':            np.asarray(dGaps, dtype=float),
#                'dual':             np.asarray(dual, dtype=float), #MODIF
#                'primal':           np.asarray(primal, dtype=float), #MODIF
                'dGaps_est':        np.asarray(dGaps_est, dtype=float),
                'time':             duration,
                'time1':            duration1, #DEBUG TIME
                'time2':            duration2, #DEBUG TIME
                'time3':            duration3, #DEBUG TIME
                'nbIter':           Algo_approx.itCount,
                'flops':            flop_calc(EmbedTest,K,N,screenrate,zeros,Gr,RC,switch_it),
                'time_per_it':      time_per_it,
                'problem':          problem,
                'stopCrit':         Algo_approx.stopCrit,
                'monVar':           monvar,
                'switch_it':        switch_it}
                   
    return Result    

## Calculates the conventional screening rate parallel to the stable screening
## as a comparison (if desired).
## See variable 'Screen_conv'
def solver_approx_parallel(y=None, D=None, RC=1, normE=np.zeros(1), norm2E=0, lasso=None, Gr=[], problem=None, stop = dict(), switching = '',
               L='backtracking',scr_type="Dome",EmbedTest='dynamic',mon=False, 
               algo_type ='FISTA', warm_start = None, verbose=0, switching_gamma=2e-1):
    """
    This function solve the Lasso and Group-Lasso problems for a dictionary D,
    an observation y and a regularization parameter :math:`\lambda`
    thanks to iterative first order algorithm (ISTA, FISTA, SpARSA, Chambolle-Pock) 
    with 3 possible screening strategy (none/'static'/'dynamic').
    
    The problem that are solvedd are the Lasso:
    
    .. math::    
         \min_{x \in {R}^N} \dfrac{1}{2} \| Dx - y \| + \lambda \|x\|_1
                  
                  
    and the Group-Lasso :
    
    .. math::
         \min_{x \in {R}^N} \dfrac{1}{2} \| Dx - y \| + \lambda \sum_{g \in \mathcal{G}}\|x_g\|_2
         
         
            
    Parameters
    -----------
    y : 1-column array, float
        the observation if not normalized we normalize it.
         
    D : Dict from BuildDict module or ndarray
        the dictionnary (each atoms is normalized if a ndarray is given)
     
    lasso : float
        the penalization parameter :math:`\lambda`
        
    Gr : list of tuple default ([]: no grouping)
        the groups of dictionary columns :math:`\mathcal{G}`
        
    problem : Problem class instance, optional
        the problem to solve, (alternative to the 4 previous entries)
         
    stop : dict
        - 'rel_tol' for relative variation of the functional stopping criterion
        - 'max_iter' for fixing maximum iteration number 

    L : ['backtracking'],float,None
        step size strategy ( when None: is computed as  L = ||D||^2 very slow)
        
    scr_type : 'ST1', 'ST3' or ['Dome']
        choose the screening test
        
    algo_type : string
        Name of the algo among ISTA, [FISTA], SPARSA, Chambolle-Pock
        
    EmbedTest : None, 'static' or ['dynamic']
        embedded screening is performed if 'dynamic', 
        static screening is performed if 'static' and 
        no screening when None
        
    mon : boolean
        True to monitor the iterates, and screening vectors
        
    warm_start : array_like
        Specify the initial value a x

    switching_gamma : parameter that controls the convergence-based switching
        criterion. The closest to zero switching_gamma is, the longer the 
        approximate dictionary is kept.        
    
    Returns
    ---------
    res : dict
        {'sol':         sparse code of y;
        'eval':         objective value;
        'objective':    array of objective values along iterations;
        'radius':       radius values along iterations;
        'screenrate':   screening rates along iterations;
        'zeros':        numbers of zeros in the iterates alogn the iterations;
        'dGaps':        dualGaps;
        'time':         duration;
        'nbIter':       number of iteration;
        'flops':        number of flops;
        'problem':      problem solved;
        'stopCrit':     the stoping criterion used;
        'monVar':       additional monitored variables}
        
    See Also
    --------
    BuildDict, Problem, 
     
    """      
    
    
    if not np.logical_xor( problem!=None, (D!=None and y!=None and lasso!=None)):
        raise NameError(" Problem must be specified, and only once")
        
    if not problem:
        if Gr:        
            problem = GroupLasso(D, y, Gr, lasso)
        else:
            problem = Lasso(D,y,lasso)
            
    if algo_type in [cls.__name__ for cls in OptimAlgo.__subclasses__()]:
        Algo_approx = selfmod.__getattribute__(algo_type)(problem)
    
    # Set the value of L if needed
    if L == None:
        print "computation of the square norm of D... might be long"
        L = LA.norm(problem.D.data,ord=2)**2 

    # Convergence switching criterion
    stop_approx = stop.copy()
#    if switching is not 'screening_only': # No convergence criterion for switching
    if switching not in {'screening_only','off'}: # No convergence criterion for switching
        if ('dgap_tol' in  stop.keys()) or ('dgap_rel_tol'  in  stop.keys()):
            # gap ratio - switching criterion
            stop_approx["dgap_ratio"] = switching_gamma # 2e-1 gives very close results to stop_approx["dgap_rel_tol"] = 5e-3. 5e-2 seems to give better results for MEG
            # gap relative variation - switching criterion
#            stop_approx["dgap_rel_tol"] = 5e-3
            # gap absolute variation - switching criterion
#            stop_approx["dgap_tol"] = np.mean(normE)
#            stop_approx["dgap_tol"] = 1e-1*np.mean(normE); print "MODIF! stop_approx" # Never switches
        elif 'rel_tol' in  stop.keys():
            stop_approx["rel_tol"] = stop["rel_tol"]*1e8*(np.mean(normE)**2)
        elif 'conv_speed' in  stop.keys():
            stop_approx["conv_speed"] = np.mean(normE)**2 # not calibrated by experiments
        else:
            raise NotImplementedError('Convergence-based switching criterion not defined for this particular convergence criterion')

    checkpoint1, checkpoint2, checkpoint3 = list(),list(),list() # DEBUG TIME
    checkpoint4, checkpoint5, checkpoint6 = list(),list(),list() # DEBUG TIME
    checkpoint7, checkpoint8 = list(),list() # DEBUG TIME
    
    startTime = time.time()  
    
    # initialize the variables 
    N,K = problem.D.shape
    Algo_approx.initialization( L=L, warm_start = warm_start, stop = stop_approx)
         

    Screen = ScreenTestApprox(K,scr_type + "_approx") #"ST1_approx")
    
    Rate, Rate_old, Rate_est = 0, 0, 0
                
    objective = [problem.objective(Algo_approx.x, Screen)]
    rayons = [Screen.R]
    screenrate = [Screen.GetRate()]
    screenrate_est = [Screen.GetRateEst()] # Overhead
    zeros = [K - np.count_nonzero(Algo_approx.x)]
    dGaps = [problem.dualGap(Algo_approx.x,Screen = Screen)]
    dGaps_est = list(dGaps) # It contains the dgap_est, calculated with feasDual_est (unsafe!). It doesn't saturate before switching as does the real gap.
    
    if mon: # monitoring data
        xmon = np.array(Algo_approx.x)
        screenmon = np.array(Screen.screen[np.newaxis].T)

    Screen_conv = ScreenTest(K,scr_type)
    screenrate_conv = [Screen_conv.GetRate()]
    problem_conv = Lasso(D=problem.D_bis, y=problem.y, D_bis=problem.D) # Is it really necessary
    problem_conv.pen_param = problem.pen_param
#    problem_conv.D, problem_conv.D_bis = problem_conv.D_bis, problem_conv.D # start with the approximate (fast) dictionary
    
    ## Enter the Loop of approximate problem (before switching)
#    while not  switching_criterion(N,K,RC,Rate,Rate_old,Rate_est) and not Algo_approx.stopCrit:
    while (not  switching_criterion(N,K,RC,Rate,Rate_old,Rate_est) or switching=='off' ) and not Algo_approx.stopCrit:
        checkpoint1.append(time.time()) # DEBUG TIME
        #####    One Iteration step    #############
        Algo_approx.Iterate(Screen)
        checkpoint2.append(time.time()) # DEBUG TIME        
        
        #####    Dynamic Screening    ##############
        # dynamic screening
        if EmbedTest=='dynamic':               
            if Screen.init==0: # at the first iteration need to compute the test vector
                # Using original atoms for the term |d^T c|. Internal product needs to be computed anyway
                scalProd = problem.D_bis.ApplyTranspose(problem.y) #TODO: GAP doesn't use it. So this calculation could be avoided.
                # Using atoms from the approximate dictionary. Can be taken from the algorithm iteration in case of no warm_start
#                scalProd = None
#                if warm_start is None:
#                    if Algo_approx.__class__.__name__ == 'Chambolle-Pock':
#                        scalProd = -Algo_approx.grad*(1+Algo_approx.stepsigma)/Algo_approx.stepsigma
#                    else:
#                        scalProd = -Algo_approx.grad
                Screen.Initialisation(problem, scalProd, \
                                              lasso=problem.pen_param, normE = normE, norm2E = norm2E)
                Screen_conv.Initialisation(problem_conv, problem_conv.D.ApplyTranspose(problem_conv.y), \
                                              lasso=problem_conv.pen_param)                      

            Screen.RefineR(Algo_approx.dualpt,Algo_approx.grad,Algo_approx.x)
            _, dualpt_conv, grad_conv = problem_conv.gradient(Algo_approx.xprev, Screen)
            Screen_conv.RefineR(dualpt_conv,grad_conv,Algo_approx.x)
            # screen with the new test
            Screen.SetScreen()
            Rate_old = Rate         # the swtiching criterion need the previous rate
            Rate = Screen.GetRate()
            Rate_est = Screen.GetRateEst() # Overhead
            
            Screen_conv.SetScreen()
            Rate_conv = Screen_conv.GetRate()

        checkpoint3.append(time.time()) # DEBUG TIME
        Algo_approx.itCount += 1 
        Algo_approx.StopCrit(Screen,EmbedTest)
        
        rayons.append(Screen.newR)
        objective.append(Algo_approx.lastErrs[-1])
        screenrate.append(Rate)
        screenrate_est.append(Rate_est) # Overhead
        screenrate_conv.append(Rate_conv) # Overhead
        zeros.append( K - np.count_nonzero(Algo_approx.x))
        dGaps.append(Algo_approx.dgap)
        dGaps_est.append(Algo_approx.dgap_est)

        if mon: # monitoring data
            screenmon = np.append(screenmon,Screen.screen[np.newaxis].T,axis=1)
            xmon = np.append(xmon,Algo_approx.x,axis=1)
        checkpoint4.append(time.time()) # DEBUG TIME

    duration1 = time.time() - startTime #DEBUG TIME
    time_per_it = checkpoint4 - np.append(startTime,checkpoint4[:-1])
    
    ## Enter the Loop of original problem
    # Reinitialisations - Overhead
    switch_it = Algo_approx.itCount
    Screen.TestType = scr_type #'ST1'
    Screen.init = 0 #TODO is it really necessary to reinitialize
    
    Algo_approx.stopCrit = ''
    #Algo_approx.D = problem.D_bis # artigo
    if 'dgap_rel_tol' in  stop.keys():
        Algo_approx.stopParams["dgap_rel_tol"] = stop["dgap_rel_tol"]
    elif 'dgap_tol' in  stop.keys():
        Algo_approx.stopParams["dgap_tol"] = stop["dgap_tol"]
        Algo_approx.stopParams["dgap_rel_tol"] = -np.inf
    elif 'rel_tol' in  stop.keys():
        Algo_approx.stopParams["rel_tol"] = stop["rel_tol"]
    elif 'conv_speed' in  stop.keys():
        Algo_approx.stopParams["conv_speed"] = stop["conv_speed"]
    elif 'max_iter' in  stop.keys():
        Algo_approx.stopParams["max_iter"] = stop["max_iter"]
    else:
        raise NotImplementedError('Convergence-based switching criterion not defined for this particular convergence criterion')
        
    problem.D, problem.D_bis = problem.D_bis, problem.D
    #Algo_approx.D = problem.D
    
    duration2 = time.time() - startTime  - duration1 #DEBUG TIME
    # Avoiding complexity peak at switching point
    # screen_est is used on the first iteration instead of screen - Not safe!
#    screenrate[-1] = Rate_est
#    rayons[-1] = Screen.newR_est
#    Screen.screen, Screen.screen_est = Screen.screen_est, Screen.screen


    while not Algo_approx.stopCrit:
        checkpoint5.append(time.time()) # DEBUG TIME
        #####    One Iteration step    #############
        Algo_approx.Iterate(Screen)
        checkpoint6.append(time.time()) # DEBUG TIME
        
        #####    Dynamic Screening    ##############
        # dynamic screening
        if EmbedTest=='dynamic':
            if Screen.init==0: # at the first iteration need to compute the test vector
                if warm_start is None:
                    if Algo_approx.__class__.__name__ == 'Chambolle-Pock':
                        Screen.Initialisation(problem, \
                            -Algo_approx.grad*(1+Algo_approx.stepsigma)/Algo_approx.stepsigma)                    
                    else:
                        Screen.Initialisation(problem, -Algo_approx.grad,
                                              lasso=problem.pen_param)
                else:
                    if Algo_approx.__class__.__name__ == 'Chambolle-Pock':
                        Screen.Initialisation(problem)                    
                    else:
                        Screen.Initialisation(problem, lasso = problem.pen_param)

            Screen.RefineR(Algo_approx.dualpt,Algo_approx.grad,Algo_approx.x)
            # screen with the new test
            Screen.SetScreen()                 
            Rate = Screen.GetRate()

        checkpoint7.append(time.time()) # DEBUG TIME                                
        Algo_approx.itCount += 1 
        Algo_approx.StopCrit(Screen,EmbedTest)      
        
        rayons.append(Screen.newR)
        objective.append(Algo_approx.lastErrs[-1])
        screenrate.append(Rate)
        screenrate_est.append(Rate)
        screenrate_conv.append(Rate) # Overhead
        zeros.append( K - np.count_nonzero(Algo_approx.x))
        dGaps.append(Algo_approx.dgap)
        dGaps_est.append(Algo_approx.dgap_est)
        
        if mon: # monitoring data
            screenmon = np.append(screenmon,Screen.screen[np.newaxis].T,axis=1)
            xmon = np.append(xmon,Algo_approx.x,axis=1)
        checkpoint8.append(time.time()) # DEBUG TIME
    
    duration3 = time.time() - startTime  - duration2 - duration1 #DEBUG TIME
    duration = time.time() - startTime
    time_per_it = np.append(time_per_it, checkpoint8 - np.append(checkpoint4[-1],checkpoint8[:-1]))

    if not(mon):
        monvar = dict()
    else:
        monvar={'xmon':         xmon,
                'screenmon':    screenmon}
 
#    print "DURATION1: approx dict loop  %.3f ms in %d iterations"%(duration1*1000, switch_it) #DEBUG TIME
#    print([t2-t1 for t2,t1 in zip(checkpoint2,checkpoint1)])
#    print([t2-t1 for t2,t1 in zip(checkpoint3,checkpoint2)])
#    print([t2-t1 for t2,t1 in zip(checkpoint4,checkpoint3)])
#    print "DURATION2  %.3f ms"%(duration2*1000) #DEBUG TIME
#    print "DURATION3  %.3f ms in %d iterations"%(duration3*1000,Algo_approx.itCount-switch_it) #DEBUG TIME
#    print([t2-t1 for t2,t1 in zip(checkpoint6,checkpoint5)])
#    print([t2-t1 for t2,t1 in zip(checkpoint7,checkpoint6)])
#    print([t2-t1 for t2,t1 in zip(checkpoint8,checkpoint7)])
#    print ""    
    
    if verbose >=1:
        print "solved in %d"%(Algo_approx.itCount)
        
    Result =   {'sol':              Algo_approx.x,
                'eval':             objective[-1],
                'objective':        np.asarray(objective).flatten(),
                'radius':           np.asarray(rayons),
                'screenrate':       np.asarray(screenrate, dtype=float),
                'screenrate_est':   np.asarray(screenrate_est, dtype=float),
                'screenrate_conv':  np.asarray(screenrate_conv, dtype=float),
                'zeros':            np.asarray(zeros, dtype=float),
                'dGaps':            np.asarray(dGaps, dtype=float),
                'dGaps_est':        np.asarray(dGaps_est, dtype=float),
                'time':             duration,
                'time1':            duration1, #DEBUG TIME
                'time2':            duration2, #DEBUG TIME
                'time3':            duration3, #DEBUG TIME
                'nbIter':           Algo_approx.itCount,
                'flops':            flop_calc(EmbedTest,K,N,screenrate,zeros,Gr,RC,switch_it),
                'time_per_it':      time_per_it,
                'problem':          problem,
                'stopCrit':         Algo_approx.stopCrit,
                'monVar':           monvar,
                'switch_it':        switch_it}
                   
    return Result    

####################################################
###  Iterative Optimization Algorithms
####################################################

class OptimAlgo(object): 
    """
    This class contain all the variables and method of the optimization procedure 

    The function set the problem of interest   with 
    
    .. math:: \lambda 
    
    Parameters
    -----------
    Name : string
        Define which Algo to use
        
    y : 1-column array, float
        observation
         
    D : Dict Class from BuildDict module
        dictionnary
     
    """ 
    def __init__(self,problem):
        self.problem = problem
        self.y = problem.y
        self.D = problem.D        
        self.continuation = 0
        self.Gr = problem.Gr
        self.lasso = problem.pen_param
        
    ####    The Lasso objective
    def objective(self,x,Screen):
        return self.problem.objective(x, Screen)
        
    def initialization(self, L=None, warm_start = None, stop = dict(rel_tol=1e-7,max_iter=200)): 
        
        """
        Initialized the algorithm and the variables with the corresonding value of lasso
        
        Parameters
        ----------
        lasso : float
            the hyperparameter of the LASSO problem
                
        """ 
        self.itCount =0
        N,K= self.D.shape
        self.L=L
        self.prevIter = 2
        self.lastErrs = list()     
        self.lastDgaps = list()   
        self.stopCrit = ''
        self.stopParams = { 'abs_tol':      -np.inf,
                            'rel_tol':      -np.inf, #1e-7,
                            'conv_speed':   -np.inf,
                            'max_iter':      np.inf,
                            'dgap_tol':     -np.inf,
                            'dgap_rel_tol': -np.inf,
                            'dgap_ratio':   -np.inf}
        try :
            if set(stop.keys()).issubset(set(self.stopParams.keys())):           
                self.stopParams.update(stop)
            else:
                raise ValueError('''Stop criterion is not supported\n 
                    The list is: \n\t %s'''%(self.stopParams.keys()))
        except AttributeError:
            raise ValueError('stop argument must be a dict')
        
        if L == 'backtracking' or L == None:
             self.eta = 1.2 
             self.step = .1 
        else :
            self.step = 1/L 
            

        self.dgap = np.inf
#        self.dual = -np.inf #MODIF
#        self.primal = np.inf #MODIF
        self.dgap_est = np.inf #calculated with feasDual_est
        if warm_start is None:
            self.x = np.zeros((K,1),dtype=np.float,order='F')    
        else:
            assert warm_start.shape[0] == K
            self.x = warm_start.copy()            
        self.xprev = np.ones_like(self.x) 
        self.xtmp = np.ones_like(self.x)  
        self.z = self.x.copy()
        self.grad =  np.zeros_like(self.x)  
        self.app = np.zeros((N,1),dtype = np.float, order = 'F')
        self.dualpt = np.zeros_like(self.app)
        self.reg = 0
        self.loss = 0.5*np.sqrt(self.problem.y.T.dot(self.problem.y))
        
        
    def StopCrit(self,Screen,EmbedTest="dynamic"): 
        """
        Compute the stopping criterion for the algorithm
        """
        self.stopCrit =''
        
        # Calculate Dual Gap if necessary
        if (self.stopParams['dgap_tol'] != -np.inf) or (self.stopParams['dgap_rel_tol'] != -np.inf) or (self.stopParams['dgap_ratio'] != -np.inf):
            if Screen.TestType in {"GAP","GAP_approx"} and EmbedTest == "dynamic": #Screen.dgap != self.dgap: # gap has already been calculated for the screening (GAP Safe dynamic rule)
                self.dgap = Screen.dgap
#                self.dual = Screen.dual #MODIF
#                self.primal = Screen.primal #MODIF
                if Screen.TestType == "GAP_approx": self.dgap_est = Screen.dgap_est
            else:
                if hasattr(Screen, 'feasDual'): # dynamic screening - feasible point already calculated
                    feasDual = Screen.feasDual
                else: # calculate dual feasible point
                    dualN2 = self.dualpt.T.dot(self.dualpt)
                    gradNinf = np.max(np.abs(self.grad))
                    mu = max(-1/gradNinf,min(1/gradNinf,self.dualpt.T.dot(self.y/self.lasso)/dualN2))
                    feasDual = mu*self.dualpt
                    
                self.dgap = self.problem.dualGap(self.x,dualpt=self.dualpt, grad=self.grad,  feasDual = feasDual)
#                _,self.primal,self.dual = self.problem.dualGap_all(self.x,dualpt=self.dualpt, grad=self.grad,  feasDual = feasDual) # MODIF
                # Calculating dgap_est. Computational overhead but only used (maybe) as switching criterion. Otherwise, could be removed
                if hasattr(Screen, 'feasDual_est'): self.dgap_est = self.problem.dualGap(self.x,dualpt=self.dualpt, grad=self.grad,  feasDual = Screen.feasDual_est) 

            self.lastDgaps.append(self.dgap)
            if np.size(self.lastDgaps) > self.prevIter:
                self.lastDgaps.pop(0)

            # dgap_rel_tol
            if  self.itCount>1 and \
                (max(self.lastDgaps) - min(self.lastDgaps))/(self.lastDgaps[-1] + 1e-10) < self.stopParams['dgap_rel_tol']:
                #(max(self.lastDgaps) - min(self.lastDgaps)) < self.stopParams['dgap_rel_tol']: #TODO see if it's better to divide or not
                self.stopCrit += 'Dgap_Tol'+ str(self.stopParams['dgap_rel_tol'])
            
            # dgap_ratio
            if  self.itCount>1 and self.dgap_est/self.dgap < self.stopParams['dgap_ratio']:
                self.stopCrit += 'Dgap_Ratio'+ str(self.stopParams['dgap_ratio'])


        self.lastErrs.append(self.loss+self.lasso*self.reg)
        # not able to take the difference when less than two 
        # elements have been computed        
        if self.itCount<2: 
            return 1
        if np.size(self.lastErrs) > self.prevIter:
            self.lastErrs.pop(0) 

        if abs(self.dgap) < self.stopParams['dgap_tol']:
            self.stopCrit += 'dual_gap' + str(self.stopParams['dgap_tol'])

        if self.lastErrs[-1] < self.stopParams['abs_tol']:
            self.stopCrit += 'Abs_Tol'+ str(self.stopParams['abs_tol'])

        if (max(self.lastErrs) - min(self.lastErrs))/(self.lastErrs[-1] + 1e-10) < self.stopParams['rel_tol']:
            self.stopCrit += 'Rel_Tol'+str(self.stopParams['rel_tol'])

        if self.itCount >= self.stopParams['max_iter']:
            self.stopCrit += 'Max_Iter' + str(self.stopParams['max_iter'])
            
        if max(self.lastErrs) - min(self.lastErrs) < self.stopParams['conv_speed']:
            self.stopCrit += 'Conv_Speed' + str(self.stopParams['conv_speed'])

        

        return 
        
        

    def Iterate(self, Screen, lasso):  
        """
        Perform One step of an optimization algorithm
        """
        return

    def SetStep(self, Screen):   
        """ 
        This method set the step size
        """
        if self.L != 'backtracking':
            loss = self.problem.loss(self.xtmp,Screen) # MODIF: fixed step-size wasn't working before, since loss and reg were never calculated
            reg = self.problem.reg(self.xtmp, Screen)
            return True, loss, reg
        ### the usual Backtracking strategy
        else:
            loss = self.problem.loss(self.xtmp,Screen)
            reg = self.problem.reg(self.xtmp, Screen)
            difftmp = self.xtmp-self.z
            if (loss + self.lasso * reg > \
                1./2*(self.dualpt.T.dot(self.dualpt)) +\
                (difftmp).T.dot(self.grad)  + \
                1./(2*self.step) *(difftmp.T.dot(difftmp)) +\
                self.lasso*reg):

                self.step = 1./self.eta * self.step 
                return False,loss,reg
            else:
                return True,loss,reg
                   
        
        
        
        
class ISTA(OptimAlgo):
    def Iterate(self, Screen):
        self.xprev = self.x.copy() # needed for the stopping criterium
        self.app, self.dualpt, self.grad = self.problem.gradient(self.x, Screen)
        while True:
            self.xtmp = \
                self.problem.prox(Screen.screen[:,np.newaxis]*self.x - self.step*self.grad, self.step*self.lasso, Screen)  
            test, loss, reg = self.SetStep(Screen)
            if test:
                self.x = self.xtmp
                self.loss = loss
                self.reg = reg
                self.z = self.x # do not remove, self.z is used in SetStep
                break
        

        
class FISTA(OptimAlgo):
    def initialization(self, L=None, warm_start = None, stop = dict()):
        super(FISTA, self).initialization( L=L, warm_start = warm_start, stop=stop)
        self.t = 1

    def Iterate(self, Screen):
        self.xprev = self.x.copy() # needed for the stopping criterium
        self.app, self.dualpt, self.grad = self.problem.gradient(self.z, Screen)
        while True:                
            self.xtmp = self.problem.prox(Screen.screen[:,np.newaxis]*self.z - \
                        self.step*self.grad,self.step*self.lasso, Screen)
            test, self.loss, self.reg = self.SetStep(Screen)
            if test:
                self.x =self.xtmp
                t0 = self.t
                self.t = (1+np.sqrt(1+4*t0**2))/2
                self.z = self.x + (t0-1)/self.t*(self.x-self.xprev)   
                break  
            

        
class TWIST(OptimAlgo):
    """
    TWIST instance of the optimAlgo
    """
    def initialization(self, L=None, warm_start = None, stop = dict()):
        super(TWIST, self).initialization(L=L, warm_start = warm_start, stop=stop)
        # eq 20 in Bioucas Figureido TwIST
        lam1=1e-4
        lamN=1
        rho = (1-lam1/lamN)/(1+lam1/lamN) 
        self.alfa = 2/(1+np.sqrt(1-rho**2))
        # eq 27
        # epsilon = 1e-1 ou 1e-3 pour le peu problem midly ar severely illconditionned 
        self.beta = 2*self.alfa / (lam1 + lamN)
        self.xprevprev = self.xprev.copy()
        
    
    def Iterate(self, Screen):        
        self.xprevprev = self.xprev.copy()
        self.xprev = self.x.copy() # needed for the stopping criterium
        self.app, self.dualpt, self.grad = self.problem.gradient(self.x, Screen)
        self.x = ((1-self.alfa)*Screen.screen[:,np.newaxis]*self.xprevprev + \
            (self.alfa - self.beta)*Screen.screen[:,np.newaxis]*self.x + \
            self.beta*self.problem.prox( self.x-self.grad, self.lasso, Screen))
        self.z = self.x
        test, self.loss, self.reg = self.SetStep(Screen)
        
        
        
class SPARSA(OptimAlgo):
    def initialization(self, L=None, warm_start = None, stop = dict()):
        super(SPARSA, self).initialization(L=L, warm_start = warm_start, stop=stop)
        self.eta = 10 
        self.step = 0. 
        self.MonotonityLevel = 5
        self.continuationParam = 0.5
        self.grad = self.D.ApplyTranspose(self.y)
        self.lasso = max(self.continuationParam*np.max(np.abs(self.grad)),\
                    self.problem.pen_param)
        self.diffIterate = np.zeros_like(self.x)
        self.stepmin = 1e-8
        self.stepmax = 1e3
        self.tau = 0.1
        self.continuationStop = 1e-3
        
    def Iterate(self, Screen):
        self.xprev = self.x.copy() # needed for the stopping criterium
        self.app, self.dualpt, self.grad = self.problem.gradient(self.x, Screen)
        tmp = np.sqrt(self.diffIterate.T.dot(self.diffIterate))            
        if tmp <=1e-15:
            self.step = 1.
        else:
            apptmp = self.D.ApplyScreen(self.diffIterate,Screen.screen)
            self.step = np.sqrt(apptmp.T.dot(apptmp))/tmp
            self.step = max(min(self.step,self.stepmax),self.stepmin)
        while True:
            self.xtmp = self.problem.prox(Screen.screen[:,np.newaxis]*self.x-\
                self.step*self.grad,self.step*self.lasso,Screen)
            test, loss, reg = self.SetStep(Screen)
            if test:
                self.x = self.xtmp
                self.loss = loss
                self.reg = reg
                self.z = self.x # do not remove, self.z is used in SetStep
                self.diffIterate = self.xtmp-self.x    
                break     
        
    def Setstep(self, Screen):
        # the Brazilai-Borwein strategy
        difftmp = self.xtmp-self.z            
        loss = self.problem.loss(self.xtmp,Screen)
        reg = self.problem.reg(self.xtmp, Screen) 
        if (loss + self.lasso * reg> \
                       np.max(self.lastErrs[-self.MonotonityLevel:]) -\
                       self.tau/(2*self.step)*difftmp.T.dot(difftmp)):
            self.step = 1./self.eta * self.step
            self.step = max(min(self.step,self.stepmax),self.stepmin)
            return False, loss,reg
        else:
            return True, loss,reg
            
    def StopCrit(self,Screen): 
        """
        Compute the stopping criterion for the algorithm
        
        For SPARSA only the rel_tol and max_iter have been implemented according to the paper considerations
        """
        self.lastErrs.append(self.loss+self.lasso*self.reg)
        # not able to take the difference when less than two 
        # elements have been computed        
        if len(self.lastErrs)<2: 
            return 1
        if np.size(self.lastErrs) > self.prevIter:
            self.lastErrs.pop(0) 
          
        self.stopCrit =''     
        if self.lasso == self.problem.pen_param:
            if (max(self.lastErrs) - min(self.lastErrs))/(self.lastErrs[-1] + 1e-10) <\
                self.stopParams['rel_tol']:
                self.stopCrit += 'Rel_Tol'+str(self.stopParams['rel_tol'])
    
            if self.itCount >= self.stopParams['max_iter']:
                self.stopCrit += 'Max_Iter' + str(self.stopParams['max_iter'])
        else:
            if (max(self.lastErrs) - min(self.lastErrs))/(self.lastErrs[-1] + 1e-10) <\
                        self.continuationStop: 
                self.lasso = max(self.continuationParam* min(self.lasso,np.max(np.abs(self.grad))),\
                        self.problem.pen_param)
#                Screen.init =0
                tmp = self.lastErrs.pop()
                self.lastErrs = list()
                self.lastErrs.append(tmp) 

            if self.itCount >= self.stopParams['max_iter']-50:
                self.lasso = self.problem.pen_param
#                Screen.init = 0
                tmp = self.lastErrs.pop()
                self.lastErrs = list()
                self.lastErrs.append(tmp) 
                


        return 
        
      
class Chambolle_Pock(OptimAlgo):
    """
    The Chambolle Pock algorithm
    """        
    
    def initialization(self, L=None, warm_start = None, stop = dict()):
        super(Chambolle_Pock, self).initialization(L=L, warm_start = warm_start, stop=stop)
        
        L = np.linalg.norm(self.D.data[:,:10],ord=2)**2 * self.D.shape[1]/10

        self.stepdual = 10
        self.stepprimal = 0.9/(self.stepdual*(L))
        self.gamma = 0.1
        self.phi = 0.5
        self.xhat =  np.zeros_like(self.x) 
        self.xprev =  np.zeros_like(self.x) 
        self.dualprev = np.zeros_like(self.dualpt)
        
        
        
    def Iterate(self, Screen):
        self.dualprev = self.dualpt
        self.app = self.D.ApplyScreen(self.xhat,Screen.screen)   
        self.dualpt = 1./(1+self.stepdual)*(self.dualpt + self.stepdual*(self.app-self.y))
        self.grad = self.D.ApplyTransposeScreen(self.dualpt,Screen.screen)
        self.x = self.problem.prox(self.x - self.stepprimal*self.grad,\
                self.stepprimal*self.lasso, Screen)
        self.phi = 1./np.sqrt(1+2*self.gamma*self.stepprimal)  
        self.stepprimal = self.phi*self.stepprimal
        self.stepdual = self.stepdual/self.phi
        self.xhat = self.x + self.phi*(self.x-self.xprev)  
        
        self.loss = self.problem.loss(self.xhat,Screen)
        self.reg = self.problem.reg(self.xhat,Screen)
        
        
#########################        
###    Screening tests
#########################        

class ScreenTest:
    """
    The class managing the screening tests
    
    Method : create the instance with given size and type
    Parameters
    ----------
    K : int
        size of the vector to screen
    Type : string
        Name of the test ('ST1, 'ST3' or 'Dome')
    """
    

    def __init__(self, K, Type="ST3"):
        self.TestType = Type    
        self.screen = np.ones(K,dtype=np.int)
        self.R = np.inf
        self.newR = np.inf
        self.dgap = np.inf
#        self.dual = - np.inf #MODIF
#        self.primal = np.inf #MODIF
        self.star = -1
        self.lstar = -1
        self.init = 0
        self.dstar = [[]]
        
                
    
    def Initialisation(self,problem, scalProd=[], lasso=None):
        """
        Initialize all fields of the instance acoording to the given parameters
        
        Parameters
        ----------
        D : Dict
            the dictionary
            
        y : 1d array
            the observation to approximate through D
            
        lasso : double
            Lasso hyperparameter, controling sparsity of the solution
        
        scalProd : nd array (optional)
            if the (D^T y) is given it is not recomputed in the initialization
        
        """
        if lasso is None:            
            self.lasso = problem.pen_param     
        else :
            self.lasso = lasso
        self.problem = problem
        if not 'scalProd' in dir(self): 
            if scalProd==[]:
                self.scalProd = problem.D.ApplyTranspose(problem.y)
            else:
                self.scalProd = scalProd

        if problem.__class__.__name__ == 'Lasso':    
            
            self.star = np.argmax(np.abs(self.scalProd))
            self.lstar = np.abs(self.scalProd[self.star,0])
            self.c0 = problem.y/self.lasso
            if self.TestType == "ST1":
                self.c = self.c0     
                self.testvect = 1/self.lasso*np.abs(self.scalProd).ravel() - 1
            elif self.TestType == "ST3":
                self.dstar = problem.D.data[:,self.star:self.star+1]*np.sign(self.scalProd[self.star,0])
                self.c = self.c0 - (self.lstar/self.lasso-1)*self.dstar
                self.dist = (self.lstar/self.lasso-1)**2
                self.testvect = np.abs(problem.D.ApplyTranspose(self.c)).ravel() - 1
            elif self.TestType == "GAP":
                self.testvect = - 1
            elif self.TestType == "Dome":
                self.dstar = problem.D.data[:,self.star:self.star+1]*\
                        np.sign(self.scalProd[self.star,0])
                self.dist = (self.lstar/self.lasso-1)**2
                self.Dtdstar = problem.D.ApplyTranspose(self.dstar).ravel()
                self.testvect = self.scalProd.ravel()
            else:
                raise ValueError("Not Valid Screening test, must be 'ST1','ST3' or 'Dome'")
                
            if "matrix" in self.testvect.__class__.__name__:
                self.testvect = np.array(self.testvect).ravel() 
                
                
        elif problem.__class__.__name__ ==  'GroupLasso':
            self.c0 = problem.y/self.lasso
            self.star , self.lstar = problem.getStar()
            if self.TestType == "ST1":
                Dtc = self.scalProd/self.lasso    
                self.testvect = np.zeros(len(problem.Gr)) 
                grnorm = fprod.BlasGrNorm(Dtc,problem.Gr,\
                    np.ones(problem.D.shape[1], dtype=int))
                self.testvect = grnorm/problem.grMatNorm - \
                    np.asarray([wg for ind,(g,wg) in\
                    enumerate(problem.Gr)])/problem.grMatNorm    
            
            elif self.TestType == "ST3":
                self.wgstar = problem.Gr[self.star][1]
                Dgstar = problem.D.data[:,problem.Gr[self.star][0]]
                self.n = Dgstar.dot(Dgstar.T.dot(problem.y))/self.lstar
                normn = np.sqrt(self.n.T.dot(self.n))
                proj = (self.n.T.dot(self.c0) - 1.* self.wgstar**2) * self.n / normn**2
                self.c = self.c0 - proj
                self.dist = proj.T.dot(proj)
                Dtc = problem.D.ApplyTranspose(self.c)
                self.testvect = np.zeros(len(problem.Gr)) 
                grnorm = fprod.BlasGrNorm(Dtc,problem.Gr, \
                    np.ones(problem.D.shape[1], dtype=int))
                self.testvect = grnorm/problem.grMatNorm - \
                    np.asarray([wg for ind,(g,wg) in enumerate(problem.Gr)])\
                    /problem.grMatNorm
            elif self.TestType == "Dome":
                raise ValueError(" The Dome test does not exist for Group-Lasso")
            else:
                raise ValueError("Not Valid Screening test, must be 'ST1','ST3','GAP' or 'Dome'")
                    
        self.newR = 1/self.lasso-1/self.lstar
        self.R = self.newR+1
        self.init = 1
     
     
    def RefineR(self, dualPt, grad, x):
        """
        Refines the radius thanks to dual scaling with the dualpoint and the grad already computed
        """
        dualN2 = dualPt.T.dot(dualPt)
        if self.problem.__class__.__name__ == "Lasso":
            gradNinf = np.max(np.abs(grad))
            mu = max(-1/gradNinf,min(1/gradNinf,dualPt.T.dot(self.c0)/dualN2))        
        else: 
            grnorm = fprod.BlasGrNorm(grad,self.problem.Gr,self.screen)
            groupGradNinf = np.max(grnorm/np.asarray([wg for ind,(g,wg) in enumerate(self.problem.Gr)]))
            mu = max(-1/groupGradNinf,\
                min(1/groupGradNinf,dualPt.T.dot(self.c0)/dualN2))    
            
        self.feasDual =  mu*dualPt
        if self.TestType == "GAP":
            self.dgap = self.problem.dualGap(x,dualpt=dualPt, grad= grad, feasDual = self.feasDual)
#            _,self.primal,self.dual = self.problem.dualGap_all(x,dualpt=dualPt, grad= grad, feasDual = self.feasDual) #MODIF
            self.newR = np.sqrt(2*(self.dgap))/self.lasso
            self.testvect = - 1 -mu*np.abs(grad) # Addind |x^T c| which changes since c = \theta            
        else:
            rayvect = self.feasDual-self.c0
            self.newR = np.sqrt(rayvect.T.dot(rayvect))    
        
    def SetScreen(self):
        """
        Compute the screening vector
        """ 
        if self.newR < self.R:        
            self.R = self.newR 
            if self.TestType == "ST1" or self.TestType == "GAP":
                rScr = self.R 
                scrtmp = (self.testvect >= -rScr).astype(int)
            elif self.TestType == "ST3":
                rScr = (np.sqrt(max(1e-20,self.R**2-self.dist)))
                scrtmp = (self.testvect >= -rScr).astype(int)
            elif self.TestType == "Dome":
                rScr = (np.sqrt(max(1e-10,self.R**2-self.dist)))
                scrtmp = np.bitwise_or(Ql(self.Dtdstar, self.lstar, self.lasso,rScr)>= self.testvect ,\
                Qu(self.Dtdstar ,self.lstar,self.lasso,rScr)<= self.testvect ).astype(int)
                # pas sur que ca marche
            if np.isnan(rScr):
                print('Probleme avec le rafinement du rayon on veut la racine de %lf'%(self.R**2-self.dist))
            
            if self.problem.__class__.__name__ == "Lasso":
                self.screen = scrtmp
            else:
                self.screen = np.repeat(scrtmp,[len(g) for ind, (g,wg) in enumerate(self.problem.Gr)])
                 
            if self.screen.ndim != 1:
                self.screen = self.screen.flatten()
            
    def GetRate(self):
        scrRate = 1 - float(self.screen.sum())/self.screen.shape[0]
        return(scrRate)
        

class ScreenTestApprox:
    """
    The class managing the screening tests
    
    Method : create the instance with given size and type
    Parameters
    ----------
    K : int
        size of the vector to screen
    Type : string
        Name of the test ('ST1, 'ST3' or 'Dome')
    """

    def __init__(self, K, Type="ST3"):
        self.TestType = Type    
        self.screen = np.ones(K,dtype=np.int)
        self.screen_est = np.ones(K,dtype=np.int)
        self.R = np.inf
        self.newR = np.inf
        self.dgap = np.inf
#        self.dual = - np.inf #MODIF
#        self.primal = np.inf #MODIF
        self.dgap_est = np.inf
        self.star = -1
        self.lstar = -1
        self.init = 0
        self.dstar = [[]]
        
                
    
    def Initialisation(self,problem, scalProd=[], lasso=None, normE=np.zeros(1), norm2E=0):
        """
        Initialize all fields of the instance acoording to the given parameters
        
        Parameters
        ----------
        D : Dict
            the dictionary
            
        y : 1d array
            the observation to approximate through D
            
        lasso : double
            Lasso hyperparameter, controling sparsity of the solution
        
        scalProd : nd array (optional)
            if (D^T y) is given it is not recomputed in the initialization
        
        """
        if lasso is None:            
            self.lasso = problem.pen_param     
        else :
            self.lasso = lasso
        self.problem = problem
        if not 'scalProd' in dir(self):
            #if self.TestType in {"GAP","GAP_approx"}: #TODO the product (X^T y) is not necessary
            #    self.scalProd = np.zeros(1)
            if scalProd==[]:
                self.scalProd = problem.D.ApplyTranspose(problem.y)
            else:
                self.scalProd = scalProd
            
        self.normE = normE.ravel()
        #self.normE = normE
        self.norm2E2 = norm2E**2
        self.normE_21 = np.max(self.normE)**2 #TEST_OPNORM Testing bounds using different operator norm


        if problem.__class__.__name__ == 'Lasso':    
            
            self.star = np.argmax(np.abs(self.scalProd))
            self.lstar = np.abs(self.scalProd[self.star,0])
            self.c0 = problem.y/self.lasso
            self.margin = 0
            if self.TestType == "ST1":
                self.c = self.c0     
                self.testvect = 1/self.lasso*np.abs(self.scalProd).ravel() - 1
            elif self.TestType == "ST1_approx":
                #self.margin = 1/self.lasso*normE.ravel()*np.sqrt(problem.y.T.dot(problem.y)) # CONFIG: when the original atoms, no margin is needed
                self.c = self.c0     
                self.testvect = 1/self.lasso*np.abs(self.scalProd).ravel() + self.margin - 1
            elif self.TestType in {"ST3","ST3_approx"}:
                self.dstar = problem.D.data[:,self.star:self.star+1]*np.sign(self.scalProd[self.star,0])
                self.c = self.c0 - (self.lstar/self.lasso-1)*self.dstar
                self.dist = (self.lstar/self.lasso-1)**2
                self.testvect = np.abs(problem.D.ApplyTranspose(self.c)).ravel() - 1
            elif self.TestType in {"GAP","GAP_approx"}:                
                self.margin = (1/self.lasso*normE*np.sqrt(problem.y.T.dot(problem.y))).ravel() # It is required here, since |x^T c| uses the approximate atoms. Differently from ST1_approx
                #self.margin = 1/self.lasso*normE*np.sqrt(problem.y.T.dot(problem.y)) # It is required here, since |x^T c| uses the approximate atoms. Differently from ST1_approx
                self.testvect = self.margin - 1
            elif self.TestType == "Dome":
                self.dstar = problem.D.data[:,self.star:self.star+1]*\
                        np.sign(self.scalProd[self.star,0])
                self.dist = (self.lstar/self.lasso-1)**2
                self.Dtdstar = problem.D.ApplyTranspose(self.dstar).ravel()
                self.testvect = self.scalProd.ravel()
            else:
                raise ValueError("Not Valid Screening test, must be 'ST1','ST3', 'GAP' or 'Dome'")
                
            if "matrix" in self.testvect.__class__.__name__:
                self.testvect = np.array(self.testvect).ravel() 
                
                
        elif problem.__class__.__name__ ==  'GroupLasso':
            self.c0 = problem.y/self.lasso
            self.star , self.lstar = problem.getStar()
            if self.TestType == "ST1":
                Dtc = self.scalProd/self.lasso    
                self.testvect = np.zeros(len(problem.Gr)) 
                grnorm = fprod.BlasGrNorm(Dtc,problem.Gr,\
                    np.ones(problem.D.shape[1], dtype=int))
                self.testvect = grnorm/problem.grMatNorm - \
                    np.asarray([wg for ind,(g,wg) in\
                    enumerate(problem.Gr)])/problem.grMatNorm    
            
            elif self.TestType == "ST3":
                self.wgstar = problem.Gr[self.star][1]
                Dgstar = problem.D.data[:,problem.Gr[self.star][0]]
                self.n = Dgstar.dot(Dgstar.T.dot(problem.y))/self.lstar
                normn = np.sqrt(self.n.T.dot(self.n))
                proj = (self.n.T.dot(self.c0) - 1.* self.wgstar**2) * self.n / normn**2
                self.c = self.c0 - proj
                self.dist = proj.T.dot(proj)
                Dtc = problem.D.ApplyTranspose(self.c)
                self.testvect = np.zeros(len(problem.Gr)) 
                grnorm = fprod.BlasGrNorm(Dtc,problem.Gr, \
                    np.ones(problem.D.shape[1], dtype=int))
                self.testvect = grnorm/problem.grMatNorm - \
                    np.asarray([wg for ind,(g,wg) in enumerate(problem.Gr)])\
                    /problem.grMatNorm
            elif self.TestType == "Dome":
                raise ValueError(" The Dome test does not exist for Group-Lasso")
            else:
                raise ValueError("Not Valid Screening test, must be 'ST1','ST3' or 'Dome'")

        if self.R == np.inf:
            self.newR = 1/self.lasso-1/self.lstar
            self.R = self.newR+1
            # Mimicking the original screening (used by the switching criterion)
            self.newR_est = 1/self.lasso-1/self.lstar
        self.init = 1
     
     
    def RefineR(self, dualPt, grad, x, Algo=[]):
        """
        Refines the radius thanks to dual scaling with the dualpoint and the grad already computed
        """
        dualN2 = dualPt.T.dot(dualPt)
        # Calculation of mu (feasibility multiplication coefficient)
        if self.problem.__class__.__name__ == "Lasso":
            if self.TestType in {"ST1_approx","ST3_approx","GAP_approx"}:
                gradNinf = np.max(np.abs(grad).ravel() + self.normE*np.sqrt(dualN2) ) #grad is (K,1) and normE is (K,)
                #gradNinf = np.max(np.abs(grad) + self.normE*np.sqrt(dualN2) ) #grad is (K,1) and normE is (K,1)
                mu = max(-1/gradNinf,min(1/gradNinf,dualPt.T.dot(self.c0)/dualN2))
                
                # Mimicking the original screening (used by the switching criterion)
                gradNinf = np.max(np.abs(grad))
                mu_est = max(-1/gradNinf,min(1/gradNinf,dualPt.T.dot(self.c0)/dualN2))
                
                self.feasDual_est =  mu_est*dualPt
            else:
                gradNinf = np.max(np.abs(grad) )
                mu = max(-1/gradNinf,min(1/gradNinf,dualPt.T.dot(self.c0)/dualN2))
        else: 
            grnorm = fprod.BlasGrNorm(grad,self.problem.Gr,self.screen)
            groupGradNinf = np.max(grnorm/np.asarray([wg for ind,(g,wg) in enumerate(self.problem.Gr)]))
            mu = max(-1/groupGradNinf,\
                min(1/groupGradNinf,dualPt.T.dot(self.c0)/dualN2))    

        # Dual feasible point
        self.feasDual =  mu*dualPt
        
        # Radius calculation
        if self.TestType in {"GAP","GAP_approx"}:
            self.dgap = self.problem.dualGap(x,dualpt=dualPt, grad= grad, feasDual = self.feasDual)
#            _,self.primal,self.dual = self.problem.dualGap_all(x,dualpt=dualPt, grad= grad, feasDual = self.feasDual) # MODIF
            normE2x2 = self.norm2E2*x.T.dot(x)
            #normE2x2 = self.normE_21*(np.abs(x).sum())**2 #TEST_OPNORM Testing bounds using different operator norm 
            margin_dgap = 0.5*normE2x2 + np.sqrt(normE2x2*dualN2) #*(1 + self.lasso*self.normy/(2*dualN2)) dual margin - not necessary
            self.newR = np.sqrt(2*(self.dgap + margin_dgap))/self.lasso
            self.testvect = self.margin - 1 -mu*np.abs(grad).ravel() # Addind |x^T c| which changes since c = \theta            
            #self.testvect = self.margin - 1 -mu*np.abs(grad) # Addind |x^T c| which changes since c = \theta
            
            # Unsafe radius - for switching criterion
            if self.TestType == "GAP_approx":
                self.dgap_est = self.problem.dualGap(x,dualpt=dualPt, grad= grad, feasDual = self.feasDual_est)
                self.newR_est = np.sqrt(2*self.dgap_est)/self.lasso
                self.testvect_est = self.margin - 1 -mu_est*np.abs(grad).ravel()
                #self.testvect_est = self.margin - 1 -mu_est*np.abs(grad)
        else: #ST1, ST3
            rayvect = self.feasDual-self.c0
            self.newR = np.sqrt(rayvect.T.dot(rayvect))
            # Unsafe radius - for switching criterion
            if self.TestType in {"ST1_approx","ST3_approx"}:
                rayvect_est = self.feasDual_est-self.c0
                self.newR_est = np.sqrt(rayvect_est.T.dot(rayvect_est)) 
        
    def SetScreen(self):
        """
        Compute the screening vector
        """ 
        if self.newR < self.R:        
            self.R = self.newR 
            if self.TestType in {"ST1","ST1_approx","GAP","GAP_approx"}: 
                rScr = self.R 
                scrtmp = (self.testvect >= -rScr).astype(int)
            elif self.TestType in {"ST3","ST3_approx"}:
                rScr = (np.sqrt(max(1e-20,self.R**2-self.dist)))
                scrtmp = (self.testvect >= -rScr).astype(int)
            elif self.TestType == "Dome":
                rScr = (np.sqrt(max(1e-10,self.R**2-self.dist)))
                scrtmp = np.bitwise_or(Ql(self.Dtdstar, self.lstar, self.lasso,rScr)>= self.testvect ,\
                Qu(self.Dtdstar ,self.lstar,self.lasso,rScr)<= self.testvect ).astype(int)
                # pas sur que ca marche
            if np.isnan(rScr):
                print('Probleme avec le rafinement du rayon on veut la racine de %lf'%(self.R**2-self.dist))
            
            if self.problem.__class__.__name__ == "Lasso":
                self.screen = scrtmp
            else:
                self.screen = np.repeat(scrtmp,[len(g) for ind, (g,wg) in enumerate(self.problem.Gr)])
                 
            if self.screen.ndim != 1:
                self.screen = self.screen.flatten()

        # Mimicking the original screening (used by the switching criterion) - Overhead
        if self.TestType in {"ST1_approx","GAP_approx"}:                    
            screen_est = (self.testvect - self.margin >= -self.newR_est).astype(int)
            self.screen_est = screen_est.flatten() & self.screen_est
        elif self.TestType == "ST3_approx":
            rScr = (np.sqrt(max(1e-20,self.newR_est**2-self.dist)))
            screen_est = (self.testvect - self.margin >= -rScr).astype(int)
            self.screen_est = screen_est.flatten() & self.screen_est
            
            
    def GetRate(self):
        scrRate = 1 - float(self.screen.sum())/self.screen.shape[0]
        return(scrRate)
        
    def GetRateEst(self):
        scrRate = 1 - float(self.screen_est.sum())/self.screen_est.shape[0]
        return(scrRate)
        
### Auxiliary functions for the Dome Test
def Ql(x,lstar,lasso,r):
    return  np.where(x<=lstar, (lstar-lasso)*x - lasso + lasso*r*np.sqrt(np.where(1-x**2<1e-10,1e-10,1-x**2)), -(lasso - 1 + lasso/lstar))

def Qu(x,lstar,lasso,r):
    return  np.where(x>=-lstar, (lstar-lasso)*x + lasso - lasso*r*np.sqrt(np.where(1-x**2<1e-10,1e-10,1-x**2)), (lasso - 1 + lasso/lstar))      
