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
    
    
    startTime = time.clock()  
    
    # initialize the variables 
    N,K = problem.D.shape
    Algo.initialization( L=L, warm_start = warm_start, stop = stop)
    Screen = ScreenTest(K,scr_type)
    
    # perform the static screening
    if EmbedTest=='static':
        app, dualpt ,grad = problem.gradient(Algo.x, Screen)
        if warm_start is None:
            Screen.Initialisation(problem, grad = grad, lasso = problem.pen_param)
        else:
            Screen.Initialisation(problem,  lasso = problem.pen_param)
        
        Screen.RefineR(dualpt,grad)
        Screen.SetScreen()
        Rate = Screen.GetRate() 
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
        #####    One Iteration step    #############
        Algo.Iterate(Screen)
        
        #####    Dynamic Screening    ##############
        # dynamic screening
        if EmbedTest=='dynamic':               
            if Screen.init==0: # at the first iteration need to compute the test vector
                if warm_start == None:
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
                    
                Screen.SetScreen()    
                
            else: # compute only the new radius
                Screen.RefineR(Algo.dualpt,Algo.grad)
                
            # screen with the new test
            Screen.SetScreen()                 
            Rate = Screen.GetRate()
                                
        Algo.itCount += 1 
        Algo.StopCrit(Screen)      
        
        rayons.append(Screen.newR)
        objective.append(Algo.lastErrs[-1])
        screenrate.append(Rate)
        zeros.append( K - np.count_nonzero(Algo.x))
        dGaps.append(Algo.dgap)
        if mon: # monitoring data
            screenmon = np.append(screenmon,Screen.screen[np.newaxis].T,axis=1)
            xmon = np.append(xmon,Algo.x,axis=1)
       
    duration = time.clock() - startTime
    if not(mon):
        monvar = dict()
    else:
        monvar={'xmon':         xmon,
                'screenmon':    screenmon}
 
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
            flops = ( N*K*(RC + 1 -
                        np.asarray(zeros[0:switch_it]).mean()/K) + \
                        7*(1-np.asarray(screenrate[0:switch_it]).mean()) + 5*N + 5*nb_gr)* switch_it + \
                    ( N*K*(2 - np.asarray(screenrate[switch_it:]).mean() -
                        np.asarray(zeros[switch_it:]).mean()/K) + \
                        6*(1-np.asarray(screenrate[switch_it:]).mean()) + 5*N + 5*nb_gr)* (nbit - switch_it) + N*K
                        #6*K + 5*N + 5*nb_gr) * (nbit - switch_it)
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
    crit_RC = (1-Rate_est < RC*float(N)/(N-1))  # complexity gain of approximate dicitonary doesn't pay off
#    crit_RC = (1-Rate < RC*float(N)/(N-1))  # complexity gain of approximate dicitonary doesn't pay off
    if Rate_est == 0:
        crit_Rate = (Rate == Rate_old) and (Rate != 0) # Screening has saturated
    else:
        crit_Rate = False
    return (crit_RC or crit_Rate)

def solver_approx(y=None, D=None, RC=1, normE=0, lasso=None, Gr=[], problem=None, stop = dict(),
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
        Algo_approx = selfmod.__getattribute__(algo_type)(problem)
    
    # Set the value of L if needed
    if L == None:
        print "computation of the square norm of D... might be long"
        L = LA.norm(problem.D.data,ord=2)**2 
    
    startTime = time.clock()  
    
    # initialize the variables 
    N,K = problem.D.shape
    stop_approx = stop.copy()
    #stop_approx["rel_tol"] = 1e-8
    stop_approx["rel_tol"] = stop["rel_tol"]*1e8*(float(normE)**2)
    Algo_approx.initialization( L=L, warm_start = warm_start, stop = stop_approx)

    Screen = ScreenTestApprox(K,scr_type + "_approx") #"ST1_approx")
    
    Rate, Rate_old, Rate_est = 0, 0, 0
                
    objective = [problem.objective(Algo_approx.x, Screen)]
    rayons = [Screen.R]
    screenrate = [Screen.GetRate()]
    screenrate_est = [Screen.GetRateEst()]
    zeros = [K - np.count_nonzero(Algo_approx.x)]
    dGaps = [problem.dualGap(Algo_approx.x,Screen = Screen)]
    if mon: # monitoring data
        xmon = np.array(Algo_approx.x)
        screenmon = np.array(Screen.screen[np.newaxis].T)
        
    ## Enter the Loop of approximate problem (before switching)
    while not  switching_criterion(N,K,RC,Rate,Rate_old,Rate_est) and not Algo_approx.stopCrit: #TODO uncomment
        #####    One Iteration step    #############
        Algo_approx.Iterate(Screen)
        
        #####    Dynamic Screening    ##############
        # dynamic screening
        if EmbedTest=='dynamic':               
            if Screen.init==0: # at the first iteration need to compute the test vector
                if warm_start == None:
                    if Algo_approx.__class__.__name__ == 'Chambolle-Pock':
                        Screen.Initialisation(problem, \
                            -Algo_approx.grad*(1+Algo_approx.stepsigma)/Algo_approx.stepsigma)                    
                    else:
                        scalProd = problem.D_bis.ApplyTranspose(problem.y) # CONFIG: Using original atoms for the term |d^T c|
                        #scalProd = -Algo_approx.grad
                        
                        Screen.Initialisation(problem, scalProd, \
                                              lasso=problem.pen_param, normE = normE)
                else:
                    if Algo_approx.__class__.__name__ == 'Chambolle-Pock':
                        Screen.Initialisation(problem)                    
                    else:
                        Screen.Initialisation(problem, lasso = problem.pen_param, normE = normE)
                    
                #Screen.SetScreen()    
                
            else: # compute only the new radius
                Screen.RefineR(Algo_approx.dualpt,Algo_approx.grad)
                
            # screen with the new test
            Screen.SetScreen()
            Rate_old = Rate         # the swtiching criterion need the previous rate
            Rate = Screen.GetRate()
            Rate_est = Screen.GetRateEst()
                                
        Algo_approx.itCount += 1 
        Algo_approx.StopCrit(Screen)
        
        rayons.append(Screen.newR)
        objective.append(Algo_approx.lastErrs[-1])
        screenrate.append(Rate)
        screenrate_est.append(Rate_est)
        zeros.append( K - np.count_nonzero(Algo_approx.x))
        dGaps.append(Algo_approx.dgap)
        if mon: # monitoring data
            screenmon = np.append(screenmon,Screen.screen[np.newaxis].T,axis=1)
            xmon = np.append(xmon,Algo_approx.x,axis=1)
    
    ## Enter the Loop of original problem
    switch_it = Algo_approx.itCount
    Screen.TestType = scr_type #'ST1'
    Screen.init = 0
    Screen.normE = 0
    
    Algo_approx.stopCrit = ''
    Algo_approx.D = problem.D_bis
    Algo_approx.stopParams["rel_tol"] = stop["rel_tol"]
    problem.D, problem.D_bis = problem.D_bis, problem.D
    
    # Avoiding complexity peak at switching point
    # screen_est is used on the first iteration instead of screen
#    screenrate[-1] = Rate_est
#    rayons[-1] = Screen.newR_est
#    Screen.screen, Screen.screen_est = Screen.screen_est, Screen.screen


    while not Algo_approx.stopCrit:
        #####    One Iteration step    #############
        Algo_approx.Iterate(Screen)
        
        #####    Dynamic Screening    ##############
        # dynamic screening
        if EmbedTest=='dynamic':
            if Screen.init==0: # at the first iteration need to compute the test vector
                if warm_start == None:
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
                    
                Screen.RefineR(Algo_approx.dualpt,Algo_approx.grad)
                
            else: # compute only the new radius
                Screen.RefineR(Algo_approx.dualpt,Algo_approx.grad)               
                
            # screen with the new test
            Screen.SetScreen()                 
            Rate = Screen.GetRate()
                                
        Algo_approx.itCount += 1 
        Algo_approx.StopCrit(Screen)      
        
        rayons.append(Screen.newR)
        objective.append(Algo_approx.lastErrs[-1])
        screenrate.append(Rate)
        screenrate_est.append(Rate)
        zeros.append( K - np.count_nonzero(Algo_approx.x))
        dGaps.append(Algo_approx.dgap)
        if mon: # monitoring data
            screenmon = np.append(screenmon,Screen.screen[np.newaxis].T,axis=1)
            xmon = np.append(xmon,Algo_approx.x,axis=1)
       
    duration = time.clock() - startTime
    if not(mon):
        monvar = dict()
    else:
        monvar={'xmon':         xmon,
                'screenmon':    screenmon}
 
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
                'time':             duration,
                'nbIter':           Algo_approx.itCount,
                'flops':            flop_calc(EmbedTest,K,N,screenrate,zeros,Gr,RC,switch_it),
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
        
    def initialization(self, L=None, warm_start = None, stop = dict()): 
        
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
        self.stopCrit = ''
        self.stopParams = { 'abs_tol':      -np.inf,
                            'rel_tol':      1e-7,
                            'conv_speed':   -np.inf,
                            'max_iter':     200,
                            'dgap_tol':     -np.inf}
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
        if warm_start == None:
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
        
        
    def StopCrit(self,Screen): 
        """
        Compute the stopping criterion for the algorithm
        """
        self.lastErrs.append(self.loss+self.lasso*self.reg)
        # not able to take the difference when less than two 
        # elements have been computed        
        if self.itCount<2: 
            return 1
        if np.size(self.lastErrs) > self.prevIter:
            self.lastErrs.pop(0) 
          
        self.stopCrit =''     
        if self.stopParams['dgap_tol'] != -np.inf: 
            self.dgap = self.problem.dualGap(self.x,dualpt=self.dualpt, grad=self.grad)
            #FIXME strange the dual is sometime negative
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
            return True
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
    #TODO
    

    def __init__(self, K, Type="ST3"):
        self.TestType = Type    
        self.screen = np.ones(K,dtype=np.int)
        self.R = np.inf
        self.newR = np.inf
        self.star = -1
        self.lstar = -1
        self.init = 0
        self.dstar = [[]]
        
                
    
    def Initialisation(self,problem, grad=[], lasso=None):
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
        
        grad : nd array (optional)
            if the grad is given it is not recomputed in the initialization        
        
        """
        if lasso is None:            
            self.lasso = problem.pen_param     
        else :
            self.lasso = lasso
        self.problem = problem
        if not 'scalProd' in dir(self): 
            if grad==[]:        
                self.scalProd = problem.D.ApplyTranspose(problem.y)
            else:
                self.scalProd = grad

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
                raise ValueError("Not Valid Screening test, must be 'ST1','ST3' or 'Dome'")
                    
        self.newR = 1/self.lasso-1/self.lstar
        self.R = self.newR+1
        self.init = 1
     
     
    def RefineR(self, dualPt, grad):
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
            
        feasDual =  mu*dualPt
        rayvect = feasDual-self.c0
        self.newR = np.sqrt(rayvect.T.dot(rayvect))    
        
    def SetScreen(self):
        """
        Compute the screening vector
        """ 
        if self.newR < self.R:        
            self.R = self.newR 
            if self.TestType == "ST1":
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
    #TODO
    

    def __init__(self, K, Type="ST3"):
        self.TestType = Type    
        self.screen = np.ones(K,dtype=np.int)
        self.screen_est = np.ones(K,dtype=np.int)
        self.R = np.inf
        self.newR = np.inf
        self.star = -1
        self.lstar = -1
        self.init = 0
        self.dstar = [[]]
        
                
    
    def Initialisation(self,problem, grad=[], lasso=None, normE=0):
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
        
        grad : nd array (optional)
            if the grad is given it is not recomputed in the initialization        
        
        """
        if lasso is None:            
            self.lasso = problem.pen_param     
        else :
            self.lasso = lasso
        self.problem = problem
        if not 'scalProd' in dir(self): 
            if grad==[]:        
                self.scalProd = problem.D.ApplyTranspose(problem.y)
            else:
                self.scalProd = grad
            
        self.normE = normE


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
            elif self.TestType == "ST3" or self.TestType == "ST3_approx":
                self.dstar = problem.D.data[:,self.star:self.star+1]*np.sign(self.scalProd[self.star,0])
                self.c = self.c0 - (self.lstar/self.lasso-1)*self.dstar
                self.dist = (self.lstar/self.lasso-1)**2
                self.testvect = np.abs(problem.D.ApplyTranspose(self.c)).ravel() - 1
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
                raise ValueError("Not Valid Screening test, must be 'ST1','ST3' or 'Dome'")

        if self.R == np.inf:
            self.newR = 1/self.lasso-1/self.lstar
            self.R = self.newR+1
            # Mimicking the original screening (used by the switching criterion)
            self.newR_est = 1/self.lasso-1/self.lstar
        self.init = 1
     
     
    def RefineR(self, dualPt, grad):
        """
        Refines the radius thanks to dual scaling with the dualpoint and the grad already computed
        """
        dualN2 = dualPt.T.dot(dualPt)
        if self.problem.__class__.__name__ == "Lasso":
            if self.TestType == "ST1_approx" or self.TestType == "ST3_approx":
                gradNinf = np.max(np.abs(grad) + self.normE*np.sqrt(dualN2) )
                mu = max(-1/gradNinf,min(1/gradNinf,dualPt.T.dot(self.c0)/dualN2))
                
                # Mimicking the original screening (used by the switching criterion)
                gradNinf = np.max(np.abs(grad))
                mu_est = max(-1/gradNinf,min(1/gradNinf,dualPt.T.dot(self.c0)/dualN2))
                
                feasDual_est =  mu_est*dualPt
                rayvect_est = feasDual_est-self.c0
                self.newR_est = np.sqrt(rayvect_est.T.dot(rayvect_est)) 
            else:
                gradNinf = np.max(np.abs(grad) )
                mu = max(-1/gradNinf,min(1/gradNinf,dualPt.T.dot(self.c0)/dualN2))        
        else: 
            grnorm = fprod.BlasGrNorm(grad,self.problem.Gr,self.screen)
            groupGradNinf = np.max(grnorm/np.asarray([wg for ind,(g,wg) in enumerate(self.problem.Gr)]))
            mu = max(-1/groupGradNinf,\
                min(1/groupGradNinf,dualPt.T.dot(self.c0)/dualN2))    
            
        feasDual =  mu*dualPt
        rayvect = feasDual-self.c0
        self.newR = np.sqrt(rayvect.T.dot(rayvect))     
        
    def SetScreen(self):
        """
        Compute the screening vector
        """ 
        if self.newR < self.R:        
            self.R = self.newR 
            if self.TestType == "ST1" or self.TestType == "ST1_approx": 
                rScr = self.R 
                scrtmp = (self.testvect >= -rScr).astype(int)
            elif self.TestType == "ST3" or self.TestType == "ST3_approx":
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

        # Mimicking the original screening (used by the switching criterion)
        if self.TestType == "ST1_approx":                    
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