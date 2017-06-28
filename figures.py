# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 11:51:44 2017

@author: cfragada
"""
import numpy as np
import matplotlib
import os
from matplotlib import pyplot as plt 

import experiments
from experiments.expe_approx import flop_calc_it
from experiments.misc import make_file_name, type2name, make_pen_param_list

#from .misc import mergeopt, make_file_name, type2name
#from .misc import testopt, default_expe, make_pen_param_list

# Plot properties
matplotlib.rc('axes', labelsize = 24)
matplotlib.rc('xtick', labelsize = 20)
matplotlib.rc('ytick', labelsize = 20)
matplotlib.rc('axes', titlesize = 24)
matplotlib.rc('lines', linewidth = 3)

matplotlib.rc('mathtext', fontset='cm')

############## FIGURE -1 ###############
# Screen and Flops. noMargin: ON, NO SWITCHING
# N=100, K=500
# lambda = 0.75*lambda_max

# NEW FORMATTING! WITH SCIENTIFIC NOTATION AND TOTAL COMPLEXITY AXIS.

# Load data Figure 1
filename = './ResSynthData/figures/-1-ISTA_gnoise_N100_K500_ST1_regpath10_lambda_0.75.npz'
Data = np.load(filename)
# Load data Figure 2
#filename = './ResSynthData/figures/2-ISTA_gnoise_N100_K500_ST1_regpath10_lambda_0.75.npz'
#Data = np.load(filename)

#plt.figure()
f , (axScreen) = \
    plt.subplots(1,1,sharex=True,figsize=1.27*plt.figaspect(0.55))

opt = Data['opt'][()]
K = opt['K']
N = opt['N']
length = Data['nbIter']+1
length_approx = Data['nbIter_approx']+1
length_approx2 = Data['nbIter_approx2']+1
length_approx3 = Data['nbIter_approx3']+1
markers_on1 = [Data['switch_it'] -1] #, length_approx-1]
markers_on2 = [Data['switch_it2']-1] #, length_approx2-1]
markers_on3 = [Data['switch_it3']-1] #, length_approx3-1]
markersize = 12

flops_d = flop_calc_it("dynamic",K,N,Data['scrRate'], Data['zeros'],[])
flops_d1 = flop_calc_it("dynamic",K,N,Data['scrRate_approx'], Data['zeros_approx'],[],Data['RC'],Data['switch_it'])
flops_d2 = flop_calc_it("dynamic",K,N,Data['scrRate_approx2'], Data['zeros_approx2'],[],Data['RC'],Data['switch_it2'])
flops_d3 = flop_calc_it("dynamic",K,N,Data['scrRate_approx3'], Data['zeros_approx3'],[],Data['RC'],Data['switch_it3'])
       
axScreen.plot(np.arange(length),(1 - Data['scrRate'])*K, '-k', linewidth = 6,\
                label = 'D'+opt['scr_type']) #, 'x', markevery=[length-1]
#axScreen.plot(np.arange(length_approx),(1 - Data['scrRate_approx'])*K, '-mx', markevery=markers_on1,markersize = markersize,\
#                label = 'A-D'+opt['scr_type']+r'$: \| \mathbf{e}_j \|\!=\!10^{-1}$') # Marker on the swithing point 
axScreen.plot(np.arange(length_approx2),(1 - Data['scrRate_approx2'])*K, '-g', markevery=markers_on2,markersize = markersize,\
                label = 'A-D'+opt['scr_type']+r'$: \| \mathbf{e}_j \|\!=\!10^{-2}$') 
axScreen.plot(np.arange(length_approx3),(1 - Data['scrRate_approx3'])*K, '-r', markevery=markers_on3,markersize = markersize,\
                label = 'A-D'+opt['scr_type']+r'$: \| \mathbf{e}_j \|\!=\!10^{-3}$')
#axScreen.plot(np.arange(length_approx2),(1 - Data['scrRate_est_approx2'])*K, '-gx', markevery=markers_on2)

axScreen.axis([0,50,0,K*1.1]) #0.5*max(length,length_approx,length_approx2,length_approx3), 0, K*1.1])
#axScreen.grid(True) 
axScreen.set_ylabel("Number of active atoms")
axScreen.set_xlabel("Iteration t")
axScreen.legend(fontsize=18,loc=1,frameon=False)
axScreen.ticklabel_format(style='sci', axis='y', scilimits=(0,0)) #ScientifiC Notation axis 'y'

f.savefig('./ResSynthData/-1-Simu_screenRate_'+make_file_name(opt)+\
    '_lasso'+str(opt['lasso'])+'approx.eps',bbox_inches = 'tight',bbox_pad = 2 )
    
f.savefig('./ResSynthData/-1-Simu_screenRate_'+make_file_name(opt)+\
    '_lasso'+str(opt['lasso'])+'approx.pdf',bbox_inches = 'tight',bbox_pad = 2 )

############## FIGURE 0 ###############
# Screen and Flops. noMargin: OFF, Screen_est: OFF, noPeak:OFF
# Intermediate lambda

# Load data
filename = './ResSynthData/figures/0-ISTA_gnoise_N100_K500_ST1_regpath10_lambda_0.75.npz'
Data = np.load(filename)

#plt.figure()
f , (axScreen, axFlops_it) = \
    plt.subplots(2,1,sharex=True,figsize=2*plt.figaspect(1.3))    
opt = Data['opt'][()]
K = opt['K']
N = opt['N']
length = Data['nbIter']+1
length_approx = Data['nbIter_approx']+1
length_approx2 = Data['nbIter_approx2']+1
length_approx3 = Data['nbIter_approx3']+1
markers_on1 = [Data['switch_it'] -1] #, length_approx-1]
markers_on2 = [Data['switch_it2']-1] #, length_approx2-1]
markers_on3 = [Data['switch_it3']-1] #, length_approx3-1]

flops_d = flop_calc_it("dynamic",K,N,Data['scrRate'], Data['zeros'],[])
flops_d1 = flop_calc_it("dynamic",K,N,Data['scrRate_approx'], Data['zeros_approx'],[],Data['RC'],Data['switch_it'])
flops_d2 = flop_calc_it("dynamic",K,N,Data['scrRate_approx2'], Data['zeros_approx2'],[],Data['RC'],Data['switch_it2'])
flops_d3 = flop_calc_it("dynamic",K,N,Data['scrRate_approx3'], Data['zeros_approx3'],[],Data['RC'],Data['switch_it3'])
       
axScreen.plot(np.arange(length),(1 - Data['scrRate'])*K, '-k', linewidth = 6,\
                label = 'D'+opt['scr_type']) #, 'x', markevery=[length-1]
axScreen.plot(np.arange(length_approx),(1 - Data['scrRate_approx'])*K, '-mx', markevery=markers_on1,\
                label = 'A-D'+opt['scr_type']+r'$: \| \mathbf{e}_j \|\!=\!10^{-1}$') # Marker on the swithing point 
axScreen.plot(np.arange(length_approx2),(1 - Data['scrRate_approx2'])*K, '-gx', markevery=markers_on2,\
                label = 'A-D'+opt['scr_type']+r'$: \| \mathbf{e}_j \|\!=\!10^{-2}$') 
axScreen.plot(np.arange(length_approx3),(1 - Data['scrRate_approx3'])*K, '-rx', markevery=markers_on3,\
                label = 'A-D'+opt['scr_type']+r'$: \| \mathbf{e}_j \|\!=\!10^{-3}$')
#axScreen.plot(np.arange(length_approx2),(1 - Data['scrRate_est_approx2'])*K, '-gx', markevery=markers_on2)

axScreen.axis([0,max(length,length_approx,length_approx2,length_approx3), 0, K*1.1])
axScreen.grid(True) 
axScreen.set_ylabel("Size of the dictionary")
axScreen.set_xlabel("Iteration t")
#axScreen.legend(fontsize=18,loc=1,frameon=False)


#axFlops_it.plot(np.arange(length),flops_ns,'-b')
axFlops_it.plot(np.arange(length),fimport numpy as np
import matplotlib
import os
from matplotlib import pyplot as plt 

import experiments
from experiments.expe_approx import flop_calc_it
from experiments.misc import make_file_name, type2name, make_pen_param_list

#from .misc import mergeopt, make_file_name, type2name
#from .misc import testopt, default_expe, make_pen_param_list

# Plot properties
matplotlib.rc('axes', labelsize = 24)
matplotlib.rc('xtick', labelsize = 20)
matplotlib.rc('ytick', labelsize = 20)
matplotlib.rc('axes', titlesize = 24)
matplotlib.rc('lines', linewidth = 3)

matplotlib.rc('mathtext', fontset='cm')lops_d, '-k', linewidth = 6,\
                label = 'D'+opt['scr_type']) #markevery=[length-1])
axFlops_it.plot(np.arange(length_approx),flops_d1, '-mx', markevery=markers_on1,\
                label = 'A-D'+opt['scr_type']+r'$: \| \mathbf{e}_j \|\!=\!10^{-1}$')
axFlops_it.plot(np.arange(length_approx2),flops_d2, '-gx', markevery=markers_on2,\
                label = 'A-D'+opt['scr_type']+r'$: \| \mathbf{e}_j \|\!=\!10^{-2}$') 
axFlops_it.plot(np.arange(length_approx3),flops_d3, '-rx', markevery=markers_on3,\
                label = 'A-D'+opt['scr_type']+r'$: \| \mathbf{e}_j \|\!=\!10^{-3}$')
axFlops_it.grid(True)         
#axFlops_it.set_ylim((-0.19,1.15))
axFlops_it.set_ylim((0,1.1*max(max(flops_d),max(flops_d1),max(flops_d2),max(flops_d3))))
axFlops_it.set_xlim((0,50))
axFlops_it.set_ylabel("Flops number") #,fontsize = 24)
axFlops_it.set_xlabel("Iteration t")
axFlops_it.legend(fontsize=18,loc=1,frameon=False)

f.savefig('./ResSynthData/0-Simu_screenRate_'+make_file_name(opt)+\
    '_lasso'+str(opt['lasso'])+'approx.eps',bbox_inches = 'tight',bbox_pad = 2 )


############## FIGURE 1 ###############
# Screen and Flops. noMargin: ON, Screen_est: OFF, noPeak:OFF
# Intermediate lambda

# NEW FORMATTING! WITH SCIENTIFIC NOTATION AND TOTAL COMPLEXITY AXIS.

# Load data Figure 1
#filename = './ResSynthData/figures/1-ISTA_gnoise_N100_K500_ST1_regpath10_lambda_0.75.npz'
# Load data Figure 2
#filename = './ResSynthData/figures/2-ISTA_gnoise_N100_K500_ST1_regpath10_lambda_0.75.npz'

# Load data Figure 1 - beta bernoulli p=0.02
filename = './ResSynthData/figures/1-beta02_ISTA_gnoise_N100_K500_ST1_regpath10_lambda_0.6.npz'
# Load data Figure 2 - beta bernoulli p=0.02
#filename = './ResSynthData/figures/2-beta02_ISTA_gnoise_N100_K500_ST1_regpath10_lambda_0.6.npz'

Data = np.load(filename)

#plt.figure()
f , (axScreen, axFlops_it) = \
    plt.subplots(2,1,sharex=True,figsize=2*plt.figaspect(1.3))
axFlops_tot = axFlops_it.twinx()
opt = Data['opt'][()]
K = opt['K']
N = opt['N']
length = Data['nbIter']+1
length_approx = Data['nbIter_approx']+1
length_approx2 = Data['nbIter_approx2']+1
length_approx3 = Data['nbIter_approx3']+1
markers_on1 = [Data['switch_it'] -1] #, length_approx-1]
markers_on2 = [Data['switch_it2']-1] #, length_approx2-1]
markers_on3 = [Data['switch_it3']-1] #, length_approx3-1]
markersize = 12

flops_d = flop_calc_it("dynamic",K,N,Data['scrRate'], Data['zeros'],[])
flops_d1 = flop_calc_it("dynamic",K,N,Data['scrRate_approx'], Data['zeros_approx'],[],Data['RC'],Data['switch_it'])
flops_d2 = flop_calc_it("dynamic",K,N,Data['scrRate_approx2'], Data['zeros_approx2'],[],Data['RC'],Data['switch_it2'])
flops_d3 = flop_calc_it("dynamic",K,N,Data['scrRate_approx3'], Data['zeros_approx3'],[],Data['RC'],Data['switch_it3'])
       
axScreen.plot(np.arange(length),(1 - Data['scrRate'])*K, '-k', linewidth = 6,\
                label = 'D'+opt['scr_type']) #, 'x', markevery=[length-1]
axScreen.plot(np.arange(length_approx),(1 - Data['scrRate_approx'])*K, '-mx', markevery=markers_on1,markersize = markersize,\
                label = 'A-D'+opt['scr_type']+r'$: \sigma\!=\!10^{-1}$') # Marker on the swithing point 
axScreen.plot(np.arange(length_approx2),(1 - Data['scrRate_approx2'])*K, '-gx', markevery=markers_on2,markersize = markersize,\
                label = 'A-D'+opt['scr_type']+r'$: \sigma\!=\!10^{-2}$') 
axScreen.plot(np.arange(length_approx3),(1 - Data['scrRate_approx3'])*K, '-rx', markevery=markers_on3,markersize = markersize,\
                label = 'A-D'+opt['scr_type']+r'$: \sigma\!=\!10^{-3}$')
#axScreen.plot(np.arange(length_approx2),(1 - Data['scrRate_est_approx2'])*K, '-gx', markevery=markers_on2)

axScreen.axis([0,max(length,length_approx,length_approx2,length_approx3), 0, K*1.1])
#axScreen.grid(True) 
axScreen.set_ylabel("Number of active atoms")
axScreen.set_xlabel("Iteration t")
axScreen.legend(fontsize=18,loc='center right',frameon=False)
axScreen.ticklabel_format(style='sci', axis='y', scilimits=(0,0)) #ScientifiC Notation axis 'y'

#axFlops_it.plot(np.arange(length),flops_ns,'-b')
axFlops_it.plot(np.arange(length),flops_d, '-k', linewidth = 6,\
                label = 'D'+opt['scr_type']) #markevery=[length-1])
axFlops_it.plot(np.arange(length_approx),flops_d1, '-mx', markevery=markers_on1,markersize = markersize,\
                label = 'A-D'+opt['scr_type']+r'$: \| \mathbf{e}_j \|\!=\!10^{-1}$')
axFlops_it.plot(np.arange(length_approx2),flops_d2, '-gx', markevery=markers_on2,markersize = markersize,\
                label = 'A-D'+opt['scr_type']+r'$: \| \mathbf{e}_j \|\!=\!10^{-2}$') 
axFlops_it.plot(np.arange(length_approx3),flops_d3, '-rx', markevery=markers_on3,markersize = markersize,\
                label = 'A-D'+opt['scr_type']+r'$: \| \mathbf{e}_j \|\!=\!10^{-3}$')
#axFlops_it.grid(True)         
#axFlops_it.set_ylim((-0.19,1.15))
axFlops_it.set_ylim((0,1.1*max(max(flops_d),max(flops_d1),max(flops_d2),max(flops_d3))))
xmax = 50
axFlops_it.set_xlim((0,xmax))
axFlops_it.set_ylabel("Flops number") #,fontsize = 24)
axFlops_it.set_xlabel("Iteration t")
#axFlops_it.legend(fontsize=18,loc=1,frameon=False)
axFlops_it.ticklabel_format(style='sci', axis='y', scilimits=(0,0)) #ScientifiC Notation axis 'y'

axFlops_tot.ticklabel_format(style='sci', axis='y', scilimits=(0,0)) #ScientifiC Notation axis 'y'
axFlops_tot.plot(np.arange(length),np.cumsum(flops_d), '--k',linewidth = 0.5,dashes=(3, 5))
axFlops_tot.plot(np.arange(length_approx),np.cumsum(flops_d1), '--m',linewidth = 0.5,dashes=(3, 5))
axFlops_tot.plot(np.arange(length_approx2),np.cumsum(flops_d2), '--g',linewidth = 0.5,dashes=(3, 5)) 
axFlops_tot.plot(np.arange(length_approx3),np.cumsum(flops_d3), '--r',linewidth = 0.5,dashes=(3, 5))
flops_tot_max = np.sum(flops_d[:xmax+1])
axFlops_tot.set_ylim((0,1.1*flops_tot_max))
#axFlops_tot.set_ylim((0,flops_tot_max)) #used for beta p=0.02
#axFlops_tot.set_ylabel("Cumulative flops") #,fontsize = 24)


f.savefig('./ResSynthData/1-Simu_screenRate_'+make_file_name(opt)+\
    '_lasso'+str(opt['lasso'])+'approx.eps',bbox_inches = 'tight',bbox_pad = 2 )
    
f.savefig('./ResSynthData/1-Simu_screenRate_'+make_file_name(opt)+\
    '_lasso'+str(opt['lasso'])+'approx.pdf',bbox_inches = 'tight',bbox_pad = 2 )

############## FIGURE 2 ###############
# Screen and Flops. noMargin: ON, Screen_est: ON, noPeak:OFF
# Intermediate lambda

# Load data
filename = './ResSynthData/figures/2-ISTA_gnoise_N100_K500_ST1_regpath10_lambda_0.75.npz'
Data = np.load(filename)

f , (axScreen, axFlops_it) = \
    plt.subplots(2,1,sharex=True,figsize=2*plt.figaspect(1.3))
axFlops_tot = axFlops_it.twinx()    
opt = Data['opt'][()]
K = opt['K']
N = opt['N']
length = Data['nbIter']+1
length_approx = Data['nbIter_approx']+1
length_approx2 = Data['nbIter_approx2']+1
length_approx3 = Data['nbIter_approx3']+1
markers_on1 = [Data['switch_it'] -1] #, length_approx-1]
markers_on2 = [Data['switch_it2']-1] #, length_approx2-1]
markers_on3 = [Data['switch_it3']-1] #, length_approx3-1]

flops_d = flop_calc_it("dynamic",K,N,Data['scrRate'], Data['zeros'],[])
flops_d1 = flop_calc_it("dynamic",K,N,Data['scrRate_approx'], Data['zeros_approx'],[],Data['RC'],Data['switch_it'])
flops_d2 = flop_calc_it("dynamic",K,N,Data['scrRate_approx2'], Data['zeros_approx2'],[],Data['RC'],Data['switch_it2'])
flops_d3 = flop_calc_it("dynamic",K,N,Data['scrRate_approx3'], Data['zeros_approx3'],[],Data['RC'],Data['switch_it3'])
       
axScreen.plot(np.arange(length),(1 - Data['scrRate'])*K, '-k', linewidth = 6,\
                label = 'D'+opt['scr_type']) #, 'x', markevery=[length-1]
axScreen.plot(np.arange(length_approx),(1 - Data['scrRate_approx'])*K, '-mx', markevery=markers_on1,\
                label = 'A-D'+opt['scr_type']+r'$: \| \mathbf{e}_j \|\!=\!10^{-1}$') # Marker on the swithing point 
axScreen.plot(np.arange(length_approx2),(1 - Data['scrRate_approx2'])*K, '-gx', markevery=markers_on2,\
                label = 'A-D'+opt['scr_type']+r'$: \| \mathbf{e}_j \|\!=\!10^{-2}$') 
axScreen.plot(np.arange(length_approx3),(1 - Data['scrRate_approx3'])*K, '-rx', markevery=markers_on3,\
                label = 'A-D'+opt['scr_type']+r'$: \| \mathbf{e}_j \|\!=\!10^{-3}$')
#axScreen.plot(np.arange(length_approx2),(1 - Data['scrRate_est_approx2'])*K, '-gx', markevery=markers_on2)

axScreen.axis([0,max(length,length_approx,length_approx2,length_approx3), 0, K*1.1])
axScreen.grid(True) 
axScreen.set_ylabel("Size of the dictionary")
axScreen.set_xlabel("Iteration t")
#axScreen.legend(fontsize=18,loc='center right',frameon=False)
axScreen.ticklabel_format(style='sci', axis='y', scilimits=(0,0)) #ScientifiC Notation axis 'y'

#axFlops_it.plot(np.arange(length),flops_ns,'-b')
axFlops_it.plot(np.arange(length),flops_d, '-k', linewidth = 6,\
                label = 'D'+opt['scr_type']) #markevery=[length-1])
axFlops_it.plot(np.arange(length_approx),flops_d1, '-mx', markevery=markers_on1,\
                label = 'A-D'+opt['scr_type']+r'$: \| \mathbf{e}_j \|\!=\!10^{-1}$')
axFlops_it.plot(np.arange(length_approx2),flops_d2, '-gx', markevery=markers_on2,\
                label = 'A-D'+opt['scr_type']+r'$: \| \mathbf{e}_j \|\!=\!10^{-2}$') 
axFlops_it.plot(np.arange(length_approx3),flops_d3, '-rx', markevery=markers_on3,\
                label = 'A-D'+opt['scr_type']+r'$: \| \mathbf{e}_j \|\!=\!10^{-3}$')
axFlops_it.grid(True)         
#axFlops_it.set_ylim((-0.19,1.15))
axFlops_it.set_ylim((0,1.1*max(max(flops_d),max(flops_d1),max(flops_d2),max(flops_d3))))
xmax = 50
axFlops_it.set_xlim((0,xmax))
axFlops_it.set_ylabel("Flops number") #,fontsize = 24)
axFlops_it.set_xlabel("Iteration t")
axFlops_it.legend(fontsize=18,loc=1,frameon=False)
axFlops_it.ticklabel_format(style='sci', axis='y', scilimits=(0,0)) #ScientifiC Notation axis 'y'

f.savefig('./ResSynthData/2-Simu_screenRate_'+make_file_name(opt)+\
    '_lasso'+str(opt['lasso'])+'approx.eps',bbox_inches = 'tight',bbox_pad = 2 )
    
############## FIGURE 3 ###############
# Screen and Flops. noMargin: ON, Screen_est: ON, noPeak:ON
# Intermediate lambda
    
# Load data
filename = './ResSynthData/figures/3-ISTA_gnoise_N100_K500_ST1_regpath10_lambda_0.75.npz'
Data = np.load(filename)

f , (axScreen, axFlops_it) = \
    plt.subplots(2,1,sharex=True,figsize=2*plt.figaspect(1.3))    
opt = Data['opt'][()]
K = opt['K']
N = opt['N']
length = Data['nbIter']+1
length_approx = Data['nbIter_approx']+1
length_approx2 = Data['nbIter_approx2']+1
length_approx3 = Data['nbIter_approx3']+1
markers_on1 = [Data['switch_it'] -1] #, length_approx-1]
markers_on2 = [Data['switch_it2']-1] #, length_approx2-1]
markers_on3 = [Data['switch_it3']-1] #, length_approx3-1]

flops_d = flop_calc_it("dynamic",K,N,Data['scrRate'], Data['zeros'],[])
flops_d1 = flop_calc_it("dynamic",K,N,Data['scrRate_approx'], Data['zeros_approx'],[],Data['RC'],Data['switch_it'])
flops_d2 = flop_calc_it("dynamic",K,N,Data['scrRate_approx2'], Data['zeros_approx2'],[],Data['RC'],Data['switch_it2'])
flops_d3 = flop_calc_it("dynamic",K,N,Data['scrRate_approx3'], Data['zeros_approx3'],[],Data['RC'],Data['switch_it3'])
       
axScreen.plot(np.arange(length),(1 - Data['scrRate'])*K, '-k', linewidth = 6,\
                label = 'D'+opt['scr_type']) #, 'x', markevery=[length-1]
axScreen.plot(np.arange(length_approx),(1 - Data['scrRate_approx'])*K, '-mx', markevery=markers_on1,\
                label = 'A-D'+opt['scr_type']+r'$: \| \mathbf{e}_j \|\!=\!10^{-1}$') # Marker on the swithing point 
axScreen.plot(np.arange(length_approx2),(1 - Data['scrRate_approx2'])*K, '-gx', markevery=markers_on2,\
                label = 'A-D'+opt['scr_type']+r'$: \| \mathbf{e}_j \|\!=\!10^{-2}$') 
axScreen.plot(np.arange(length_approx3),(1 - Data['scrRate_approx3'])*K, '-rx', markevery=markers_on3,\
                label = 'A-D'+opt['scr_type']+r'$: \| \mathbf{e}_j \|\!=\!10^{-3}$')
#axScreen.plot(np.arange(length_approx2),(1 - Data['scrRate_est_approx2'])*K, '-gx', markevery=markers_on2)

axScreen.axis([0,max(length,length_approx,length_approx2,length_approx3), 0, K*1.1])
axScreen.grid(True) 
axScreen.set_ylabel("Size of the dictionary")
axScreen.set_xlabel("Iteration t")
#axScreen.legend(fontsize=18,loc=1,frameon=False)


#axFlops_it.plot(np.arange(length),flops_ns,'-b')
axFlops_it.plot(np.arange(length),flops_d, '-k', linewidth = 6,\
                label = 'D'+opt['scr_type']) #markevery=[length-1])
axFlops_it.plot(np.arange(length_approx),flops_d1, '-mx', markevery=markers_on1,\
                label = 'A-D'+opt['scr_type']+r'$: \| \mathbf{e}_j \|\!=\!10^{-1}$')
axFlops_it.plot(np.arange(length_approx2),flops_d2, '-gx', markevery=markers_on2,\
                label = 'A-D'+opt['scr_type']+r'$: \| \mathbf{e}_j \|\!=\!10^{-2}$') 
axFlops_it.plot(np.arange(length_approx3),flops_d3, '-rx', markevery=markers_on3,\
                label = 'A-D'+opt['scr_type']+r'$: \| \mathbf{e}_j \|\!=\!10^{-3}$')
axFlops_it.grid(True)         
#axFlops_it.set_ylim((-0.19,1.15))
axFlops_it.set_ylim((0,1.1*max(max(flops_d),max(flops_d1),max(flops_d2),max(flops_d3))))
axFlops_it.set_xlim((0,50))
axFlops_it.set_ylabel("Flops number") #,fontsize = 24)
axFlops_it.set_xlabel("Iteration t")
axFlops_it.legend(fontsize=18,loc=1,frameon=False)
axFlops_it.ticklabel_format(style='sci', axis='y', scilimits=(0,0)) #ScientifiC Notation axis 'y'

f.savefig('./ResSynthData/3-Simu_screenRate_'+make_file_name(opt)+\
    '_lasso'+str(opt['lasso'])+'approx.eps',bbox_inches = 'tight',bbox_pad = 2 )

############## FIGURE 4 ###############
# Screen and Flops. noMargin: ON, Screen_est: ON, noPeak:ON
# High lambda

# Load data
filename = './ResSynthData/figures/4-ISTA_gnoise_N100_K500_ST1_regpath10_lambda_0.85.npz'
Data = np.load(filename)

f , (axScreen, axFlops_it) = \
    plt.subplots(2,1,sharex=True,figsize=2*plt.figaspect(1.3))    
opt = Data['opt'][()]
K = opt['K']
N = opt['N']
length = Data['nbIter']+1
length_approx = Data['nbIter_approx']+1
length_approx2 = Data['nbIter_approx2']+1
length_approx3 = Data['nbIter_approx3']+1
markers_on1 = [Data['switch_it'] -1] #, length_approx-1]
markers_on2 = [Data['switch_it2']-1] #, length_approx2-1]
markers_on3 = [Data['switch_it3']-1] #, length_approx3-1]

flops_d = flop_calc_it("dynamic",K,N,Data['scrRate'], Data['zeros'],[])
flops_d1 = flop_calc_it("dynamic",K,N,Data['scrRate_approx'], Data['zeros_approx'],[],Data['RC'],Data['switch_it'])
flops_d2 = flop_calc_it("dynamic",K,N,Data['scrRate_approx2'], Data['zeros_approx2'],[],Data['RC'],Data['switch_it2'])
flops_d3 = flop_calc_it("dynamic",K,N,Data['scrRate_approx3'], Data['zeros_approx3'],[],Data['RC'],Data['switch_it3'])
       
axScreen.plot(np.arange(length),(1 - Data['scrRate'])*K, '-k', linewidth = 6,\
                label = 'D'+opt['scr_type']) #, 'x', markevery=[length-1]
axScreen.plot(np.arange(length_approx),(1 - Data['scrRate_approx'])*K, '-mx', markevery=markers_on1,\
                label = 'A-D'+opt['scr_type']+r'$: \| \mathbf{e}_j \|\!=\!10^{-1}$') # Marker on the swithing point 
axScreen.plot(np.arange(length_approx2),(1 - Data['scrRate_approx2'])*K, '-gx', markevery=markers_on2,\
                label = 'A-D'+opt['scr_type']+r'$: \| \mathbf{e}_j \|\!=\!10^{-2}$') 
axScreen.plot(np.arange(length_approx3),(1 - Data['scrRate_approx3'])*K, '-rx', markevery=markers_on3,\
                label = 'A-D'+opt['scr_type']+r'$: \| \mathbf{e}_j \|\!=\!10^{-3}$')
#axScreen.plot(np.arange(length_approx2),(1 - Data['scrRate_est_approx2'])*K, '-gx', markevery=markers_on2)

axScreen.axis([0,max(length,length_approx,length_approx2,length_approx3), 0, K*1.1])
axScreen.grid(True) 
axScreen.set_ylabel("Size of the dictionary")
axScreen.set_xlabel("Iteration t")
#axScreen.legend(fontsize=18,loc=1,frameon=False)


#axFlops_it.plot(np.arange(length),flops_ns,'-b')
axFlops_it.plot(np.arange(length),flops_d, '-k', linewidth = 6,\
                label = 'D'+opt['scr_type']) #markevery=[length-1])
axFlops_it.plot(np.arange(length_approx),flops_d1, '-mx', markevery=markers_on1,\
                label = 'A-D'+opt['scr_type']+r'$: \| \mathbf{e}_j \|\!=\!10^{-1}$')
axFlops_it.plot(np.arange(length_approx2),flops_d2, '-gx', markevery=markers_on2,\
                label = 'A-D'+opt['scr_type']+r'$: \| \mathbf{e}_j \|\!=\!10^{-2}$') 
axFlops_it.plot(np.arange(length_approx3),flops_d3, '-rx', markevery=markers_on3,\
                label = 'A-D'+opt['scr_type']+r'$: \| \mathbf{e}_j \|\!=\!10^{-3}$')
axFlops_it.grid(True)         
#axFlops_it.set_ylim((-0.19,1.15))
axFlops_it.set_ylim((0,1.1*max(max(flops_d),max(flops_d1),max(flops_d2),max(flops_d3))))
axFlops_it.set_xlim((0,30))
axFlops_it.set_ylabel("Flops number") #,fontsize = 24)
axFlops_it.set_xlabel("Iteration t")
axFlops_it.legend(fontsize=18,loc=1,frameon=False)
axFlops_it.ticklabel_format(style='sci', axis='y', scilimits=(0,0)) #ScientifiC Notation axis 'y'

f.savefig('./ResSynthData/4-Simu_screenRate_'+make_file_name(opt)+\
    '_lasso'+str(opt['lasso'])+'approx.eps',bbox_inches = 'tight',bbox_pad = 2 )
    
############## FIGURE 5 ###############
# Low lambda

# Load data
filename = './ResSynthData/figures/5-ISTA_gnoise_N100_K500_ST1_regpath10_lambda_0.4.npz'
Data = np.load(filename)

f , (axScreen, axFlops_it) = \
    plt.subplots(2,1,sharex=True,figsize=2*plt.figaspect(1.3))    
opt = Data['opt'][()]
K = opt['K']
N = opt['N']
length = Data['nbIter']+1
length_approx = Data['nbIter_approx']+1
length_approx2 = Data['nbIter_approx2']+1
length_approx3 = Data['nbIter_approx3']+1
markers_on1 = [Data['switch_it'] -1] #, length_approx-1]
markers_on2 = [Data['switch_it2']-1] #, length_approx2-1]
markers_on3 = [Data['switch_it3']-1] #, length_approx3-1]

flops_d = flop_calc_it("dynamic",K,N,Data['scrRate'], Data['zeros'],[])
flops_d1 = flop_calc_it("dynamic",K,N,Data['scrRate_approx'], Data['zeros_approx'],[],Data['RC'],Data['switch_it'])
flops_d2 = flop_calc_it("dynamic",K,N,Data['scrRate_approx2'], Data['zeros_approx2'],[],Data['RC'],Data['switch_it2'])
flops_d3 = flop_calc_it("dynamic",K,N,Data['scrRate_approx3'], Data['zeros_approx3'],[],Data['RC'],Data['switch_it3'])
       
axScreen.plot(np.arange(length),(1 - Data['scrRate'])*K, '-k', linewidth = 6,\
                label = 'D'+opt['scr_type']) #, 'x', markevery=[length-1]
axScreen.plot(np.arange(length_approx),(1 - Data['scrRate_approx'])*K, '-mx', markevery=markers_on1,\
                label = 'A-D'+opt['scr_type']+r'$: \| \mathbf{e}_j \|\!=\!10^{-1}$') # Marker on the swithing point 
axScreen.plot(np.arange(length_approx2),(1 - Data['scrRate_approx2'])*K, '-gx', markevery=markers_on2,\
                label = 'A-D'+opt['scr_type']+r'$: \| \mathbf{e}_j \|\!=\!10^{-2}$') 
axScreen.plot(np.arange(length_approx3),(1 - Data['scrRate_approx3'])*K, '-rx', markevery=markers_on3,\
                label = 'A-D'+opt['scr_type']+r'$: \| \mathbf{e}_j \|\!=\!10^{-3}$')
#axScreen.plot(np.arange(length_approx2),(1 - Data['scrRate_est_approx2'])*K, '-gx', markevery=markers_on2)

axScreen.axis([0,max(length,length_approx,length_approx2,length_approx3), 0, K*1.1])
axScreen.grid(True) 
axScreen.set_ylabel("Size of the dictionary")
axScreen.set_xlabel("Iteration t")
#axScreen.legend(fontsize=18,loc=1,frameon=False)


#axFlops_it.plot(np.arange(length),flops_ns,'-b')
axFlops_it.plot(np.arange(length),flops_d, '-k', linewidth = 6,\
                label = 'D'+opt['scr_type']) #markevery=[length-1])
axFlops_it.plot(np.arange(length_approx),flops_d1, '-mx', markevery=markers_on1,\
                label = 'A-D'+opt['scr_type']+r'$: \| \mathbf{e}_j \|\!=\!10^{-1}$')
axFlops_it.plot(np.arange(length_approx2),flops_d2, '-gx', markevery=markers_on2,\
                label = 'A-D'+opt['scr_type']+r'$: \| \mathbf{e}_j \|\!=\!10^{-2}$') 
axFlops_it.plot(np.arange(length_approx3),flops_d3, '-rx', markevery=markers_on3,\
                label = 'A-D'+opt['scr_type']+r'$: \| \mathbf{e}_j \|\!=\!10^{-3}$')
axFlops_it.grid(True)         
#axFlops_it.set_ylim((-0.19,1.15))
axFlops_it.set_ylim((0,1.1*max(max(flops_d),max(flops_d1),max(flops_d2),max(flops_d3))))
#axFlops_it.set_xlim((0,50))
axFlops_it.set_ylabel("Flops number") #,fontsize = 24)
axFlops_it.set_xlabel("Iteration t")
axFlops_it.legend(fontsize=18,loc=4,frameon=False)
axFlops_it.ticklabel_format(style='sci', axis='y', scilimits=(0,0)) #ScientifiC Notation axis 'y'

f.savefig('./ResSynthData/4-Simu_screenRate_'+make_file_name(opt)+\
    '_lasso'+str(opt['lasso'])+'approx.eps',bbox_inches = 'tight',bbox_pad = 2 )
    
############## FIGURE 7 ###############
# All lambdas
    
# Load data
#filename = './ResSynthData/expe_5/best_ISTA_gnoise_N1000_K5000_ST1_regpath10_done.npz'
filename = './ResSynthData/expe_5/bernoulli-gaussian/beta02_ISTA_gnoise_N1000_K5000_ST1_regpath10_done.npz'
Data = np.load(filename)

opt = Data['opt'][()]

timeRes = Data['timeRes'][()]
nbFlops = Data['nbFlops'][()]
nbIter = Data['nbIter'][()]

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
   
pen_param_list = make_pen_param_list(opt['samp'])  
mkevry = max(1,len(pen_param_list)/10)


if opt['Gr']:
        Gstr = 'G'
else:
        Gstr =''    

# Dynamic      
f = plt.figure(figsize=1.35*plt.figaspect(0.6)) #1.27*plt.figaspect(0.6)
ax = f.add_subplot(111)
ax.set_rasterization_zorder(1) 
                 
plt.semilogx(pen_param_list,flop_q2_d,'ks-',
             label = 'D'+Gstr+opt['scr_type'],                 
             markevery= mkevry, markersize = markersize)  
plt.fill_between(pen_param_list, flop_q1_d, flop_q3_d,alpha = 0.2,
                     facecolor = 'k')
#    plt.fill_between(pen_param_list,flop_q1_d, flop_q3_d,alpha = 0.1,
#                         edgecolor = 'k', hatch = '/',color='none')
# Dynamic approx 1e-1
plt.semilogx(pen_param_list,flop_q2_d1,'m^-',
             label = 'A-D'+Gstr+opt['scr_type']+r'$: \sigma\!=\!10^{-1}$', 
             markevery= mkevry, markersize = markersize)  
plt.fill_between(pen_param_list, flop_q1_d1, flop_q3_d1,alpha = 0.2,
                     facecolor = 'm')
#    plt.fill_between(pen_param_list,flop_q1_d1, flop_q3_d1,alpha = 0.1,
#                         edgecolor = 'm', hatch = '/',color='none')
# Dynamic approx 1e-2
plt.semilogx(pen_param_list,flop_q2_d2,'gD-',
             label = 'A-D'+Gstr+opt['scr_type']+ r'$: \sigma\!=\!10^{-2}$',                 
             markevery= mkevry, markersize = markersize)  
plt.fill_between(pen_param_list, flop_q1_d2, flop_q3_d2,alpha = 0.2,
                     facecolor = 'g')
#    plt.fill_between(pen_param_list,flop_q1_d2, flop_q3_d2,alpha = 0.1,
#                         edgecolor = 'g', hatch = '/',color='none')
# Dynamic approx 1e-3
plt.semilogx(pen_param_list,flop_q2_d3,'rv-',
             label = 'A-D'+Gstr+opt['scr_type']+r'$: \sigma\!=\!10^{-3}$',                 
             markevery= mkevry, markersize = markersize)  
plt.fill_between(pen_param_list, flop_q1_d3, flop_q3_d3,alpha = 0.2,
                     facecolor = 'r')
#    plt.fill_between(pen_param_list,flop_q1_d3, flop_q3_d3,alpha = 0.1,
#                         edgecolor = 'r', hatch = '/',color='none')                         

   
plt.grid(True, which="both")
plt.xticks(pen_param_list, ['0.1','','0.3','','0.5','','0.7','','0.9'])
plt.ylim((0,1.15))
plt.ylabel("Normalized flops number",fontsize = 24)
plt.xlabel(r"$\lambda/\lambda_{\mathrm{max}}$")
plt.legend(fontsize=20,loc=3,frameon=False)

f.suptitle(type2name(opt['dict_type']) + ' + ' + opt['algo_type'],fontsize=26)

#f.set_rasterized(True)
#f.savefig('./ResSynthData/'+make_file_name(opt)+'_Simu_relNoScreen.eps',bbox_inches = 'tight', rasterized=True, dpi=300 )

f.savefig('./ResSynthData/'+make_file_name(opt)+'_Simu_relNoScreen.pdf',bbox_inches = 'tight',bbox_pad = 2)

############## FIGURE 7 (2 plots in 1)###############
# All lambdas. y gaussian and y=X\beta (beta bernoulli-gaussian)
    
# Load data
filename = './ResSynthData/expe_5/best_ISTA_gnoise_N1000_K5000_ST1_regpath10_done.npz'
Data = np.load(filename)

opt = Data['opt'][()]

timeRes = Data['timeRes'][()]
nbFlops = Data['nbFlops'][()]
nbIter = Data['nbIter'][()]

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
   
pen_param_list = make_pen_param_list(opt['samp'])  
mkevry = max(1,len(pen_param_list)/10)


if opt['Gr']:
        Gstr = 'G'
else:
        Gstr =''    

# Dynamic      
f = plt.figure(figsize=1.35*plt.figaspect(0.6)) #1.27*plt.figaspect(0.6)
ax = f.add_subplot(111)
ax.set_rasterization_zorder(1) 
                 
plt.plot(pen_param_list,flop_q2_d,'k:',
             #label = 'D'+Gstr+opt['scr_type'],                 
             markevery= mkevry, markersize = markersize)  
#plt.fill_between(pen_param_list, flop_q1_d, flop_q3_d,alpha = 0.2,
#                     facecolor = 'k')
# Dynamic approx 1e-1
plt.plot(pen_param_list,flop_q2_d1,'m:',
             #label = 'A-D'+Gstr+opt['scr_type']+r'$: \sigma\!=\!10^{-1}$', 
             markevery= mkevry, markersize = markersize)  
#plt.fill_between(pen_param_list, flop_q1_d1, flop_q3_d1,alpha = 0.2,
#                     facecolor = 'm')
# Dynamic approx 1e-2
plt.plot(pen_param_list,flop_q2_d2,'g:',
             #label = 'A-D'+Gstr+opt['scr_type']+ r'$: \sigma\!=\!10^{-2}$',                 
             markevery= mkevry, markersize = markersize)  
#plt.fill_between(pen_param_list, flop_q1_d2, flop_q3_d2,alpha = 0.2,
#                     facecolor = 'g')
# Dynamic approx 1e-3
plt.plot(pen_param_list,flop_q2_d3,'r:',
             #label = 'A-D'+Gstr+opt['scr_type']+r'$: \sigma\!=\!10^{-3}$',                 
             markevery= mkevry, markersize = markersize)  
#plt.fill_between(pen_param_list, flop_q1_d3, flop_q3_d3,alpha = 0.2,
#                     facecolor = 'r')

# Load data
filename = './ResSynthData/expe_5/bernoulli-gaussian/beta02_ISTA_gnoise_N1000_K5000_ST1_regpath10_done.npz'
Data = np.load(filename)

opt = Data['opt'][()]

timeRes = Data['timeRes'][()]
nbFlops = Data['nbFlops'][()]
nbIter = Data['nbIter'][()]

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
   
pen_param_list = make_pen_param_list(opt['samp'])  
mkevry = max(1,len(pen_param_list)/10)


if opt['Gr']:
        Gstr = 'G'
else:
        Gstr =''    

# Dynamic      
                
plt.plot(pen_param_list,flop_q2_d,'ks-',
             label = 'D'+Gstr+opt['scr_type'],                 
             markevery= mkevry, markersize = markersize)  
plt.fill_between(pen_param_list, flop_q1_d, flop_q3_d,alpha = 0.2,
                     facecolor = 'k')
#    plt.fill_between(pen_param_list,flop_q1_d, flop_q3_d,alpha = 0.1,
#                         edgecolor = 'k', hatch = '/',color='none')
# Dynamic approx 1e-1
plt.plot(pen_param_list,flop_q2_d1,'m^-',
             label = 'A-D'+Gstr+opt['scr_type']+r'$: \sigma\!=\!10^{-1}$', 
             markevery= mkevry, markersize = markersize)  
plt.fill_between(pen_param_list, flop_q1_d1, flop_q3_d1,alpha = 0.2,
                     facecolor = 'm')
#    plt.fill_between(pen_param_list,flop_q1_d1, flop_q3_d1,alpha = 0.1,
#                         edgecolor = 'm', hatch = '/',color='none')
# Dynamic approx 1e-2
plt.plot(pen_param_list,flop_q2_d2,'gD-',
             label = 'A-D'+Gstr+opt['scr_type']+ r'$: \sigma\!=\!10^{-2}$',                 
             markevery= mkevry, markersize = markersize)  
plt.fill_between(pen_param_list, flop_q1_d2, flop_q3_d2,alpha = 0.2,
                     facecolor = 'g')
#    plt.fill_between(pen_param_list,flop_q1_d2, flop_q3_d2,alpha = 0.1,
#                         edgecolor = 'g', hatch = '/',color='none')
# Dynamic approx 1e-3
plt.plot(pen_param_list,flop_q2_d3,'rv-',
             label = 'A-D'+Gstr+opt['scr_type']+r'$: \sigma\!=\!10^{-3}$',                 
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

#f.set_rasterized(True)
#f.savefig('./ResSynthData/'+make_file_name(opt)+'_Simu_relNoScreen.eps',bbox_inches = 'tight', rasterized=True, dpi=300 )

f.savefig('./ResSynthData/'+make_file_name(opt)+'_Simu_relNoScreen.pdf',bbox_inches = 'tight',bbox_pad = 2)
