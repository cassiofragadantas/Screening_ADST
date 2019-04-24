# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 16:47:02 2014

@author: antoinebonnefoy
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Oct 29 19:09:35 2014

@author: antoinebonnefoy
"""
import argparse
 
import expe

import expe_approx

import expe_journal

import os
                                 
if __name__=='__main__':

    parser = argparse.ArgumentParser(prog = 'Test Dynamic Screening',
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     description='''\
----------------------------------------------------------------------
    This programm tests the performance of the Dybnamic Screening.                         
    Reproduce experiments proposed in : 
    "A Dynamic Screening Principle for the LASSO"
                A. Bonnefoy, V. Emiya, L. Ralaivola, R. Gribonval
                EUSIPCO 2014
----------------------------------------------------------------------
                        ''',  add_help =True)

    parser.add_argument('Expe', type=int, nargs='+',
                       help='''\
Indicate which experiment to run.
        1: screening progression
        2: normalized time and flops for the lasso
        3: normalized time and flops for the group-lasso\
                                ''')
                                
    parser.add_argument('-dict', nargs = 1,type = str, metavar = 'type',
                       help='Change the dictionary type : gnoise, pnoise (default)',
                       dest = 'dict_type')    
    parser.add_argument('-sizes', nargs = 2,type = int, metavar = ('N','K'),
                       help='Change the size of the dictionary (default: 1000*5000)',
                       dest = 'shape')              
    parser.add_argument('-algo', nargs = 1,type = str, metavar = 'type',
                       help='Change the algorithm possible types : ISTA (default), FISTA, SPARSA, SPARSA, Chambolle-Pock',
                       dest = 'algo_type')
    parser.add_argument('-stop', nargs = 1,type = float,  metavar = 'val',
                       help='Change the stopping criteria threshold : 1e-8(default)',
                       dest = 'stop')
    parser.add_argument('-decay', nargs = 1,type = float, metavar = 'val',
                       help='(Only if dict_type is sukro_approx) Changes the dictionary approximation-speedup compromise (higher decay implies better compromises)',
                       dest = 'svd_decay_const_list')
    parser.add_argument('-extra', nargs = 2,  metavar = ('option','value'),
                       help='Change extra default values : -extra scr_type ST1',
                       action = 'append') 


                       


    args = parser.parse_args() 
    option =dict()
    for key, val in args._get_kwargs():
        if val!=None and key!='extra' and key!='shape':
            option[key]=val[0]
        if key == 'stop' and val!=None :
            option['stop'] = dict(rel_tol=val)            
        
    if args.extra!=None:
        exec('option['+'args.extra[0][0]'+']='+'args.extra[0][1]')
    if args.shape!=None:        
        option['N']=args.shape[0]
        option['K']=args.shape[1]

    if 'ResSynthData' not in os.listdir('./'):
        os.mkdir('ResSynthData')
        
    if args.Expe[0]==1: 
        expe.first(option)
    elif args.Expe[0]==2: 
        expe.second(option)        
    elif args.Expe[0]==3: 
        expe.third(option)
    elif args.Expe[0]==4: 
        expe_approx.first(option)
    elif args.Expe[0]==5: 
        expe_approx.second(option)
    elif args.Expe[0]==6: 
        expe_approx.first_sukro(option)
    elif args.Expe[0]==7: 
        expe_approx.second_sukro(option)
    elif args.Expe[0]==8: 
        expe_approx.second_sukro_per_it(option)
    # Jounal experiments
    elif args.Expe[0]==9: 
        expe_journal.first(option)
    elif args.Expe[0]==10:
        expe_journal.second(option)
    elif args.Expe[0]==11:
        expe_journal.complete(option)
    elif args.Expe[0]==12:
        expe_journal.approx_RC_compromise(option)
    elif args.Expe[0]==13:
        expe_journal.scatterplot_screening_rates(option)
    elif args.Expe[0]==14:
        expe_journal.gap_evolution_it_time(option)
    elif args.Expe[0]==15:
        expe_journal.gap_evolution_it_time_tol(option)
    elif args.Expe[0]==16:
        expe_journal.MEG_gap_evolution_it_time_tol(option)
    else:
        print 'Experiment number Not valid'
    
