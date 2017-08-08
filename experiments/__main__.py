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
    else:
        print 'Experiment number Not valid'
    
