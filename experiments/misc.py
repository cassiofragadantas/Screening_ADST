# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 18:01:35 2014

@author: antoinebonnefoy, cassiofraga

Copyright (C) 2019 Cassio Fraga Dantas

SPDX-License-Identifier: AGPL-3.0-or-later
"""

import numpy as np



def mergeopt(opt, default, keywords):
    '''
    Merges the 3 ways of giving parameters into opt
    the priority order is keywords>opt>default
    '''
    for key in keywords.keys():
        opt[key] = keywords[key]
    for key in default.keys():
        if key not in opt.keys():
            opt[key]=default[key]
            
    if opt['dict_type']=='cancer':
        #opt['nbRuns']=1
        opt['K'] = 15126
        opt['N'] = 295          
        opt['Gr']= 10
            
    return opt

def make_file_name(opt):
#    for key in opt.keys():
#        exec(key+' = '+repr(opt[key]))  
    name = opt['algo_type']+'_'+opt['dict_type']+'-dict_'+opt['data_type']+'-data_N'+str(opt['N'])+"_K"+\
            str(opt['K'])+'_'+str(opt['scr_type'])

    if opt['dict_type'] == 'sukro':
        name += "_sep-terms"+str(opt['dict_params']['n_kron'])
        
    if opt['dict_type'] == 'sukro_approx':
        if not opt['dict_params'].has_key('svd_decay_const'):
            opt['dict_params']['svd_decay_const'] = 0.5
        name += "_decay"+str(opt['dict_params']['svd_decay_const'])
       
    if opt['sparse'] != None:
        name += "_sp"+str(opt['sparse'])
    
    if opt['wstart']:
        name += "_wstart"
    
    if opt['Gr'] != 0:
        name += '_Gr'+str(opt['grsize'])

    if 'samp' in opt.keys() :
        if opt['samp']!=20:
            name+= '_regpath'+str(opt['samp'])
        
        if opt['spacing'] != 'linear':
                    name+= '_spacing'+str(opt['spacing'])

    # Stop criterion
    name += '_Stop-'+opt['stop'].keys()[0] + str(opt['stop'].values()[0])
                    
    if 'nbRuns' in opt.keys():
        name += "_nbRuns"+str(opt['nbRuns'])
            
    return name
    
    
def make_pen_param_list(samp=10,min_reg=0.1,samp_type='linear'):
    #return ((np.arange(samp-1,dtype=float)+1.)/samp)
    max_reg = 0.9
    if samp_type is 'linear':
        return np.linspace(min_reg,max_reg,samp-1)
    else:
        return np.logspace(np.log10(min_reg),np.log10(max_reg),samp)
        
        

def type2name(dict_type,opt={}):    
    if dict_type =='gnoise':
        return 'Gaussian dictionary' 
    elif dict_type == 'pnoise':
        return 'Pnoise dictionary' 
    elif dict_type == 'audio':
        return 'Audio Data' 
    elif dict_type =='MNIST':
        return 'MNIST Database'      
    elif dict_type =='cancer':
        return 'Cancer Database with groups'
    elif dict_type =='sukro':
        return 'SuKro ('+str(opt['dict_params']['n_kron'])+' terms)'
    elif dict_type =='low-rank':
        return 'Low Rank (rank = '+str(opt['dict_params']['n_rank'])+')'
    else:
        return dict_type
def testopt(opt): 
    """
    tests if the algo and the Dictionary are valid 
    """
    if opt['algo_type'] not in ['ISTA','FISTA','SPARSA','Chambolle_Pock','TWIST']:
        raise NotImplementedError(opt['algo_type']+' Algorithm is not implemented yet')
        exit(0)
    if opt['dict_type'] not in ['gnoise','pnoise','sukro','low-rank','DCT','MEG','MEG_low-rank','sukro_approx']:
        raise ValueError('Not Valid dictionary')
    if opt['data_type'] not in ['gnoise','pnoise','bernoulli-gaussian','MNIST','audio','audioSynth','cancer']:
        raise ValueError('Not Valid data type')
        
        
        
def default_expe():
    return dict(N=1000,K=2000, Gr= 0, dict_type = "pnoise",
                dict_params  = {}, data_params = {},
                samp_type='linear', samp=10, min_reg = 0.1,\
                nbRuns = 10,disp_fig=True,algo_type='ISTA',\
                spacing = 'linear', geo_rate =3,\
                L='backtracking', wstart = False,sparse=None,scr_type = 'ST3',\
                recalc = 0,stop=dict(rel_tol=1e-10, max_iter=10000), verbose=1, grsize = 10, lasso=0.75)
