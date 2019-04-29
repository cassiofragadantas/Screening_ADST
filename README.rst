
Safe screening rules with approximate dictionaries
==================================================

This is a **Python 2.7** code originally based on the code from Antoine Bonnefoy concerning the publication "Dynamic Screening: accelerating First-order Algorithms for the Lasso and Group-Lasso." A. Bonnefoy et al., 2015.

This code corresponds to the following publications:

[1] C. F. Dantas and R. Gribonval, “Dynamic screening with approximate dictionaries,” in Colloque GRETSI (Sep 2017). hal.inria.fr/hal-01598021

[2] C. F. Dantas and R. Gribonval, “Faster and still safe: combining screening techniques and structured dictionaries to accelerate the lasso,” in IEEE International Conference on Acoustics, Speech and Signal Processing (Apr 2018). hal.inria.fr/hal-01706392

[3] C. F. Dantas and R. Gribonval, "Stable safe screening and structured dictionaries for faster L1 regularization", in IEEE Transactions on Signal Processing (Apr 2019). hal.inria.fr/hal-01954261

.. contents::


------------
Requirements
------------

- Numpy: $ pip2 install numpy
- Matplotlib: $ pip2 install matplotlib
- scipy: $ pip2 install scipy
- Cython: $ pip2 install Cython
- cblas (C library): for instance, in Ubuntu: $ sudo apt-get install libopenblas-dev


Your python 2 executable should be python2. Otherwise, you can do:

    $ alias python2="python2.7"

------------
Installation
------------

1.  Clone this repository (or download it as a .zip):

    $ git clone https://github.com/cassiofragadantas/Screening_ADST.git
    
2. Run the desired experiment

    $ python2 -m experiments [EXP_NUM]
    
    where [EXP_NUMBER] is to be replaced the number of the desired experiment (see section Experiments).

This will generate the corresponding figures and place in the ResSynthData folder.
If it doesn't work, you probably need to:

3. Recompile the  'fast_mat_prod' cython function [*]_:

    $ cd dynascreen/ ; rm fast_mat_prod.so ; ./compile.sh; cd ../

.. [*] fast_mat_prod was originally compiled for a GNU/Linux 64-bit PC with an Intel(R) Core(TM) i7-5600U CPU @ 2.60GHz. So, if you you're using a similar configuration, it might work without recompiling. 

If you have a error finding 'cblas.h', verify that you have properly installed a cblas library (see Requirements section). Then, you can also try to uncomment lines 13 and 14 in the file dynascreen/setup.py.
If you use GNU Scientific Library (GSL) CBLAS, you need replace libraries=['blas'] by library=['gslcblas'] in the file dynascreen/setup.py.

-----------
Experiments
-----------

Quick start
-----------

To test the final version of the proposed algorithm (the one described in [Dantas and Gribonval, 2019] IEEE TSP, using multiple approximate dictionaries), run:

    $ python -m experiments 9
    
    It runs for a single regularization value. A figure showing the screening progression (number of preserved atoms) over the iteratios is generated in the folder ResSynthData. A npz file containing detailed results is also created in the same folder. The first time you run this experiment, it takes longer due to the calculation of the empirical Relative Complexity (RC).
    
    $ python -m experiments 10
    
    This runs a full lasso path (a range of regularization values). It generates figures for the Normalized Running Times and Flops as a function of the regularization value. By default, the theoretical Relative Complexity (RC) is used for the fast dictionaries, that's why there can be some deviation between the two figures.

Changing the experiments parameters:

This can be easily done by modifying the 'default' variable in 'expe_journal.py' file (functions 'first' and 'second' respectively for the experiments 9 and 10).
Please refer to the section 'Parameter description' below for a brief description of each key value.

Complete list
-------------

Complete list of available experiments (its number is to be placed in [EXP_NUM] field).
See ./experiments/__main__.py to find out which function is called for each [EXP_NUM].

**Experiments concerning publication [Bonnefoy et al. 2015]:**
  
1. The screening progression. Number of preserved atoms as a function of the iteration, for one given regularization.
2. Normalized Running Times and Flops for a complete Lasso path.
3. (DEPRECATED) Idem as 2 for the Group-Lasso .

**Experiments concerning publication [Dantas and Gribonval, 2017] (GRETSI):**

4. The screening progression (as 1, but including the screening with approximate dictionary).
5. Normalized Running Times and Flops for a complete Lasso path (as in 2, but including approximate dictionary)

**Experiments concerning publication [Dantas and Gribonval, 2018] (ICASSP):**

6. The screening progression (as 1 and 4, but with a SuKro fast dictionary and using the extended GAP sphere rule).
7. Normalized Running Times and Flops for a complete Lasso path (as in 2 and 5,  but with a SuKro fast dictionary and using the extended GAP sphere rule)
8. Colormap visualisation of the normaalized running time per iteration on a complete Lasso path.

**Experiments concerning publication [Dantas and Gribonval, 2019] (IEEE TSP):**

9.  The screening progression (as 1, 4 and 6, but using multiple SuKro approximate dictionaries and using the extended GAP sphere rule).
10. Normalized Running Times and Flops for a complete Lasso path (as in 2, 5 and 7,  but with multiple SuKro fast dictionaries and using the extended GAP sphere rule)

11. This runs either:

    - a complete set of parameters (different decay constants, switching criteria, stopping criteria and algorithm type) as specified by the variable 'default' in the 'complete' function in 'expe_journal.py' file. The results of this simulation were used to generate Figs. 5, 7 and 8 in the TSP paper. It takes many hours.
    - a finer set of values for the switching parameter \Gamma (by commenting the first 'default' variable and uncommenting the second one). The results of this simulation were used to generate Fig. 6 in the TSP paper.
    
12. Generates Fig. 4 from TSP paper.
13. Generates Fig. 2 from TSP paper.
14. Generates Fig. 3 from TSP paper.
15. The results of this simulation were used to generate Fig. 9 in the paper.
16. Experiments with MEG data [*]_. The results of this simulation were used to generate Fig. 10 in the paper.
    
    .. [*] Fig. 10 is not directly reproductible since it requires to load the MEG data which is not given in the repository. Please contact the authors if you want to reproduce this experiment. You will need the measurement matrix X_meg.mat to be placed in './datasets/MEG/', as well as the FAuST approximate dictionaries in mat files. You will also need to install the FAuST python toolbox (faust.inria.fr).


Parameter description
---------------------

The user can easily change the experiments parameters by modifying the key values of the dictionary 'default' in 'expe_journal.py' file.

Here is a brief description of each key:
                
- **dict_type**: type of dictionary used in the experiment. 'sukro_approx' corresponds to a dictionary which can be efficiently approximated by a SuKro matrix. Some possible choices: 'gnoise' (random gaussian) ,'pnoise', 'low-rank', 'sukro_approx'.
- **data_type**: the distribution of the input data vector. Some choices: 'gnoise', 'pnoise', 'bernoulli-gaussian'.
- **N, K**: dimensions of the dictionary (respectively nb. of lines and columns).
- **scr_type**: screening rule used, e.g.  scr_type = 'GAP'. In case the function runProtocol is used for the experiment (e.g. in 'complete' experience) this key is a list containing all screening rules to be tested. For instance, to run only GAP safe rule use scr_type = ['GAP']
- **dict_params**: contains extra parameters for the structured dictionary. For instance, for dict_type = 'sukro_approx', the following parameters are required:
    - **N1, N2, K1, K2**: dimensions of the subdictionaries A1 (N1xK1) and A2 (N2xK2), such that A = kron(A1,A2)
    - **n_kron**: This field is a list of lists. Each list contains the number of kronecker summing terms of the (potentially multiple) approximate dictionaries used before switching back to the original dictionary. E.g. n_kron = [[5], [5, 10, 15, 20]] will run two separate simulations, the first one using one single approximate dictionary with 5 summing terms and the second simulation using multiple approximate dictionaries with 5, 10, 15 and 20 summing terms each.
    - **svd_decay**: contains the approximation profile of the dictionary w.r.t. the SuKro approximations. Currently, if svd_decay is not 'exponential', then a very tough scenario is simulated.
    - **svd_decay_const**: this value basically controls how efficiently the dictionary is approximated by the sukro approximate dictionaries. The higher, the better the approximations. Default value is 0.5 (if not provided).
    - **reuse**: if this key is present (regardless of its value, i.e. even if it is set to False), then the SuKro summing terms are saved in the results folder and reused in the whole set of experiments.
- **nbRuns**: Number of repetitions. For each run, a new input vector is generated.
- **stop**: Stopping criteria. This entry is a dictionary containing a different stopping criterion in each of its keys. In case the function runProtocol is used for the experiment this key is a list of dictionaries. One experiment will be run using each of the provided criteria in the list (see the 'complete' experience, for instance).
    - **dgap_tol**: threshold for the duality gap.
    - **max_iter**: maximum number of iterations.
- **switching**: Defines the switching criterion. Possible values are: 'off' (never switches to the original dictionary) 'screening_only' (switches only due to the speed criterion, i.e. if the screening ration is high enough)  'default' (uses both the convergence and speed criteria).
- **switching_gamma**: Threshold for convergence-based switching criterion. Typically some value between 0.1 and 0.8.
- **min_reg**: minimum regularization value (:math:`\lambda/\lambda_{max}`) to be simulated.
- **samp**: number of regularization values to be tested within the range [min_reg, 1].
- **samp_type**: defines how the simulated regularizations are spaced. Possible values: 'linear'(linear sampling), 'log' (logarithmic sampling)
- **algo_type**: optimization algorithm. Possible choices: 'ISTA', 'FISTA', 'SPARSA', 'Chambolle_Pock', 'TWIST'.
- **wstart**: Warm start when solving a full lasso path. Values: True or False.
    
**Beware!** *Some combinations of parameters are not guaranteed to work in the current version of the code.*

Reproducing results from ICASSP and GRETSI papers
-------------------------------------------------

To be sure to work on the exact same version as in the ICASSP paper, simply clone the STABLE branch by replacing step 1 on the Installation section by the following command:
    
 $ git clone -b STABLE https://github.com/cassiofragadantas/Screening_ADST.git
 

For any question feel free to contact me at:
    cassio.fraga-dantas at inria dot fr 
    cassiofragadantas at gmail dot com

--------------
Code hierarchy 
--------------
.. image:: code_hierarchy.svg
