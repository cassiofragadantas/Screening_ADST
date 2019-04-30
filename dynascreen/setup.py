# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 16:41:32 2017

@author: cfragada
"""

from distutils.core import setup, Extension
from Cython.Distutils import build_ext
import numpy

includes = [numpy.get_include()]
from numpy.distutils.system_info import get_info
if get_info("blas_opt").has_key("library_dirs"): includes = includes+get_info("blas_opt")["library_dirs"]
else: print("\n\n!!!!!!!!!\n!WARNING! You probably do not have a CBLAS library installed.\n!!!!!!!!!\n\n")
if get_info("blas_opt").has_key("include_dirs"): includes = includes+get_info("blas_opt")["include_dirs"]

setup(
   cmdclass={'build_ext': build_ext},
   ext_modules=[Extension("fast_mat_prod", ["fast_mat_prod.pyx"],
                          libraries=["blas"],
                          #libraries=["gslcblas"],
                          include_dirs=includes
                          )
               ]

)
