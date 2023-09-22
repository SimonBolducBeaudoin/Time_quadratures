#!/bin/env/python
#! -*- coding: utf-8 -*-
import numpy as _np
from numba import float64,complex128,vectorize,guvectorize

from special_function import Tukey 

@guvectorize([(complex128[:], float64, complex128[:])], '(n),()->(n)')
def normalize(beta,df,res):
    """	Normalized to 1/2 :
						 beta(f)
	Beta(f)	=	---------------------------
			   ( 2 int |beta(f)|^2 df )^1/2
    """
    res[:] = beta[:]/(  _np.sqrt(2*df*( _np.abs(beta[:])**2 ).sum() ) )
    
@guvectorize([(float64[:],float64,float64,float64,float64,complex128[:])], '(n),(),(),(),()->(n)')
def flatband(f,f_1,f_2,f_3,f_4,res):
    """
    flatband( f,f_1,f_2,f_3,f_4)
    """
    df = f[1]-f[0]
    res[:] = _Tukey(f[:],f_1,f_2,f_3,f_4)
    res[:] = res[:]/(  _np.sqrt(2*df*( _np.abs(res[:])**2 ).sum() ) ) # normalization