#!/bin/env/python
#! -*- coding: utf-8 -*-
from __future__ import division
from past.utils import old_div
import numpy as _np
import numba as _nb

from ..Math_extra.special_functions import _Tukey

@_nb.guvectorize([(_nb.complex128[:], _nb.float64, _nb.complex128[:])], '(n),()->(n)')
def normalize(beta,df,res):
    """	Normalized to 1/2 :
						 beta(f)
	Beta(f)	=	---------------------------
			   ( 2 int |beta(f)|^2 df )^1/2
    """
    res[:] = old_div(beta[:],(  _np.sqrt(2*df*( _np.abs(beta[:])**2 ).sum() ) ))
    
def f_bar(betas,freq):
    """
    I couldn't implement it in numba for some reason ..?
    """
    df = freq[1] - freq[0]
    return ( freq* (_np.abs(betas)**2) ).sum(axis=-1)*2*df
        
@_nb.vectorize([_nb.float64(_nb.float64,_nb.float64,_nb.float64)])
def _gaussian (x,mu=0.0,sigma=1.0) :
    return _np.exp( old_div((-(x-mu)**2),(2.0*sigma**2)) )

@_nb.guvectorize([(_nb.float64[:],_nb.float64[:],_nb.complex128[:])], '(n),(m)->(n)') 
def gaussian(f,params,res):
    f_mean,f_std = params[0],params[1]
    df = f[1]-f[0]
    res[:] = _gaussian (f[:],f_mean,f_std)
    res[:] = old_div(res[:],(  _np.sqrt(2*df*( _np.abs(res[:])**2 ).sum() ) )) # normalization
    
@_nb.guvectorize([(_nb.float64[:],_nb.float64[:],_nb.float64[:],_nb.complex128[:])], '(n),(m),(m)->(n)') 
def bigaussian(f,p1,p2,res):
    f1_m,f1_s,phi1 = p1[0],p1[1],p1[2]
    f2_m,f2_s,phi2  = p2[0],p2[1],p2[2]
    df = f[1]-f[0]
    res[:] = _np.exp(1j*phi1)*_gaussian (f[:],f1_m,f1_s) + _np.exp(1j*phi2)*_gaussian (f[:],f2_m,f2_s)
    res[:] = old_div(res[:],(  _np.sqrt(2*df*( _np.abs(res[:])**2 ).sum() ) )) # normalization
     
@_nb.guvectorize([(_nb.float64[:],_nb.complex128[:,:],_nb.complex128[:])], '(n),(l,m)->(n)') 
def multigaussian(f,ps,res):
    """
    multigaussian(f,ps)
    Calculate a linear combination of Gaussian functions with varying parameters and normalize the result.

    Parameters:
    -----------
    f : numpy.ndarray, float64[:]
        Array of input values for the independent variable.

    ps : numpy.ndarray, complex128[:, :]
        Array of Gaussian function parameters with shape (l, m), where 'l' is the number of Gaussian functions and 'm' is the number of parameters for each Gaussian function. Each row of 'ps' should contain three values: mean, standard deviation and prefactor.
        
    Returns:
    --------
    res : numpy.ndarray, complex128[:]
        Output array where the result is stored. It should have the same length as 'f'.

    Notes
    The formula used for normalization is based on the following equation:
    res[:] = res[:] / sqrt(2 * df * sum(|res[:]|^2))

    Where 'df' is the spacing between adjacent values in the 'f' array.
    """
    df = f[1]-f[0]
    res[:] = 0
    for i in range(ps.shape[0]) :
        res[:] += ps[i,2]*_gaussian(f[:],_np.abs(ps[i,0]),_np.abs(ps[i,1]))
    res[:] = old_div(res[:],(  _np.sqrt(2*df*( _np.abs(res[:])**2 ).sum() ) )) # normalization
    
@_nb.guvectorize([(_nb.float64[:],_nb.float64[:],_nb.complex128[:])], '(n),(l)->(n)')
def flatband(f,params,res):
    """
    flatband( f,[f_1,f_2,f_3,f_4])
    """
    df = f[1]-f[0]
    f_1,f_2,f_3,f_4 = params[0],params[1],params[2],params[3]
    res[:] = _Tukey(f[:],f_1,f_2,f_3,f_4)
    res[:] = old_div(res[:],(  _np.sqrt(2*df*( _np.abs(res[:])**2 ).sum() ) )) # normalization
    
@_nb.guvectorize([(_nb.float64[:],_nb.float64[:],_nb.complex128[:])], '(n),(l)->(n)')
def flatband_v(f,params,res):
    """
    Flatband for voltage modes
    flatband_v( f,[f_1,f_2,f_3,f_4])
    """
    df = f[1]-f[0]
    f_1,f_2,f_3,f_4 = params[0],params[1],params[2],params[3]
    res[:] = _np.sqrt(f[:])*_Tukey(f[:],f_1,f_2,f_3,f_4)
    res[:] = old_div(res[:],(  _np.sqrt(2*df*( _np.abs(res[:])**2 ).sum() ) )) # normalization
    
def concatenate_betas(*args):
    t = tuple()
    for arg in args :
        if not (arg.size==0) :
            t += (arg,)
    return _np.concatenate( t, axis = 0 ) 
    