#!/bin/env/python
#! -*- coding: utf-8 -*-
import numpy as _np
from scipy import constants as _C
from numba import float64,complex128,vectorize,guvectorize,jit
from scipy.signal.windows import tukey

from ..Math_extra.special_functions import FresnelCos, FresnelSin

def gen_freq(n,dt):
    return _np.fft.rfftfreq(n,dt)
    
def gen_t(n,dt):
    if n%2 == 1 :
        return _np.r_[-(n//2):(n//2+1)]*dt
    else :
        return _np.r_[-(n//2)+0.5:(n//2+0.5)]*dt
    
@vectorize([float64(float64, float64)])
def _kp(t,dt):
    if t == 0 :
        return 2.0 * _np.sqrt(2.0/dt)
    else :
        return 2.0 / _np.sqrt(_np.abs(t)) * FresnelCos(_np.sqrt(2.0 * _np.abs(t) / dt))
        
@vectorize([float64(float64, float64)])
def _kq(t,dt):
    if t == 0 :
        return 0.0
    else :
        return _np.sign(t)* 2.0 / _np.sqrt(_np.abs(t)) * FresnelSin(_np.sqrt(2.0 * _np.abs(t) / dt))

@vectorize([float64(float64, float64)])
def _delta(t,dt):
    if t == 0 :
        return 1.0/dt ;
    else :
        return 0.0

@guvectorize([(float64[:],float64,float64[:])], '(n),()->(n)')
def kp(t,Z=50.0,res=None):
    K = _np.sqrt( 1.0/ (Z*_C.h) ) 
    dt = _np.abs(t[1]-t[0])
    res[:] = K*_kp(t[:],dt)
  
@guvectorize([(float64[:],float64,float64[:])], '(n),()->(n)')
def kq(t,Z=50.0,res=None):
    K = _np.sqrt( 1.0/ (Z*_C.h) ) 
    dt = _np.abs(t[1]-t[0])
    res[:] = K*_kq(t[:],dt)

@vectorize([float64(float64,float64,float64,float64)])
def _k_Theta(t,dt,Theta,Z=50.0):
    K = _np.sqrt( 1.0/ (Z*_C.h) ) 
    return K*( _kp(t,dt)*_np.sin(Theta) + _kq(t,dt)*_np.cos(Theta) )

@guvectorize([(float64[:],float64,float64,float64[:])], '(n),(),()->(n)')
def k_Theta(t,Theta,Z=50.0,res=None):
    """
    k_Theta(t,Theta,Z)
    
    This function is a Generalized Universal Function (gufunc)
    that calculate the time kernel with angle Theta.

    This function computes the K-Theta values based on the formula:
    K = sqrt(1.0 / (Z * C.h))
    dt = abs(t[1] - t[0])
    res[:] = K * (_kp(t[:], dt) * _np.sin(Theta) + _kq(t[:], dt) * _np.cos(Theta))

    Parameters:
    -----------
    t : 1D numpy.ndarray, float64 
        time values in second (units are important).

    Theta : float64 
        in radians.

    Z : float64, optional
        Line impedance
        
    Returns:
    --------
    res : 1D numpy.ndarray, float64
    """
    K = _np.sqrt( 1.0/ (Z*_C.h) ) 
    dt = _np.abs(t[1]-t[0])
    res[:] = _k_Theta(t[:],dt,Theta,Z)

@guvectorize([(float64[:],float64[:])], '(n)->(n)')
def delta(t,res=None):
    dt = _np.abs(t[1]-t[0])
    res[:] = _delta(t[:],dt)

@jit
def generate_a_tukey_window(ks,alpha=0.5):
    """
    just a reminder on how to use scipy's tukey
    """
    return tukey(ks.shape[-1],alpha=alpha) # A Tukey window with ks.shape[-1] points

@guvectorize([(float64[:],float64[:],float64)], '(n)->(n),()') 
def half_normalization(k,res,hn):
    """
    half_normalization(k)
    This function is designed to be used alongside 'half_denormalization.'
    Its purpose is to ensure that kernels have similar amplitudes, 
    which allows for consistent histogram bounds after convolution.
    """
    hn = _np.sqrt( (k[:]*k[:]).sum() )
    res[:] = k[:]/hn

@guvectorize([(float64[:],float64,float64[:])], '(n),()->(n)') 
def half_denormalization(k,hn,res):
    """
    half_denormalization(k,hn)
    """
    res[:] = k[:]*hn
        
def apply_filters(ks,filters):
    n = ks.shape[-1]
    return _np.fft.fftshift( _np.fft.irfft( _np.fft.rfft(_np.fft.fftshift(ks))*filters,n) )

def make_kernels(t,betas,g=None,window=True,alpha=0.5,Z=50.,Theta=0.,half_norm=True):
    ks = k_Theta(t,Theta,Z)
    if window :
        T = tukey(ks.shape[-1],alpha=alpha)
        ks = ks*T
    if g :
        filters = betas/g
    else :
        filters = betas
    ks = apply_filters(ks,filters)
    if half_norm :
        ks, hn = half_normalization(ks)
    else :
        hn = None
    return ks, hn
    


