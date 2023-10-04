#!/bin/env/python
#! -*- coding: utf-8 -*-
import numpy as _np
from scipy import constants as C
from numba import float64,complex128,vectorize,guvectorize
from ..Math_extra.special_functions import FresnelCos, FresnelSin

def compute_n_quads(self, kernel_conf):
    if kernel_conf == 0:
        return 1  # q seulement
    elif kernel_conf == 1:
        return 2  # p et q
    # elif kernel_conf == 2:
    #     return 2  # pi/4 et 3pi/4
    elif kernel_conf == 3:
        return 1  # Pas de kernel (i.e. ones )
    else:
        raise RuntimeError("Invalid kernel_conf")  # Invalid kernel_conf

def make_kernels(betas,):
{
	betas = normalize_betas(betas); # Can be done first because only depend on betas 
    
	kernels = vanilla_kernels(kernel_conf);
	normalize_for_dfts(); 
	apply_windows();
	compute_filters();
	apply_filters();
	
    half_normalization();
}

def vanilla_kernels(kernel_conf):
    if kernel_conf == 0:
        for i in range(self.n_kernels):
            vanilla_kq(0, i)  # quadrature_index , mode_index
    elif kernel_conf == 1:
        for i in range(self.n_kernels):
            vanilla_kp(0, i)  # quadrature_index , mode_index
            vanilla_kq(1, i)  # quadrature_index , mode_index
    # elif kernel_conf == 2:
    #     for i in range(self.n_kernels):
    #         vanilla_k_pi_over_4(0, i)  # quadrature_index , mode_index
    #         vanilla_k_3_pi_over_4(0, i)  # quadrature_index , mode_index
    elif kernel_conf == 3:
        for i in range(self.n_kernels):
            delta(0, i)  # quadrature_index , mode_index
    else:
        raise RuntimeError("Invalid kernel_conf")
    return kernels

@vectorize([float64(float64, float64)])
def _kp(t,dt):
    if t == 0 :
        return 2.0 * sqrt(2.0/dt)
    else :
        return 2.0 / sqrt(abs(t)) * FresnelCos(sqrt(2.0 * abs(t) / dt))
        
@vectorize([float64(float64, float64)])
def _kq(t,dt):
    if t == 0 :
        return 0.0
    else :
        return sign(t)* 2.0 / sqrt(abs(t)) * FresnelSin(sqrt(2.0 * abs(t) / dt))

@vectorize([float64(float64, float64)])
def _delta(t,dt):
    if t == 0 :
        return 1.0/dt ;
    else :
        return 0.0

@guvectorize([(float64[:],float64,float64[:])], '(n),(),()->(n)')
def kp(t,Z=50.0,res=None):
    K = sqrt( 1.0/ (Z*C.h) ) 
    dt = abs(t[1]-t[0])
    res[:] = K*_kp(t[:],dt)
  
@guvectorize([(float64[:],float64,float64[:])], '(n),()->(n)')
def kq(t,Z=50.0,res=None):
    K = sqrt( 1.0/ (Z*C.h) ) 
    dt = abs(t[1]-t[0])
    res[:] = K*_kq(t[:],dt)

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
    K = sqrt( 1.0/ (Z*C.h) ) 
    dt = abs(t[1]-t[0])
    res[:] = K*( _kp(t[:],dt)*_np.sin(Theta) + _kq(t[:],dt)*_np.cos(Theta) )

@guvectorize([(float64[:],float64[:])], '(n)->(n)')
def delta(t,res=None):
    dt = abs(t[1]-t[0])
    res[:] = _delta(t[:],dt)

def generate_a_tukey_window(ks,alpha=0.5):
    """
    just a reminder on how to use scipy's tukey
    """
    from scipy.signal.windows import tukey
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
        
@guvectorize([(float64[:],float64,float64[:])], '(n),()->(n)') 
def apply_filters(ks,filters):
    ks_f = _np.fft.rfft(ks)
    ks_f = ks_f*filters
    _np.fft.irfft(ks_f,ks.shape[-1])
    return _np.fft.irfft(ks_f,ks.shape[-1])
    
@guvectorize([(float64[:],float64,float64,float64[:])], '(n),(),()->(n),()')  
def make_kernels(t,betas,g,Z=50.,Theta=0,window=None,half_norm=True,):
    ks = k_Theta(t,Theta,Z)
    if window :
        ks = ks*T
    ks = apply_filters(ks,betas/g)
    if half_norm :
        ks, hn = half_normalization(ks)
    else :
        hn = None
    return ks, hn
    

