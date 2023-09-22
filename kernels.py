#!/bin/env/python
#! -*- coding: utf-8 -*-
import numpy as _np
from scipy import constants as C
from numba import float64,complex128,vectorize,guvectorize

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
        return 2.0 / sqrt(t) * Fresnel_Cosine_Integral(sqrt(2.0 * abs(t) / dt))
        
@vectorize([float64(float64, float64)])
def _kq(t,dt):
    if t == 0 :
        return 0.0
    else :
        return sign(t)* 2.0 / sqrt(t) * Fresnel_Sine_Integral(sqrt(2.0 * abs(t) / dt))

@vectorize([float64(float64, float64)])
def _delta(t,dt):
    if t == 0 :
        return 1.0/dt ;
    else :
        return 0.0

@guvectorize([(float64[:],float64,float64,float64[:])], '(n),(),()->(n)')
def kp(t,Z=50.0,units_correction = 10.0**(-9.0/2.0),res=None):
    K = units_correction*sqrt( 1.0/ (Z*C.h) ) 
    dt = abs(t[1]-t[0])
    res[:] = K*_kp(t[:],dt)
  
@guvectorize([(float64[:],float64,float64,float64[:])], '(n),(),()->(n)')
def kq(t,Z=50.0,units_correction = 10.0**(-9.0/2.0),res=None):
    K = units_correction*sqrt( 1.0/ (Z*C.h) ) 
    dt = abs(t[1]-t[0])
    res[:] = K*_kq(t[:],dt)

@guvectorize([(float64[:],float64,float64[:])], '(n),()->(n)')
def delta(t,units_correction = 10.0**(-9.0/2.0),res=None):
    dt = abs(t[1]-t[0])
    res[:] = units_correction*_delta(t[:],dt)


    
