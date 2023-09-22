#!/bin/env/python
#! -*- coding: utf-8 -*-

"""
This Module is deprecated and will be grdually moved to betas,filters and kernels
"""

import  numpy

from SBB.Time_quadratures.time_quadratures import TimeQuad_uint64_t
from SBB.Numpy_extra.numpy_extra import find_nearest_A_to_a

from betas import normalize_beta

__version__ = {'TimeQuadrature_helper': 0.7}
__filters__ = {'gauss':0,'bigauss':1,'flatband':2}
"""
    Public
"""

def gen_t_abscisse(l_kernel,dt):
    t=numpy.arange(l_kernel/2+1)*dt
    return numpy.concatenate((-t[-1:0:-1],t))

def gen_f_abscisse(l_kernel,dt):
    return numpy.fft.rfftfreq(l_kernel,dt)

def gen_filter_info(gauss_info,bigauss_info,flatband_info):
    """
        todos :
            - make it work for any number of inputs ?
    """
    filter_info             = {'gauss':gauss_info,'bigauss':bigauss_info,'flatband':flatband_info}
    filter_info['labels']   = _gen_labels(filter_info)
    filter_info['gauss']['slice'],filter_info['bigauss']['slice'],filter_info['flatband']['slice'] = _gen_filters_slices(filter_info)
    filter_info['strides']  = _gen_filters_strides(filter_info)
    filter_info['lengths']  = _get_filters_len(filter_info)
    filter_info['length']   = numpy.array(filter_info['lengths']).sum()
    return filter_info

def gen_gauss_info(fs,dfs,snap_on=True): 
    """
        bigauss_indexes is a n by 1 array
        fs[0] = f_0, # central frequencie of the gaussian
        fs[1] = ...
        dfs.shape == fs.shape
    """
    fs  = numpy.array(fs)
    if fs.size == 0 : # the array is empty
        gauss_info = {'fs':numpy.array([]),'dfs':numpy.array([]),'snap_on':snap_on}
    else: 
        dfs = numpy.array(dfs)
        if dfs.size == 1 :  # all the same df
            dfs = dfs*numpy.ones(fs.shape)  
        gauss_info              = {'fs':fs,'dfs':dfs,'snap_on':snap_on}
    gauss_info['labels']    = _gen_gauss_labels   (gauss_info) 
    return gauss_info

def gen_bigauss_info(fs,dfs,snap_on=True):
    """
        bigauss_indexes is a n by 2 array
        fs[0,:] = f_0, f_1 # central frequencies of the first and seconde gaussian
        fs[1,:] = ...
    """
    fs  = numpy.array(fs)
    if fs.size == 0 : # the array is empty
        bigauss_info = {'fs':numpy.array([]),'dfs':numpy.array([]),'snap_on':snap_on}
    else :
        dfs = numpy.array(dfs)
        if dfs.size == 1 :  # all the same df
            dfs = dfs*numpy.ones(fs.shape)
        bigauss_info            = {'fs':fs,'dfs':dfs,'snap_on':snap_on}
    bigauss_info['labels']  = _gen_bigauss_labels (bigauss_info)
    return bigauss_info

def gen_flatband_info(fs,rise,fall,snap_on=True):
    """
        fs[0,:]         = f_0, f_1 # central frequencies of the first and seconde gaussian
        fs[1,:]         = ...
        rise_fall[0,:]  = rise, fall
        rise_fall[1,:]  = ...
    """
    fs  = numpy.array(fs)
    if fs.size == 0 : # the array is empty
        flatband_info = {'fs':numpy.array([]),'rise_fall':numpy.array([]),'snap_on':snap_on}
    else :
        rise_fall = numpy.zeros(fs.shape)
        rise_fall[:,0] = rise
        rise_fall[:,1] = fall
        flatband_info  = {'fs':fs,'rise_fall':rise_fall,'snap_on':snap_on}
    flatband_info['labels'] = _gen_flatband_labels(flatband_info)
    return flatband_info

def gen_Filters(l_kernel,dt,filter_info):
    """
        todos :
            - make it work for any number of inputs ?
    """
    Filters_gauss                           = gen_gauss_Filters   (l_kernel,dt,filter_info['gauss']   )
    Filters_bigauss                         = gen_bigauss_Filters (l_kernel,dt,filter_info['bigauss'] )
    Filter_flatband                         = gen_flatband        (l_kernel,dt,filter_info['flatband'])
    return _concatenate_Filters(Filters_gauss,Filters_bigauss,Filter_flatband)

def gen_gauss_Filters(l_kernel,dt,gauss_info):
    fs , dfs    = _extract_gauss_info(gauss_info)
    snap_on     = _checks_snap_on(**gauss_info)
    if fs.size == 0 :
        return numpy.array([])
    if snap_on :
        F   = gen_f_abscisse(l_kernel,dt)
        fs,_  = find_nearest_A_to_a(fs,F)
        gauss_info['fs'] = fs # Modifying the dict
    Filters = numpy.empty( (len(fs),l_kernel//2+1) , dtype=complex , order='C' ) 
    for i,(f,df) in enumerate(zip(fs,dfs)):
        Filters[i,:] = Gaussian_filter_normalized( f , df , l_kernel, dt )
    return Filters  

def gen_bigauss_Filters(l_kernel,dt,bigauss_info):
    fs , dfs    = _extract_bigauss_info(bigauss_info)
    snap_on     = _checks_snap_on(**bigauss_info)
    if fs.size == 0 :
        return numpy.array([])
    if fs.shape[1] !=2 :
        raise Exception('bigauss_indexes needs to be n by 2.')
    if snap_on :
        F   = gen_f_abscisse(l_kernel,dt)
        fs,_  = find_nearest_A_to_a(fs,F)
        bigauss_info['fs'] = fs # Modifying the dict
    Filters =  numpy.empty( (fs.shape[0],l_kernel//2+1) , dtype=complex , order='C' ) 
    for i,(f,df) in enumerate(zip(fs,dfs)) :
        Filters[i,:] = _Bi_Gaussian_filter_normalized(f[0],f[1],df[0],df[1],l_kernel,dt) 
    return Filters

def gen_flatband(l_kernel,dt,flatband_info):
    l_hc            = l_kernel//2+1
    fs,rise_fall    = _extract_flatband_info(flatband_info)
    if fs.size ==0 :
        return numpy.array([])
    Filters     = numpy.empty( (fs.shape[0],l_hc),dtype=complex,order='C' ) 
    for i,(flat,r_f) in enumerate(zip(fs,rise_fall)) :  
        Filters[i,:]    = TimeQuad_uint64_t.compute_flatband(l_hc,dt,flat[0]-r_f[0],flat[0],flat[1],flat[1]+r_f[1])
    return Filters

def gen_composition_indexes(filters_info,composition):
    """
        A composition has shape m,2,n
        m : composition index
        n : combinations index
        the 2nd index is for type and subindex
    """
    filter_type_indexes = composition[:,0,:]
    filter_index        = composition[:,1,:]
    strides             = filter_info['strides'] 
    kernel_indexes      = numpy.zeros(filter_index.shape)
    for i,stride in enumerate(strides):
        kernel_indexes[numpy.where(filter_type_indexes==i)] = stride
    kernel_indexes += filter_index
    return kernel_indexes.astype(int) 

def Gaussian (x,mu=0.0,sigma=1.0) :
    return (1.0/(sigma*numpy.sqrt(2.0*numpy.pi))) * numpy.exp( (-(x-mu)**2)/(2.0*sigma**2) )

def Gaussian_filter_normalized(f,df,l_kernel,dt) :
    """
    Returns a numpy array of complex number corresponding to a gaussian filter
    of avg f and std dev df on positive frequencies and with vector length equal to  l_kernel//2 + 1.
    """
    l_hc = l_kernel//2+1 

    Y = numpy.empty( l_hc , dtype = complex , order='C') 
    x_f = numpy.fft.rfftfreq(l_kernel , dt)
    for i in range( l_hc ) :
        Y[i] =  Gaussian ( x_f[i] , f , df ) 
    Delta_f = x_f[1]-x_f[0]
    Y = Wave_function_of_f_normalization(Y,Delta_f)
    return Y 
    
########################################
# To help with the experiment conception
########################################

def _checks_snap_on(**options):
    return options['snap_on'] if 'snap_on'  in options else True

def _extract_filter_info(filter_info):
    return filter_info['gauss'],filter_info['bigauss'],filter_info['flatband'],

def _extract_gauss_info(gauss_info): 
    return gauss_info['fs'] , gauss_info['dfs']

def _extract_bigauss_info(bigauss_info):
    return _extract_gauss_info(bigauss_info)

def _extract_flatband_info(flatband_info): 
    return flatband_info['fs'],flatband_info['rise_fall'] 

def _get_filters_len(filter_info):
    gauss_info,bigauss_info,flatband_info = _extract_filter_info(filter_info)
    fs_g,_          =   _extract_gauss_info(gauss_info)
    fs_bg,_         =   _extract_bigauss_info(bigauss_info)
    fs_fb,_         =   _extract_flatband_info(flatband_info)
    return fs_g.shape[0],fs_bg.shape[0],fs_fb.shape[0]

def _gen_filters_strides(filter_info):
    l_g,l_bg,l_fb       = _get_filters_len(filter_info)
    gauss_stride        = 0
    bigauss_stride      = gauss_stride   + l_g
    flatband_stride     = bigauss_stride + l_bg
    return (gauss_stride,bigauss_stride,flatband_stride)

def _gen_filters_slices(filter_info):
    l_g,l_bg,l_fb   =   _get_filters_len(filter_info) 
    gauss_slice     =   slice(None        ,l_g            ,None)
    bigauss_slice   =   slice(l_g         ,l_g+l_bg       ,None)
    flatband_slice  =   slice(l_g+l_bg    ,l_g+l_bg+l_fb  ,None)
    return gauss_slice,bigauss_slice,flatband_slice

def _gen_labels(filter_info):
    gauss_info,bigauss_info,flatband_info = _extract_filter_info(filter_info)
    return gauss_info['labels'] + bigauss_info['labels'] + flatband_info['labels']

def _gen_gauss_labels(gauss_info,label_frmt="{:0.3f}"):
    fs , dfs    = _extract_gauss_info(gauss_info)
    labels = []
    for (f,df) in zip(fs,dfs) :
        label = label_frmt.format(f)
        labels.append(label)
    return labels

def _gen_bigauss_labels(bigauss_info,label_frmt="{:0.3f}&{:0.3f}"):
    fs , dfs    = _extract_bigauss_info(bigauss_info)
    labels = []
    for (f,df) in zip(fs,dfs) :
        label = label_frmt.format(f[0],f[1])
        labels.append(label)
    return labels

def _gen_flatband_labels(flatband_info,label_frmt="{:0.3f}-{:0.3f}"):
    fs,_ =_extract_flatband_info(flatband_info)
    labels = []
    for f in fs :
        label = label_frmt.format(f[0],f[1])
        labels.append(label)
    return labels

def _Bi_Gaussian_filter_normalized(f1,f2,df1,df2,l_kernel,dt) :
    l_hc = l_kernel//2+1 
    Y = numpy.empty( l_hc , dtype = complex , order='C') 
    x_f = numpy.fft.rfftfreq(l_kernel , dt)
    for i in range( l_hc ) :
        Y[i] =  (df1*numpy.sqrt(2.0*numpy.pi))*Gaussian ( x_f[i] , f1 , df1 ) + (df2*numpy.sqrt(2.0*numpy.pi))*Gaussian(x_f[i] , f2 , df2) 
    Delta_f = (x_f[1]-x_f[0])    
    Y = Wave_function_of_f_normalization(Y,Delta_f)
    return Y   

def _concatenate_Filters(*args):
    t = tuple()
    for arg in args :
        if not (arg.size==0) :
            t += (arg,)
    return numpy.concatenate( t, axis = 0 ) 
