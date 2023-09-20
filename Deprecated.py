#!/bin/env/python
#! -*- coding: utf-8 -*-

from SBB.Pyhegel_extra.Experiment import Info, Lagging_computation, Analysis
from SBB.Pyhegel_extra.Deprecated import logger_acq_and_compute
from SBB.Pyhegel_extra.Deprecated import Conditions_logic,Three_points_polarisation
from SBB.Numpy_extra.numpy_extra import find_nearest_A_to_a
from SBB.Time_quadratures.time_quadratures import TimeQuad_uint64_t

import numpy

class Quads_helper(object):
    """
        Last update
        -----------
            Moved some info generator into Experiment_helper
        Todos : 
            - Simplify the filter info process
        Bugs :
        
    """
    __filters__ = {'gauss':0,'bigauss':1,'flatband':2}
    """
        Public
    """
    @staticmethod
    def gen_t_abscisse(l_kernel,dt):
        t=numpy.arange(l_kernel/2+1)*dt
        return numpy.concatenate((-t[-1:0:-1],t))
    @staticmethod
    def gen_f_abscisse(l_kernel,dt):
        return numpy.fft.rfftfreq(l_kernel,dt)
    @staticmethod
    def gen_filter_info(gauss_info,bigauss_info,flatband_info):
        """
            todos :
                - make it work for any number of inputs ?
        """
        filter_info             = {'gauss':gauss_info,'bigauss':bigauss_info,'flatband':flatband_info}
        filter_info['labels']   = Quads_helper._gen_labels(filter_info)
        filter_info['gauss']['slice'],filter_info['bigauss']['slice'],filter_info['flatband']['slice'] = Quads_helper._gen_filters_slices(filter_info)
        filter_info['strides']  = Quads_helper._gen_filters_strides(filter_info)
        filter_info['lengths']  = Quads_helper._get_filters_len(filter_info)
        filter_info['length']   = numpy.array(filter_info['lengths']).sum()
        return filter_info
    @staticmethod
    def gen_gauss_info(fs,dfs,snap_on=True): 
        """
            bigauss_indexes is a n by 1 array
            fs[0] = f_0, # central frequencie of the gaussian
            fs[1] = ...
            dfs.shape == fs.shape
        """
        fs  = numpy.array(fs)
        if fs.size == 0 : # the array is empty
            return {'fs':numpy.array([]),'dfs':numpy.array([]),'snap_on':snap_on}
        else: 
            dfs = numpy.array(dfs)
            if dfs.size == 1 :  # all the same df
                dfs = dfs*numpy.ones(fs.shape)  
            gauss_info              = {'fs':fs,'dfs':dfs,'snap_on':snap_on}
        gauss_info['labels']    = Quads_helper._gen_gauss_labels   (gauss_info) 
        return gauss_info
    @staticmethod
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
        bigauss_info['labels']  = Quads_helper._gen_bigauss_labels (bigauss_info)
        return bigauss_info
    @staticmethod
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
        flatband_info['labels'] = Quads_helper._gen_flatband_labels(flatband_info)
        return flatband_info
    @staticmethod
    def gen_Filters(l_kernel,dt,filter_info):
        """
            todos :
                - make it work for any number of inputs ?
        """
        Filters_gauss                           = Quads_helper.gen_gauss_Filters   (l_kernel,dt,filter_info['gauss']   )
        Filters_bigauss                         = Quads_helper.gen_bigauss_Filters (l_kernel,dt,filter_info['bigauss'] )
        Filter_flatband                         = Quads_helper.gen_flatband        (l_kernel,dt,filter_info['flatband'])
        return Quads_helper._concatenate_Filters(Filters_gauss,Filters_bigauss,Filter_flatband)
    @staticmethod
    def gen_gauss_Filters(l_kernel,dt,gauss_info):
        fs , dfs    = Quads_helper._extract_gauss_info(gauss_info)
        snap_on     = Quads_helper._checks_snap_on(**gauss_info)
        if fs.size == 0 :
            return numpy.array([])
        if snap_on :
            F   = Quads_helper.gen_f_abscisse(l_kernel,dt)
            fs,_  = find_nearest_A_to_a(fs,F)
            gauss_info['fs'] = fs # Modifying the dict
        Filters = numpy.empty( (len(fs),l_kernel//2+1) , dtype=complex , order='C' ) 
        for i,(f,df) in enumerate(zip(fs,dfs)):
            Filters[i,:] = Quads_helper.Gaussian_filter_normalized( f , df , l_kernel, dt )
        return Filters  
    @staticmethod
    def gen_bigauss_Filters(l_kernel,dt,bigauss_info):
        fs , dfs    = Quads_helper._extract_bigauss_info(bigauss_info)
        snap_on     = Quads_helper._checks_snap_on(**bigauss_info)
        if fs.shape[1] !=2 :
            raise Exception('bigauss_indexes needs to be n by 2.')
        if fs.size == 0 :
            return numpy.array([])
        if snap_on :
            F   = Quads_helper.gen_f_abscisse(l_kernel,dt)
            fs,_  = find_nearest_A_to_a(fs,F)
            bigauss_info['fs'] = fs # Modifying the dict
        Filters =  numpy.empty( (fs.shape[0],l_kernel//2+1) , dtype=complex , order='C' ) 
        for i,(f,df) in enumerate(zip(fs,dfs)) :
            Filters[i,:] = Quads_helper._Bi_Gaussian_filter_normalized(f[0],f[1],df[0],df[1],l_kernel,dt) 
        return Filters
    @staticmethod
    def gen_flatband(l_kernel,dt,flatband_info):
        l_hc            = l_kernel//2+1
        fs,rise_fall    = Quads_helper._extract_flatband_info(flatband_info)
        if fs.size ==0 :
            return numpy.array([])
        Filters     = numpy.empty( (fs.shape[0],l_hc),dtype=complex,order='C' ) 
        for i,(flat,r_f) in enumerate(zip(fs,rise_fall)) :  
            Filters[i,:]    = TimeQuad_uint64_t.compute_flatband(l_hc,dt,flat[0]-r_f[0],flat[0],flat[1],flat[1]+r_f[1])
        return Filters
    """
        Private
    """
    @staticmethod
    def _checks_snap_on(**options):
        return options['snap_on'] if 'snap_on'  in options else True
    @staticmethod
    def _extract_filter_info(filter_info):
        return filter_info['gauss'],filter_info['bigauss'],filter_info['flatband'],
    @staticmethod
    def _extract_gauss_info(gauss_info): 
        return gauss_info['fs'] , gauss_info['dfs']
    @staticmethod
    def _extract_bigauss_info(bigauss_info):
        return Quads_helper._extract_gauss_info(bigauss_info)
    @staticmethod
    def _extract_flatband_info(flatband_info): 
        return flatband_info['fs'],flatband_info['rise_fall'] 
    @staticmethod
    def _get_filters_len(filter_info):
        gauss_info,bigauss_info,flatband_info = Quads_helper._extract_filter_info(filter_info)
        fs_g,_          =   Quads_helper._extract_gauss_info(gauss_info)
        fs_bg,_         =   Quads_helper._extract_bigauss_info(bigauss_info)
        fs_fb,_         =   Quads_helper._extract_flatband_info(flatband_info)
        return fs_g.shape[0],fs_bg.shape[0],fs_fb.shape[0]
    @staticmethod
    def _gen_filters_strides(filter_info):
        l_g,l_bg,l_fb       = Quads_helper._get_filters_len(filter_info)
        gauss_stride        = 0
        bigauss_stride      = gauss_stride   + l_g
        flatband_stride     = bigauss_stride + l_bg
        return (gauss_stride,bigauss_stride,flatband_stride)
    @staticmethod
    def _gen_filters_slices(filter_info):
        l_g,l_bg,l_fb   =   Quads_helper._get_filters_len(filter_info) 
        gauss_slice     =   slice(None        ,l_g            ,None)
        bigauss_slice   =   slice(l_g         ,l_g+l_bg       ,None)
        flatband_slice  =   slice(l_g+l_bg    ,l_g+l_bg+l_fb  ,None)
        return gauss_slice,bigauss_slice,flatband_slice
    @staticmethod
    def _gen_labels(filter_info):
        gauss_info,bigauss_info,flatband_info = Quads_helper._extract_filter_info(filter_info)
        return gauss_info['labels'] + bigauss_info['labels'] + flatband_info['labels']
    @staticmethod
    def _gen_gauss_labels(gauss_info,label_frmt="{:0.1f}"):
        fs , dfs    = Quads_helper._extract_gauss_info(gauss_info)
        labels = []
        for (f,df) in zip(fs,dfs) :
            label = label_frmt.format(f)
            labels.append(label)
        return labels
    @staticmethod
    def _gen_bigauss_labels(bigauss_info,label_frmt="{:0.1f}&{:0.1f}"):
        fs , dfs    = Quads_helper._extract_bigauss_info(bigauss_info)
        labels = []
        for (f,df) in zip(fs,dfs) :
            label = label_frmt.format(f[0],f[1])
            labels.append(label)
        return labels
    @staticmethod
    def _gen_flatband_labels(flatband_info,label_frmt="{:0.1f}-{:0.1f}"):
        fs,_ =Quads_helper._extract_flatband_info(flatband_info)
        labels = []
        for f in fs :
            label = label_frmt.format(f[0],f[1])
            labels.append(label)
        return labels
    @staticmethod
    def _gen_composition_indexes(filters_info,composition):
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
    @staticmethod
    def Wave_function_of_f_normalization(Y,df):
        """
            Note that betas are given to TimeQuad c++ class are 
            normalized internally in construction and are accessible
            through TimeQuad's attributes.
            This function is for conveniance.
        """
        sum = numpy.sqrt( 2*df*(numpy.square(numpy.abs(Y))).sum() )
        return Y/(sum)
    @staticmethod
    def Gaussian (x,mu=0.0,sigma=1.0) :
        return (1.0/(sigma*numpy.sqrt(2.0*numpy.pi))) * numpy.exp( (-(x-mu)**2)/(2.0*sigma**2) )
    @staticmethod
    def Gaussian_filter_normalized(f,df,l_kernel,dt) :
        """
        Returns a numpy array of complex number corresponding to a gaussian filter
        of avg f and std dev df on positive frequencies and with vector length equal to  l_kernel//2 + 1.
        """
        l_hc = l_kernel//2+1 

        Y = numpy.empty( l_hc , dtype = complex , order='C') 
        x_f = numpy.fft.rfftfreq(l_kernel , dt)
        for i in range( l_hc ) :
            Y[i] =  Quads_helper.Gaussian ( x_f[i] , f , df ) 
        Delta_f = x_f[1]-x_f[0]
        Y = Quads_helper.Wave_function_of_f_normalization(Y,Delta_f)
        return Y 
    @staticmethod
    def _Bi_Gaussian_filter_normalized(f1,f2,df1,df2,l_kernel,dt) :
        l_hc = l_kernel//2+1 
        Y = numpy.empty( l_hc , dtype = complex , order='C') 
        x_f = numpy.fft.rfftfreq(l_kernel , dt)
        for i in range( l_hc ) :
            Y[i] =  (df1*numpy.sqrt(2.0*numpy.pi))*Quads_helper.Gaussian ( x_f[i] , f1 , df1 ) + (df2*numpy.sqrt(2.0*numpy.pi))*Quads_helper.Gaussian(x_f[i] , f2 , df2) 
        Delta_f = (x_f[1]-x_f[0])    
        Y = Quads_helper.Wave_function_of_f_normalization(Y,Delta_f)
        return Y   
    @staticmethod
    def _concatenate_Filters(*args):
        t = tuple()
        for arg in args :
            if not (arg.size==0) :
                t += (arg,)
        return numpy.concatenate( t, axis = 0 ) 
    @staticmethod
    def _moments_correction(moments,half_norms,powers):
        """
            Correcting for half normalization
            
            moments     .shape should be  (moment_index,kernel_index,...)
            half_norms  .shape should be  (kernel_index)
            powers      .shape should be  (moment_index)
        """
        powers      = numpy.array(powers)       # moment_index
        h           = numpy.array(half_norms)   # kernel index 
        shape       = moments.shape
        dim         = len(shape)
        corrections  = (h[None,:]**powers[:,None])   # moment_index , kernel_index
        exp_axis = tuple(range(2,dim)) 
        for ax in exp_axis :
            corrections = numpy.expand_dims(corrections,ax)         # shape now match moments shape
        moments_corrected = numpy.empty(moments.shape,dtype=float)  # moment_index, kernel index , cdn index
        moments_corrected = corrections * moments 
        return moments_corrected 
 
class QsVsVdc_info(Info,Quads_helper):
    """
    Last update
    -----------
        Moved some gen info into Experiement helper
        Removed unncessary variables and methods 
    """
    powers      = numpy.array([1,2,4,8])
    @staticmethod
    def gen_meta_info(circuit_info,compute_info,aqc_info,quads_info,hist_info):
        return {'circuit_info':circuit_info,'compute_info':compute_info,'aqc_info':aqc_info,'quads_info':quads_info,'hist_info':hist_info}   
    def _set_options(self,options):
        super(QsVsVdc_info,self)._set_options(options)
        Conditions_logic._set_options(self,**options)   #stactic
    def _set_conditions(self,conditions):
        super(QsVsVdc_info,self)._set_conditions(conditions)
        self.Vdc    =   self._conditions_core_loop_raw[0]   
    def _build_attributes(self):
        super(QsVsVdc_info,self)._build_attributes()
        Conditions_logic._build_attributes(self)  
        self._R_jct         = self._meta_info['circuit_info']['R_jct']
        self._R_1M          = self._meta_info['circuit_info']['R_1M']
        self._R_tl          = self._meta_info['circuit_info']['R_tl']
        self._g             = self._meta_info['circuit_info']['g']
        self._V_th          = self._meta_info['circuit_info']['V_th']
        
        self._n_threads     = self._meta_info['compute_info']['n_threads']
        self._l_fft         = self._meta_info['compute_info']['l_fft']
        
        self._l_data        = int(self._meta_info['aqc_info']['l_data'])
        self._dt            = self._meta_info['aqc_info']['dt']
        
        self._l_kernel      = int(self._meta_info['quads_info']['l_kernel'])
        self._l_hc          = self._l_kernel/2 + 1   
        self._alpha         = self._meta_info['quads_info']['alpha']
        self._kernel_conf   = self._meta_info['quads_info']['kernel_conf']
        
        filter_info                  = self._meta_info['quads_info']['filters_info']
        self._filter_info            = filter_info
        self._n_kernels              = filter_info['length'] # alias
        self._labels                 = self._filter_info['labels']   # alias
        self._filter_info['Filters'] = Quads_helper.gen_Filters(self._l_kernel,self._dt,filter_info)
        self._Filters                = self._filter_info['Filters'] # alias
          
        self._nb_of_bin = int(self._meta_info['hist_info']['nb_of_bin'])
        self._max       = self._meta_info['hist_info']['max']
    #################
    # Loop behavior #
    #################
    def _core_loop_iterator(self):
        return Conditions_logic._core_loop_iterator(self)

class QsVsVdc_fig(object):
    def __init__(self, exp):
        self.exp = exp
    #############
    # utilities #
    #############
    @staticmethod
    def polyfit_of_n(V_jct,n,V_th,deg=1):
        """
        Parameters
        ----------
        V_jct : array_like, shape(M) or (N,M) 
        n     : array_like, shape(N,M) 
        V_th  : array_like, shape(N)
        deg   : int
            Degree of the fitting polynomial
        Returns
        -------
        P     : ndarray, shape(N,deg+1)
        """
        N           = n.shape[0]
        M           = n.shape[1]
        V_th        = numpy.array([V_th]) if type(V_th) is float or type(V_th)is  numpy.float64 else numpy.array(V_th) 
        Vth         = numpy.zeros((N,))
        Vth[...]    = V_th
        Vjct        = numpy.zeros((N,M))
        Vjct[...]   = V_jct
        I_pos       = Vjct>=Vth[:,None]
        I_neg       = Vjct<=-Vth[:,None]
        P  = numpy.zeros((2,N,deg+1))
        for j,(v,nn,i_pos,i_neg) in enumerate(zip(Vjct,n,I_pos,I_neg)):
            P[0,j]   = numpy.polyfit(v[i_pos],nn[i_pos],deg)
            P[1,j]   = numpy.polyfit(v[i_neg],nn[i_neg],deg)
        return P
    ######################
    # for experiment #
    ######################
    def hist(self,V_slice=slice(None),kernel_slice=slice(None)):
        exp = self.exp
        fig = figure()
        ax = gca()
        kernel_indexes  = range(exp._n_kernels)[kernel_slice]
        Vdc             = exp._Vdc_exp
        V_indexes       = range(len(Vdc))[V_slice]
        abscisse        = Histogram_uint64_t_double.abscisse(exp._max,exp._nb_of_bin)
        k_labels        = exp._labels
        k_linstyle      = ['solid','dotted','dashed','dashdot','solid','dotted','dashed','dashdot']
        for k in kernel_indexes :
            linestyle = k_linstyle[k]
            for v in V_indexes :
                ax.plot(abscisse,exp.hs[k,v],linestyle=linestyle,label='{:s}[GHz] ; Vdc = {: 0.1f}[V] '.format(k_labels[k],Vdc[v]))
        ax.title.set_text('Histogram of the half-normalized quadratures')
        ax.set_ylabel('Counts')
        ax.set_xlabel('Bins [units of q]')
        ax.legend()
        return fig
    def _qs(self,moment_index,kernel_slice=slice(None),errorbar=False,capsize=2.5,display_in_order=False):
        exp = self.exp
        y_labels        = ('<q>','<q**2>','<q**4>','<q**8>')
        labels          = exp._labels
        I_jct           = Three_points_polarisation.compute_I_sample(exp._Vdc_exp,exp._R_1M)
        moments         = exp.moments
        moments_std     = exp.moments_std
        plot_kw = {'linestyle':'-'}
        fig , axs = subplots(1,1)
        for i, label in enumerate(labels[kernel_slice]):
            plot_kw['label'] = label
            if errorbar : 
                plot_kw['capsize'] = capsize
                if display_in_order :
                    axs.errorbar(range(len(I_jct)),moments[moment_index,i,:],yerr=moments_std[moment_index,i,:],**plot_kw)
                else :
                    axs.errorbar(I_jct*10**6,moments[moment_index,i,:],yerr=moments_std[moment_index,i,:],**plot_kw)
            else        : 
                if display_in_order :
                    axs.plot(range(len(I_jct)),moments[moment_index,i,:],**plot_kw) 
                else : 
                    axs.plot(I_jct*10**6,moments[moment_index,i,:],**plot_kw) 
        axs.legend()
        axs.title.set_text('Moment of the total signal')
        axs.set_ylabel(y_labels[moment_index])
        s_x = 'Experimental order [ 1st --> last ]' if display_in_order else 'I_jct [uA]'
        axs.set_xlabel(s_x)
        return fig
    def q  (self,kernel_slice=slice(None),errorbar=False,capsize=2.5,display_in_order=False):
        fig = self._qs(0,kernel_slice,errorbar,capsize,display_in_order)
        return fig
    def q2 (self,kernel_slice=slice(None),errorbar=False,capsize=2.5,display_in_order=False):
        fig = self._qs(1,kernel_slice,errorbar,capsize,display_in_order)
        return fig
    def q4 (self,kernel_slice=slice(None),errorbar=False,capsize=2.5,display_in_order=False):
        fig = self._qs(2,kernel_slice,errorbar,capsize,display_in_order)
        return fig
    def q8 (self,kernel_slice=slice(None),display_in_order=False):
        fig = self._qs(3,kernel_slice,False,display_in_order=display_in_order)     
        return fig
    def filters(self):
        exp = self.exp
        fig, ax = subplots(1,1)
        Quads_helper_fig.plot_Filters(exp.filters,exp._labels,ax,l_dft = exp._l_kernel, dt=exp._dt)
        ax.set_title('betas*1/g(f)')
        ax.set_ylabel('[GHz^-1/2]/[~]')
        return fig  
    def betas(self):
        exp = self.exp
        fig, ax = subplots(1,1)
        Quads_helper_fig.plot_Filters(exp.betas,exp._labels,ax,l_dft = exp._l_kernel, dt=exp._dt)
        ax.set_title('beta(f)')
        ax.set_ylabel('[GHz^-1/2]')
        return fig
    def kernels_t(self):
        exp = self.exp
        fig, axs = subplots(1,1)
        ts = Quads_helper.gen_t_abscisse(exp._l_kernel,exp._dt) 
        Quads_helper_fig.plot_Kernels(ts,exp.ks,exp._labels,axs,exp._dt)
        fig.suptitle('half-normalized kernels')
        return fig 
    def kernels_f(self):
        exp = self.exp
        fig, axs = subplots(1,1)
        freqs = Quads_helper.gen_f_abscisse(exp._l_kernel,exp._dt) 
        Quads_helper_fig.plot_Kernels_FFT(freqs,exp.ks,exp._labels,axs,exp._dt)
        fig.suptitle('half-normalized kernels')
        return fig
    ######################
    # for analysis #
    ######################
    @staticmethod
    def _plot_n(I_jct,ns,ns_std,kernel_index,axs,errorbar=True,**plot_kw):
        if errorbar:
            plot_kw['capsize'] = 2.5
            axs.errorbar(I_jct*1.0e6,ns[0,kernel_index,:],yerr=ns_std[0,kernel_index,:],**plot_kw) 
        else :
            axs.plot(I_jct*1.0e6,ns[0,kernel_index,:],**plot_kw) 
        axs.title.set_text('<n> for the sample')
        axs.set_ylabel('Photon/mode')
        axs.set_xlabel('I_jct [uA]')
    @staticmethod
    def _plot_n2(I_jct,ns,ns_std,kernel_index,axs,errorbar=True,**plot_kw):
        if errorbar:
            plot_kw['capsize'] = 2.5
            axs.errorbar(I_jct*10**6,ns[1,kernel_index,:],yerr=ns_std[1,kernel_index,:],**plot_kw)
        else :
            axs.plot(I_jct*10**6,ns[1,kernel_index,:],**plot_kw)
        axs.title.set_text('<n**2> for the sample')
        axs.set_xlabel('I_jct [uA]')
    @staticmethod
    def _plot_dn2(I_jct,ns,ns_std,kernel_index,axs,errorbar=True,**plot_kw):
        if errorbar:
            plot_kw['capsize'] = 2.5
            axs.errorbar(I_jct*10**6,ns[2,kernel_index,:],ns_std[2,kernel_index,:],**plot_kw)
        else :
            axs.plot(I_jct*10**6,ns[2,kernel_index,:],**plot_kw)
        axs.set_ylabel('Var(n) [~]')
        axs.set_xlabel('I_jct [uA]')
    @staticmethod
    def _plot_dn2_vs_n(ns,ns_std,kernel_index,axs,errorbar=True,**plot_kw):
        plot_kw_2 = {'linestyle':'--','marker':'*','label':"n(n+1)"}
        if errorbar:
            plot_kw['capsize'] = 2.5
            axs.errorbar(ns[0,kernel_index,:],ns[2,kernel_index,:],ns_std[2,kernel_index,:],**plot_kw)
        else :
            axs.plot(ns[0,kernel_index,:],ns[2,kernel_index,:],**plot_kw)
        n = ns[0,kernel_index,:]
        axs.plot(n,n**2+n,**plot_kw_2)
        axs.legend()
        axs.set_xlabel('<n>')
        axs.set_ylabel('Var(n)')
    @staticmethod
    def _plot_diff_n_0_2_dn2(ns,ns_std,kernel_index,axs,errorbar=True,**plot_kw):
        axs.plot(ns[0,kernel_index,:],ns[0,kernel_index,:]**2-ns[2,kernel_index,:],**plot_kw)
        axs.legend()
        axs.set_xlabel('<n>')
    @staticmethod
    def _plot_sum_of_n(I_jct,ns,ns_std,i,j,axs,errorbar=True,**plot_kw):
        n1 = ns[0,i,:]
        n2 = ns[0,j,:]
        sum = (0.5)*(n1+n2)
        n1_std = ns_std[0,i,:]
        n2_std = ns_std[0,j,:]
        sum_std = (0.5)*(n1_std+n2_std)
        if errorbar:
            plot_kw['capsize'] = 2.5
            axs.errorbar(I_jct*10**6,sum,yerr=sum_std,**plot_kw) 
        else :
            axs.plot(I_jct*10**6,sum,**plot_kw) 
        axs.title.set_text('<n> for the sample')
        axs.set_ylabel('Photon/mode')
        axs.set_xlabel('I_jct [uA]')
    def _cumulants_sample(self,cumulant_index,kernel_slice=slice(None),errorbar=False,capsize=2.5):
        exp = self.exp
        y_labels        = ('<<q>>','<<q**2>>','<<q**4>>')
        labels          = exp._labels
        I_jct           = Three_points_polarisation.compute_I_sample(exp._Vdc_antisym,exp._R_1M)
        cumulants       = exp.cumulants_sample
        cumulants_std   = exp.cumulants_sample_std
        plot_kw = {'linestyle':'-'}
        fig , axs = subplots(1,1)
        for i, label in enumerate(labels[kernel_slice]):
            plot_kw['label'] = label 
            if errorbar : 
                plot_kw['capsize'] = capsize
                axs.errorbar(I_jct*10**6,cumulants[cumulant_index,i,:],yerr=cumulants_std[cumulant_index,i,:],**plot_kw) 
            else        : 
                axs.plot(I_jct*10**6,cumulants[cumulant_index,i,:],**plot_kw) 
        axs.legend()
        axs.title.set_text('Cumulant of the sample')
        axs.set_ylabel(y_labels[cumulant_index])
        axs.set_xlabel('I_jct [uA]')
    def cumulant_sample_O2(self,kernel_slice=slice(None),errorbar=False,capsize=2.5):
        """  <<p**2>> of the sample"""
        self._cumulants_sample(1,kernel_slice,errorbar,capsize)
    def cumulant_sample_O4 (self,kernel_slice=slice(None),errorbar=False,capsize=2.5):
        self._cumulants_sample(2,kernel_slice,errorbar,capsize)
    def n_no_fs(self,V_slice=slice(None)):
        """
            Filters gaussien / f_0
        """
        exp = self.exp
        labels          = exp._filter_info['gauss']['labels']
        ns              = exp.ns_corrected
        fs              = exp._filter_info['gauss']['fs']
        R_jct           = exp._R_jct
        I_jct           = Three_points_polarisation.compute_I_sample(exp._Vdc_antisym,exp._R_1M)
        plot_kw     = {'linestyle':'-'}
        fig , axs = subplots(1,1)
        for i, label in enumerate(labels):
            plot_kw['label'] = label 
            axs.plot(I_jct[V_slice]*10**6,1e27*ns[0,i,:][V_slice]*_h*fs[i]*1e9/R_jct,**plot_kw)
        axs.legend()
        axs.title.set_text('gaussian filters only divided by center freq.')
        axs.set_ylabel('SII[A**2/Hz]* 10**-27 = <n>hf/Raq')
        axs.set_xlabel('Idc [uA]')
        return fig 
    def _ns(self,kernel_slice,plot_fct,errorbar=False,corrected=False):
        """
            Code commun pour les figures <n>,<n**2>,<dn**2>
        """
        exp = self.exp
        labels          = exp._labels
        I_jct           = Three_points_polarisation.compute_I_sample(exp._Vdc_antisym,exp._R_1M)
        ns              = exp.ns_corrected if corrected else exp.ns
        ns_std          = exp.ns_std
        plot_kw = {'linestyle':'-'}
        fig , axs = subplots(1,1)
        for i, label in enumerate(labels[kernel_slice]):
            plot_kw['label'] = label 
            plot_fct(I_jct,ns,ns_std,i,axs,errorbar,**plot_kw)
        axs.legend()
        if corrected : 
            axs.title.set_text('Corrected ns')
        return fig
    def n(self,kernel_slice=slice(None),errorbar=False,corrected=False):
        """
             <n>
        """
        plot_fct = QsVsVdc_fig._plot_n
        return self._ns(kernel_slice,plot_fct,errorbar,corrected) 
    def n2(self,kernel_slice=slice(None),errorbar=False,corrected=False):
        """
            <n^2>
        """
        plot_fct = QsVsVdc_fig._plot_n2
        fig = self._ns(kernel_slice,plot_fct,errorbar,corrected)
        return fig 
    def dn2(self,kernel_slice=slice(None),errorbar=False,corrected=False):
        """
            Var n
        """
        plot_fct = QsVsVdc_fig._plot_dn2
        fig = self._ns(kernel_slice,plot_fct,errorbar,corrected)
        axs = fig.axes[0]
        title = 'Corrected' if corrected else ''
        axs.title.set_text(title)
        return fig 
    def _ns_vs_n(self,kernel_slice,plot_fct,errorbar=False,corrected=False):
        """
            Code commun pour les figures dn2_vs_n, diff_n_0_2_dn2
        """
        exp = self.exp
        labels          = exp._labels
        ns              = exp.ns_corrected if corrected else exp.ns
        ns_std          = exp.ns_std
        plot_kw = {'linestyle':'-'}
        fig , axs = subplots(2,len(labels)/2 + len(labels)%2 )
        AXS = fig.axes 
        for i, label in enumerate(labels[kernel_slice]):
            plot_kw['label'] = label 
            plot_fct(ns,ns_std,i,AXS[i],errorbar,**plot_kw)
        return fig 
    def dn2_vs_n(self,kernel_slice=slice(None),errorbar=False,corrected=False):
        """
            Var n vs n avec n(n+1)
        """
        plot_fct = QsVsVdc_fig._plot_dn2_vs_n
        fig = self._ns_vs_n(kernel_slice,plot_fct,errorbar,corrected)
        return fig 
    def diff_n_0_2_dn2(self,kernel_slice=slice(None),errorbar=False,corrected=False):
        """
            Difference entre Var n et n(n+1)
        """
        plot_fct = QsVsVdc_fig._plot_diff_n_0_2_dn2
        fig = self._ns_vs_n(kernel_slice,plot_fct,errorbar,corrected)
        return fig 
    def C4(self,kernel_slice=slice(None),errorbar=False,corrected=False):
        """
            C4 avant correction 
        """ 
        exp = self.exp
        labels          = exp._labels
        I_jct           = Three_points_polarisation.compute_I_sample(exp._Vdc_antisym,exp._R_1M)
        C4              = exp.C4_corrected if corrected else exp.C4
        plot_kw = {'linestyle':'-'}
        fig , axs = subplots(1,1)
        for i, label in enumerate(labels[kernel_slice]):
            plot_kw['label'] = label 
            axs.plot(I_jct*10**6,C4[i,:],**plot_kw)
        axs.legend()
        title = 'corrected' if corrected else ''
        axs.title.set_text(title)
        axs.set_ylabel('C4 [~]')
        axs.set_xlabel('I_jct [uA]')
        return fig 
    def sum_of_ns(self,kernel_slice=slice(None),errorbar=False,corrected=False):
        """
            Somme des n
        """
        exp = self.exp
        labels          = exp._labels
        labels_gauss    = exp._filter_info['gauss']['labels']
        fs              = exp._filter_info['gauss']['fs']
        I_jct           = Three_points_polarisation.compute_I_sample(exp._Vdc_antisym,exp._R_1M)
        ns              = exp.ns_corrected if corrected else exp.ns
        ns_std          = exp.ns_std
        plot_kw = {'linestyle':'-'}
        plot_kw_1 = {'linestyle':'--','marker':'*'}
        fig , axs = subplots(1,1)
        for i, label in enumerate(labels[kernel_slice]):
            plot_kw['label'] = label 
            QsVsVdc_fig._plot_n(I_jct,ns,ns_std,i,axs,errorbar,**plot_kw)    
        for i, label in enumerate(labels_gauss):
            for j in range(i):
                plot_kw_1['label'] = '1/2({:.1f}+{:.1f})'.format(fs[i],fs[j]) 
                QsVsVdc_fig._plot_sum_of_n(I_jct,ns,ns_std,i,j,axs,errorbar,**plot_kw_1)
        axs.legend()
        return fig
    def sum_of_Var_ns(self,kernel_slice=slice(None),errorbar=False,corrected=False,plot_cov_n0n1=False):
        """
            Somme des Var n
        """
        exp = self.exp
        gauss_slice     = exp._filter_info['gauss']['slice']
        
        labels          = exp._labels
        labels_gauss    = exp._filter_info['gauss']['labels']
        fs              = exp._filter_info['gauss']['fs']
        I_jct           = Three_points_polarisation.compute_I_sample(exp._Vdc_antisym,exp._R_1M)
        ns              = exp.ns_corrected if corrected else exp.ns
        ns_std          = exp.ns_std
        plot_kw = {'linestyle':'-'}
        plot_kw_1 = {'linestyle':'--','marker':'o'}
        plot_kw_2 = {'linestyle':'--','marker':'+'}
        plot_kw_3 = {'linestyle':'--','marker':'*'}
        fig , axs = subplots(1,1)
        for i, label in enumerate(labels[kernel_slice]):
            plot_kw['label'] = label 
            QsVsVdc_fig._plot_dn2(I_jct,ns,ns_std,i,axs,errorbar,**plot_kw)
        for i, label in enumerate(labels_gauss):
            for j in range(i): 
                n1 = ns[2,i,:]
                n2 = ns[2,j,:]
                summ    = (0.25)*(n1+n2+2.0*ns[0,i,:]*ns[0,j,:] + ns[0,i,:]+ns[0,j,:] )
                n1_std = ns_std[2,i,:]
                n2_std = ns_std[2,j,:]
                sum_std = (0.25)*(n1_std+n2_std+ 2.0*ns_std[0,i,:]*ns_std[0,j,:])
                plot_kw_3['label'] = '1/4(Var(n0)+Var(n1)+2*n0*n1) + 1/4(n0+n1) : {:.1f}&{:.1f}'.format(fs[i],fs[j])
                if errorbar:
                    plot_kw['capsize'] = 2.5
                    axs.errorbar(I_jct*10**6,summ,yerr=sum_std,**plot_kw_3) 
                else :
                    axs.plot(I_jct*10**6,summ,**plot_kw_3)
        if plot_cov_n0n1 :
            Y       = exp.cumulant_n0n1_corrected if corrected else exp.cumulant_n0n1
            axs.plot(I_jct*10**6,Y,label='<<n0n1>>')
        axs.set_ylabel('Var(n)[~]')
        axs.set_xlabel('I_jct [uA]') 
        axs.legend()
        return fig
    def n0n1_vs_n0_n1(self,corrected=False,plot_n0_n1=False):
        exp     = self.exp
        labels  = exp._labels
        n       = exp.ns[0,...]
        X       = n[0]*n[1] # staticly declared for now...
        Y       = exp.cumulant_n0n1_corrected if corrected else exp.cumulant_n0n1
        fig     = figure()
        ax      = gca()
        ax.plot(X,Y+X,label='<n0n1>')
        if plot_n0_n1:
            ax.plot(X,X,label='<n0><n1>')
        ax.set_xlabel('<n0><n1>')
        ax.set_ylabel('<n0n1>')
        s       = '<n0n1>_s = <<n0n1>>_s + <n0><n1>_s @ '+labels[2] # staticly declared for now...
        title   = 'Corrected : ' + s if corrected else s 
        ax.title.set_text(title)
        ax.legend()
        return fig
    def fit_n_vs_Vdc(self,V_th=None,deg=1):
        exp     = self.exp
        Vdc     = Three_points_polarisation.compute_V_sample(exp._Vdc_antisym,exp._R_jct,exp._R_1M)
        I_jct   = Three_points_polarisation.compute_I_sample(exp._Vdc_antisym,exp._R_1M)
        Vth     = V_th if numpy.any(V_th) else exp._V_th
        n       = exp.ns[0,...]
        P       = self.polyfit_of_n(Vdc,n,Vth,deg)
        labels  = exp._labels
        fig     = figure()
        ax      = gca()
        idx_p   = Vdc >= 0
        idx_n   = Vdc <= 0
        for nn,p_pos,p_neg,label in zip(n,P[0],P[1],labels):
            color = next(ax._get_lines.prop_cycler)['color']
            ax.plot(I_jct*1.0e6,nn,color=color,label=label)
            ax.plot(I_jct[idx_p]*1.0e6,Vdc[idx_p]*p_pos[0]+p_pos[1],color=color)
            ax.plot(I_jct[idx_n]*1.0e6,Vdc[idx_n]*p_neg[0]+p_neg[1],color=color)
        ax.legend(loc='upper right')
        string_frmt = \
        """
        V_th  = {} [uV]
        Pos a = {} [ph/V]
        Neg a = {} [ph/V]
        Pos b = {} [ph]
        Neg b = {} [ph]
        """
        Vth  = numpy.array([Vth]) if type(Vth) is float or type(Vth) is numpy.float64 else numpy.array(Vth)
        sth  = formated_tuple('{:.1f}' ,Vth*1.0e6)
        sap  = formated_tuple('{:.0f}' ,P[0,:,0])
        san  = formated_tuple('{:.0f}' ,P[1,:,0])
        sbp  = formated_tuple('{: .2f}',P[0,:,1])
        sbn  = formated_tuple('{: .2f}',P[1,:,1])
        string = string_frmt.format(sth,sap,san,sbp,sbn)  
        ax.text(0.0,0.0,string,transform=ax.transAxes)
        ax.set_ylabel('<n> [ph]')
        ax.set_xlabel('I_jct [uA]')
        return fig 
    """
    GRAVEYARD
    # def C4_and_fit_C4(self,kernel_slice=slice(None),V_th=None):
        # exp = self.exp
        # labels          = exp._labels
        # V_jct           = Three_points_polarisation.compute_V_sample(exp._Vdc_antisym,exp._R_jct,exp._R_1M)
        # C4              = exp.C4
        # n               = exp.ns[0,:,:] 
        # Vth             = V_th if V_th else exp._V_th
        # plot_kw_0 = {'linestyle':'-','marker':'*'}
        # plot_kw_1 = {'linestyle':'dotted'}
        # fig , axs = subplots(1,1)
        # fit_C4 = Ns_helper.gen_C4_fit_vector(n,C4,V_jct,Vth)
        # for i, label in enumerate(labels[kernel_slice]):
            # color = next(axs._get_lines.prop_cycler)['color']
            # plot_kw_0['label'] = label 
            # plot_kw_0['color'] = color 
            # plot_kw_1['color'] = color 
            # axs.plot(n[i,:],C4[i,:],**plot_kw_0)
            # axs.plot(n[i,:],fit_C4[i,:],**plot_kw_1)
        # axs.legend()
        # axs.title.set_text('Linear fit over threshold')
        # axs.set_ylabel('C4 [~]')
        # axs.set_xlabel('<n>')
        # return figs
    def diff_sum_of_Var_ns(self,kernel_slice=slice(None),errorbar=False,corrected=False):
        exp = self.exp
        gauss_slice     = exp._filter_info['gauss']['slice']
        
        labels          = exp._labels
        labels_gauss    = exp._filter_info['gauss']['labels']
        fs              = exp._filter_info['gauss']['fs']
        I_jct           = Three_points_polarisation.compute_I_sample(exp._Vdc_antisym,exp._R_1M)
        ns              = exp.ns_corrected if corrected else exp.ns
        ns_std          = exp.ns_std
        plot_kw = {'linestyle':'-'}
        plot_kw_1 = {'linestyle':'--','marker':'o'}
        plot_kw_2 = {'linestyle':'--','marker':'*'}
        plot_kw_3 = {'linestyle':'--','marker':'+'}
        fig , axs = subplots(1,1)
        for i, label in enumerate(labels_gauss):
            for j in range(i): 
                n1 = ns[2,i,:]
                n2 = ns[2,j,:]
                su     = ns[2,2,:] - (0.25)*(n1+n2)
                sum    = ns[2,2,:] - (0.25)*(n1+n2+2.0*ns[0,i,:]*ns[0,j,:])
                summ   = ns[2,2,:] - (0.25)*(n1+n2+2.0*ns[0,i,:]*ns[0,j,:] + ns[0,i,:]+ns[0,j,:] )
                n1_std = ns_std[2,i,:]
                n2_std = ns_std[2,j,:]
                sum_std = (0.25)*(n1_std+n2_std+ 2.0*ns_std[0,i,:]*ns_std[0,j,:])
                plot_kw_1['label'] = 'Var(n1&2) - 1/4(Var(n0)+Var(n1)) : {:.1f}&{:.1f}'.format(fs[i],fs[j])
                plot_kw_2['label'] = 'Var(n1&2) - 1/4(Var(n0)+Var(n1)+2*n0*n1) : {:.1f}&{:.1f}'.format(fs[i],fs[j])
                plot_kw_3['label'] = 'Var(n1&2) - 1/4(Var(n0)+Var(n1)+2*n0*n1) + 1/4(n0+n1) : {:.1f}&{:.1f}'.format(fs[i],fs[j])
                if errorbar:
                    plot_kw['capsize'] = 2.5
                    axs.errorbar(I_jct*10**6,su,yerr=sum_std,**plot_kw_1) 
                    axs.errorbar(I_jct*10**6,sum,yerr=sum_std,**plot_kw_2) 
                    axs.errorbar(I_jct*10**6,summ,yerr=sum_std,**plot_kw_3) 
                else :
                    axs.plot(I_jct*10**6,su,**plot_kw_1) 
                    axs.plot(I_jct*10**6,sum,**plot_kw_2) 
                    axs.plot(I_jct*10**6,summ,**plot_kw_3) 
        axs.set_ylabel('difference of Var(n)[~]')
        axs.set_xlabel('I_jct [uA]') 
        axs.legend()
        return fig
        # def diff_sum_of_ns(self,kernel_slice=slice(None),errorbar=False,corrected=False):
        # exp = self.exp
        # labels          = exp._labels
        # labels_gauss    = exp._filter_info['gauss']['labels']
        # fs              = exp._filter_info['gauss']['fs']
        # I_jct           = Three_points_polarisation.compute_I_sample(exp._Vdc_antisym,exp._R_1M)
        # ns              = exp.ns_corrected if corrected else exp.ns
        # ns_std          = exp.ns_std
        # plot_kw = {'linestyle':'-'}
        # plot_kw_1 = {'linestyle':'--','marker':'*'}
        # fig , axs = subplots(1,1)
        # for i, label in enumerate(labels[kernel_slice]):
            # plot_kw['label'] = label 
            # QsVsVdc_fig._plot_n(I_jct,ns,ns_std,i,axs,errorbar,**plot_kw)    
        # for i, label in enumerate(labels_gauss):
            # for j in range(i):
                # plot_kw_1['label'] = '1/2({:.1f}+{:.1f})'.format(fs[i],fs[j]) 
                # QsVsVdc_fig._plot_sum_of_n(I_jct,ns,ns_std,i,j,axs,errorbar,**plot_kw_1)
        # axs.legend()
        # return fig
    """
        
class QsVsVdc_exp(QsVsVdc_info,Lagging_computation):
    """
        Parent(s) : Quads
        
        Important variables :
            self.moments 
                shape = (4,n_kernels,l_Vdc)
                1st index meaning
                    0 : <q>
                    1 : <q**2>
                    2 : <q**4>
                    3 : <q**8>
            self.ns
                shape = (4,n_kernels,cumulants_sample.shape[-1])
                1st index meaning
                    0 : <n>
                    1 : <n**2>
                    2 : <dn**2>
                    3 : fano
        Important objects :
            self.Hs
                shape = ( n_kernels,l_Vdc )
        Todos : 
            - Imbed the computation part for ns in a class of its own
        Bugs :
    """
    def _set_devices(self,devices):
        self._gz                        =   devices[0] 
        self._yoko                      =   devices[1]
    def _init_log(self):
        l_Vdc                           = len(self.Vdc)
        n                               = self._n_measures
        l_data                          = self._l_data
        tmp                             = self._estimated_time_per_loop 
        times_estimate                  = (self._n_measures*tmp,tmp)
        self._log                       = logger_acq_and_compute(times_estimate,n,l_Vdc,l_data) 
    def _init_objects(self):
        self._init_TimeQuad()
        self._init_Histograms()
    def _init_TimeQuad(self):
        # Use R_jct if the gain has been premeasurement or R_tl if g = 1.0 i.e. no calibration
        self._X         = TimeQuad_uint64_t(self._R_jct,self._dt,self._l_data,self._kernel_conf,self._Filters,self._g,self._alpha,self._l_fft,self._n_threads)
        self.betas      = self._X.betas()
        self.filters    = self._X.filters()
        self.ks         = self._X.ks()
        self.qs         = self._X.quads()[0,:,:]
        self._data_gz   = self._gz.get() # int16
        self._X.execute( self._data_gz ) # force the initialization of memory
    def _init_Histograms(self):
        l_Vdc           = len(self._Vdc_exp)
        max             = self._max
        n_threads       = self._n_threads
        nb_of_bin       = self._nb_of_bin
        n_kernels       = self._n_kernels
        self.Hs         = build_array_of_objects( (n_kernels , l_Vdc ) , Histogram_uint64_t_double , *(nb_of_bin,n_threads,max) )            
        self._H_x       = Histogram_uint64_t_double.abscisse(max,nb_of_bin)  #Should be made static and called static
    def reset_objects(self):
        # Nothing to reset for Quads
        self._reset_Hs()
    def _reset_Hs(self):
        for h in self.Hs.flat :
            h.reset()
    def get_hs(self):
        """
            Converts histograms to numpy.array and returns a single array
        """
        nb_of_bin  = self._nb_of_bin
        shape = self.Hs.shape + (nb_of_bin,)
        hs    = zeros(shape,dtype=float) # histograms are doubles here
        for k in range(shape[0]) :
            for v in range(shape[1]):
                hs[k,v,:] = self.Hs[k,v].get()
        return hs
    def set_g(self,g):
        self._X.set_g(g)
    #############
    # Utilities #
    #############   
    def _compute_moments(self):
        """
            Ordre>
              0       <q>
              1       <q**2>
              2       <q**4>
              3       <q**8>
              
            Note : 
                - Could not find a way to make this function more general by broacasting .moment_no_clip
                - changed moment for centered_moment 26-02-2021 for testing
        """
        powers      = QsVsVdc_info.powers
        n_kernels   = self._n_kernels
        n_threads   = self._n_threads
        l_Vdc       = len(self._Vdc_exp)
        moments     = numpy.zeros((len(powers),n_kernels,l_Vdc),dtype=float) 
        Hs  = self.Hs
        H_x = self._H_x
        for i in range(n_kernels) :
            for j in range(len(self._Vdc_exp)) :
                n_total   = int( Hs[i,j].centered_moment(bins=H_x,exp=0        ,n_total=1        ,n_threads=n_threads , no_clip = True ) )
                moments[0,i,j] = Hs[i,j].centered_moment(bins=H_x,exp=powers[0],n_total=n_total  ,n_threads=n_threads , no_clip = True)
                moments[1,i,j] = Hs[i,j].centered_moment(bins=H_x,exp=powers[1],n_total=n_total  ,n_threads=n_threads , no_clip = True)
                moments[2,i,j] = Hs[i,j].centered_moment(bins=H_x,exp=powers[2],n_total=n_total  ,n_threads=n_threads , no_clip = True)
                moments[3,i,j] = Hs[i,j].centered_moment(bins=H_x,exp=powers[3],n_total=n_total  ,n_threads=n_threads , no_clip = True)
        return moments
    @staticmethod  
    def compute_errors(moments,n):
        shape = (moments.shape[0]-1,)
        shape += moments.shape[1:]
        errors = numpy.zeros(shape,dtype=float) 
        errors[:,...] = Analysis.SE(moments[1:,...],moments[0:-1,...],n)
        return errors
    #################
    # Loop behavior #
    #################
    def _set_and_wait_all_devices(self,conditions):
        vdc_next,       = conditions
        self._yoko.set_and_wait(vdc_next)
        # nothing to do with gz
    def _all_loop_open(self) :
        super(QsVsVdc_exp,self)._all_loop_open()
        self._yoko.set_init_state(abs(self.Vdc).max())
    def _loop_core(self,index_tuple,condition_tuple):
        """
            Works conditionnaly to the computing being slower than 0.4 sec
        """
        j,          = index_tuple     
        vdc_next,   = condition_tuple   
        self._data_gz  = self._gz.get() # int16 
        self._yoko.set(vdc_next)
        self._log.event(0)
        self._X.execute(self._data_gz)
        for i in range(self._n_kernels):
            self.Hs[i,j].accumulate( self.qs[i,:])
        self._log.event(1)
        super(QsVsVdc_exp,self)._loop_core(index_tuple,condition_tuple)
    def _last_loop_core_iteration(self):
        self._data_gz  = self._gz.get() # int16 
        self._log.event(0)
        self._X.execute(self._data_gz)
        for i in range(self._n_kernels):
            self.Hs[i,-1].accumulate( self.qs[i,:])
        self._log.event(1)
        super(QsVsVdc_exp,self)._loop_core(tuple(),tuple())
    ######################
    # Analysis Utilities #
    ######################
    def _moments_correction(self,moments):
        powers      = QsVsVdc_info.powers
        half_norms  = self._X.half_norms()[0,:]
        return Quads_helper._moments_correction(moments,half_norms,powers)
    def get_half_norms(self):
        return self._X.half_norms()[0,:]
    ############
    # Analysis #
    ############
    def _compute_reduction(self):
        self.hs             = self.get_hs()
        self.moments        = self._compute_moments()
        self.half_norms     = self.get_half_norms()
        self.moments        = self._moments_correction(self.moments)
        self.moments_std    = self.compute_errors(self.moments,self._n_measure_total()*self._l_data)
    def _build_data(self):
        return {\
            'betas'                 : self.betas ,
            'filters'               : self.filters ,
            'half_norms'            : self.half_norms ,
            'hs'                    : self.hs,
            'ks'                    : self.ks ,
            'moments'               : self.moments ,
            'moments_std'           : self.moments_std
        }

class QsVsVdc_analysis(QsVsVdc_info,Analysis):  
    """
        Last update 
            Imported many methods from Ns_helper
            Updated the correction method for C4 (and therefore ns)
            Deleted unecessary methods
        Options :
            -
        Todos : 
            - __doc__
            - Make errorbar calculations properly
    """
    __exp_class__   = QsVsVdc_exp
    #############
    # Utilities #
    #############  
    @staticmethod
    def compute_cumulants(moments):
        shape = (moments.shape[0]-1,)
        shape += moments.shape[1:]
        cumulants           = numpy.zeros( shape ,dtype=float)
        cumulants[0,...]    = moments[0,...] 
        cumulants[1,...]    = moments[1,...] 
        cumulants[2,...]    = moments[2,...]     - 3.0*(moments[1,...] + 0.5 )**2  # <p**4> -3 <p**2> **2
        return cumulants
    @staticmethod
    def compute_cumulants_std(moments,moments_std):
        shape = moments_std.shape
        cumulants_std           = numpy.zeros( shape ,dtype=float)
        cumulants_std[0,...]    = moments_std[0,...] 
        cumulants_std[1,...]    = moments_std[1,...] 
        cumulants_std[2,...]    = moments_std[2,...] - 6.0*(moments[1,...] + 0.5 )*(moments_std[1,...])  # <p**4> -3 <p**2> **2
        return cumulants_std
    @staticmethod
    def compute_fano(dn2,n):
        return dn2/n
    @staticmethod
    def compute_ns(cumulants_sample):
        """
            0 : <n>
            1 : <n**2>
            2 : <dn**2>
        """
        shape_tmp   = cumulants_sample.shape[1:]
        ns          = numpy.empty((3,)+shape_tmp,dtype=float)
        n           = cumulants_sample[1,...]
        C4          = cumulants_sample[2,...]
        ns[0,...]   = n
        ns[1,...]   = (2.0/3.0)*C4 + 2.0*n**2 - n #probablement pas bon
        ns[2,...]   = (2.0/3.0)*C4 +     n**2 + n
        return ns
    @staticmethod
    def fit_of_C4(C4_s,C2_3_s,V_th_slice,p0=(1.0,),verbose=False):
        """
            C4,d = C4,m(V) - C4,m(0) = C4,s + K( C2,m**3(V) - C2,m**3(0) )
            Assuming C4_s is zero where V_th_slice == True
            C4_del = K*C2_3_del (see model)
        """
        C4_del      = C4_s  [V_th_slice]
        C2_3_del    = C2_3_s[V_th_slice]
        def model(x,p):
            p0 = p[0]
            return p0*x
        f = lsqfit(C2_3_del,C4_del,p0,model,verbose=verbose)
        return f
    @staticmethod
    def compute_C4_corr(C4_s,C2_3_s,V_th_slice,p0=(1.0,),verbose=False):
        n_kernels   = C4_s.shape[0]
        if type(V_th_slice)== slice :
            V_th_s = numpy.empty((n_kernels,),dtype=object)
            V_th_s[...] = V_th_slice
        elif len(V_th_slice.shape)==1:
            V_th_s = numpy.empty((n_kernels,len(V_th_slice)),dtype=bool)
            V_th_s[...] = V_th_slice
        else :
            V_th_s = V_th_slice
        a           = numpy.zeros((n_kernels,))
        for i,(c4_s,c2_3_s,v_th_s) in enumerate(zip(C4_s,C2_3_s,V_th_s)):
            f    = QsVsVdc_analysis.fit_of_C4(c4_s,c2_3_s,v_th_s,p0,verbose)
            a[i] = f[0]
        return C4_s- a[:,None]*C2_3_s
    @staticmethod
    def compute_ns_corrected(C4_corr,C2_s):
        shape_tmp   = C4_corr.shape
        ns_corr     = numpy.empty((3,)+shape_tmp,dtype=float)
        ns_corr[0,...] = C2_s 
        ns_corr[1,...] = (2.0/3.0)*C4_corr + 2.0*C2_s**2 - C2_s  #probablement pas bon
        ns_corr[2,...] = (2.0/3.0)*C4_corr +     C2_s**2 + C2_s
        return ns_corr
    @staticmethod
    def compute_ns_std(ns,cumulants_sample,cumulants_sample_std):
        """
            Probably needs to be updated
        """
        shape_tmp  = cumulants_sample.shape[1:]
        ns_std = numpy.empty((4,)+shape_tmp,dtype=float)
        ns_std[0,...] = (cumulants_sample_std[1,...])
        ns_std[1,...] = (2.0/3.0)*cumulants_sample_std[2,...] + 4.0*cumulants_sample_std[1,...]*cumulants_sample[1,...] - cumulants_sample_std[1,...]
        ns_std[2,...] = (2.0/3.0)*cumulants_sample_std[2,...] + 2.0*cumulants_sample_std[1,...]*cumulants_sample[1,...] 
        return ns_std
    @staticmethod
    def compute_cumulant_n0n1_sample(C4_0_sample,C4_1_sample,C4_0_and_1_sample):
        """
            Returns <<n0n1>>(V) - <<n0n1>>(V=0)
            
            Notes :
            (see Notes Bertrand n1n2 vs cumulants.pdf for details)
            
            Here I assume that
            the combination is s_0 and s_1 is of the form
            s_tot = (s_0+s_1)/sqrt(2)
            thefore 
            C4_0_and_1_sample mus be *4.0 to get the right definition
            to fit with moments_cumulants_helper.compute_s0_square_s1_square_sample
        """
        return moments_cumulants_helper.compute_s0_square_s1_square_sample(C4_0_sample,C4_1_sample,4.0*C4_0_and_1_sample)
    ######################
    # Analysis Utilities #
    ######################
    def recompute_moments_std(self,empiric=True):
        """
            Worls only if interlacing was used
        """ 
        if empiric :
            moment_ref                  =   self.moments[:-1:,...,0::2]
            self.moments_std[...,:]     =   moment_ref.std(axis=-1)[...,None]  
        else :
            self.moments_std    = QsVsVdc_exp.compute_errors(self.moments,self._n_measure_total()*self._l_data)
    def _compute_analysis(self,V_th=None,p0=(1.0,),verbatim=False):
        self.cumulants              = self.compute_cumulants(self.moments)
        self.cumulants_std          = self.compute_cumulants_std(self.moments,self.moments_std)
        ref_options = self._ref_options
        self.cumulants_sample       = Conditions_logic.compute_cumulants_sample(self.cumulants,**ref_options)
        self.cumulants_sample_std   = Conditions_logic.compute_cumulants_sample_std(self.cumulants_std,**ref_options)
        self.ns                     = self.compute_ns(self.cumulants_sample)
        self.ns_std                 = self.compute_ns_std(self.ns,self.cumulants_sample,self.cumulants_sample_std)
        
        R_jct       = self._R_jct
        R_1M        = self._R_1M
        Vdc_antisym = self._Vdc_antisym
        V_jct       = Three_points_polarisation.compute_V_sample(Vdc_antisym,R_jct,R_1M)
            
        Vth         = V_th if type(V_th)!=type(None) else self._V_th
        Vth         = numpy.array([Vth]) if type(Vth) != numpy.ndarray else Vth
        V_th_slice  = Vth < abs(V_jct) if type(Vth)==float or len(Vth) == 1 else Vth[:,None] < abs(V_jct[None,:])
        
        C4_s        = self.cumulants_sample[2]
        C2_s        = self.cumulants_sample[1]
        C2          = self.cumulants[1,...]
        slice_r     = Conditions_logic.get_references_slice(**ref_options)
        slice_c     = Conditions_logic.get_conditions_slice(**ref_options)
        C2_3_s      = C2[...,slice_c]**3  - C2[...,slice_r]**3
        
        C4_corr     = QsVsVdc_analysis.compute_C4_corr(C4_s,C2_3_s,V_th_slice,p0,verbose)
        self.ns_corrected           = self.compute_ns_corrected(C4_corr,C2_s)
        self.C4                     = C4_s
        self.C4_corrected           = C4_corr
        
        # For now I'll feed the good C4's statically 
        self.cumulant_n0n1          = self.compute_cumulant_n0n1_sample(C4_s[0],C4_s[1],C4_s[2])
        self.cumulant_n0n1_corrected= self.compute_cumulant_n0n1_sample(C4_corr[0],C4_corr[1],C4_corr[2])
    def _build_data(self):
        return {\
            'betas'                 : self.betas ,
            'filters'               : self.filters ,
            'ks'                    : self.ks ,
            'hs'                    : self.hs ,
            'half_norms'            : self.half_norms ,
            'moments'               : self.moments,
            'moments_std'           : self.moments_std,
            'cumulants'             : self.cumulants,
            'cumulants_std'         : self.cumulants_std,
            'cumulants_sample'      : self.cumulants_sample,
            'cumulants_sample_std'  : self.cumulants_sample_std,
            'ns'                    : self.ns ,
            'ns_std'                : self.ns_std ,
            'ns_corrected'          : self.ns_corrected ,
            'C4'                    : self.C4 ,
            'cumulant_n0n1'         : self.cumulant_n0n1,
            'cumulant_n0n1_corrected'   : self.cumulant_n0n1_corrected,
        }
  
class Quads_helper_fig():
    @staticmethod
    def plot_Filters( Filters , labels, ax , l_dft , dt ):
        freqs = numpy.fft.rfftfreq(l_dft,dt) 
        for i,f in enumerate(Filters) :
            ax.plot( freqs , f , label = labels[i] , marker='o') 
        ax.set_xlabel(" GHz ")
        ax.legend()
    @staticmethod
    def plot_Kernels(ts,ks,labels,ax,dt):   
        for j in range(ks.shape[1]):
            color = next(ax._get_lines.prop_cycler)['color']
            for k in range(ks.shape[0]):
                if k==0:
                    ax.plot( ts , ks[k,j,:] , color=color , label = labels[j] ) 
                else :
                    ax.plot( ts , ks[k,j,:] , color=color , linestyle='--' ) 
        ax.set_xlabel("ns")
        ax.legend()
    @staticmethod
    def plot_Kernels_FFT(freqs,ks,labels,ax,dt):    
        for j in range(ks.shape[1]):
            color = next(ax._get_lines.prop_cycler)['color']
            for k in range(ks.shape[0]):
                if k==0:
                    ax.plot( freqs, numpy.abs(numpy.fft.rfft(ks[k,j,:])) , color=color , label = labels[j]) 
                else :
                    ax.plot( freqs, numpy.abs(numpy.fft.rfft(ks[k,j,:])) , color=color , linestyle='--'  ) 
        ax.set_xlabel("GHz")
        ax.legend()
        
def _gen_dict_helper(d):
    out = dict()
    for k,i in d.items():
        if i is not None:
            out.update({k:i})
    return out
    
def gen_quads_info(l_kernel,kernel_conf=None,alpha=None,filters_info=None):
    l_kernel    = int(l_kernel)
    l_hc        = l_kernel/2 + 1 
    quads_info  = {'l_kernel':l_kernel,'l_hc':l_hc,'kernel_conf':kernel_conf,'alpha':alpha,'filters_info':filters_info}
    return _gen_dict_helper(quads_info)
 
def gen_aqc_info(l_data,dt,gz_gain_dB=None,mv_per_bin=None):
    return _gen_dict_helper({'l_data':l_data,'dt':dt,'gz_gain_dB':gz_gain_dB,'mv_per_bin':mv_per_bin})
    
def gen_compute_info(n_threads,l_fft=None):
    return _gen_dict_helper({'n_threads':n_threads,'l_fft':l_fft})
    
def gen_circuit_info(R_jct,R_1M,R_tl=None,g=None,V_th=None):
    return _gen_dict_helper({'R_jct':R_jct,'R_1M':R_1M,'R_tl':R_tl,'g':g,'V_th':V_th})
 
def gen_hist_info(nb_of_bin,max):
    return _gen_dict_helper({'nb_of_bin':nb_of_bin,'max':max})