#pragma once
#include <sys/types.h>
#include <omp_extra.h>

#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
namespace py = pybind11;
using namespace pybind11::literals;

#include <Multi_array.h>

#include <fftw3.h>

typedef unsigned int uint ;
typedef std::complex<float> complex_f ;
typedef std::complex<double> complex_d ;
typedef py::array_t<float,py::array::c_style> np_float;
typedef py::array_t<double,py::array::c_style> np_double;
typedef py::array_t<int16_t,py::array::c_style> np_int16;
typedef py::array_t<complex_f,py::array::c_style> np_complex_f;
typedef py::array_t<complex_d,py::array::c_style> np_complex_d;


#include <iostream>
#include <chrono>
typedef std::chrono::high_resolution_clock CLOCK  ;
typedef std::chrono::time_point<CLOCK> time_point  ;

template<class TimePointType>
uint64_t duration(TimePointType t0, TimePointType t1)
{
    return std::chrono::time_point_cast<std::chrono::nanoseconds>(t1).time_since_epoch().count() - std::chrono::time_point_cast<std::chrono::nanoseconds>(t0).time_since_epoch().count();
}

template<class FloatType=double,class DataType=int16_t>
class TimeQuad_FFT
{};

template<class DataType>
class TimeQuad_FFT<double,DataType>
{
	public :
	// Contructor
	TimeQuad_FFT
	( 
		np_double ks , 
		py::array_t<DataType, py::array::c_style> data , 
		double dt , 
		uint  l_fft , 
		int n_threads 
	);
	// Destructor
	~TimeQuad_FFT();
	
    void execute( Multi_array<DataType,1,uint64_t>& data );
    void execute_py(np_double& ks,py::array_t<DataType, py::array::c_style>& data);
    
    // Returns only the valid part of the convolution
    np_double get_quads(); // Data are shared
    
	// Utilities
	uint compute_n_prod			    (np_double& ks);
    std::vector<ssize_t>  get_shape (np_double& ks);
	uint compute_l_kernels			(np_double& ks);
    uint64_t compute_l_data 		(py::array_t<DataType, py::array::c_style>& data);
    uint64_t compute_l_valid	( uint l_kernel, uint64_t l_data )	{ return l_data - l_kernel + 1 	;};
    uint64_t compute_l_full	( uint l_kernel, uint64_t l_data )	{ return l_kernel + l_data - 1 	;};
    Multi_array<double,2,uint32_t> copy_ks( np_double& np_ks, uint n_prod );
    
	uint compute_l_chunk			( uint l_kernel ,  uint l_fft  )					{ return l_fft - l_kernel + 1 	;};
	uint compute_n_chunks			( uint64_t l_data , uint l_chunk )					{ return l_data/l_chunk 		;};
	uint compute_l_reste			( uint64_t l_data , uint l_chunk )					{ return l_data%l_chunk 		;};
	
	private :

	uint n_prod ;
	std::vector<ssize_t> ks_shape;
    uint l_kernel ;
    uint64_t l_data ; //
    uint64_t l_valid ;
    uint64_t l_full  ;
    Multi_array<double,2,uint32_t> ks ;
	Multi_array<double,2> quads ;
    double dt ;
    uint l_fft; // Lenght(/number of elements) of the FFT used for FFT convolution
	int n_threads ;
	uint l_chunk ; // The length of a chunk
	uint n_chunks ; // The number of chunks
	uint l_reste ; // The length of what didn't fit into chunks
	
	fftw_plan kernel_plan;
	fftw_plan g_plan;
	fftw_plan h_plan;
	
	// Pointers to all the complex kernels
	Multi_array<complex_d,2,uint32_t> ks_complex; // [n_prod][frequency]
	Multi_array<double,2,uint32_t> 		gs ; // [thread_num][frequency] Catches data from data*
	Multi_array<complex_d,2,uint32_t> 	fs ; // [thread_num][frequency] Catches DFT of data
	Multi_array<complex_d,3,uint32_t> 	hs ; // [n_prod][thread_num][frequency]
	
    void checks();
	void prepare_plans();
	void destroy_plans();
	
	void prepare_kernels(np_double& ks) ;
    
    void execution_checks(np_double& ks, py::array_t<DataType,py::array::c_style>& data);

};

template<class DataType>
class TimeQuad_FFT<float,DataType>
{
	public :
	// Contructor
	TimeQuad_FFT
	( 
		np_double ks , 
		py::array_t<DataType, py::array::c_style> data , 
		double dt , 
		uint  l_fft , 
		int n_threads 
	);
	// Destructor
	~TimeQuad_FFT();
	
    void execute( Multi_array<DataType,1,uint64_t>& data );
    void execute_py(np_double& ks,py::array_t<DataType, py::array::c_style>& data);
    
    // Returns only the valid part of the convolution
    np_float get_quads(); // Data are shared
    
	// Utilities
	uint compute_n_prod			    (np_double& ks);
    std::vector<ssize_t>  get_shape (np_double& ks);
	uint compute_l_kernels			(np_double& ks);
    uint64_t compute_l_data 		(py::array_t<DataType, py::array::c_style>& data);
    uint64_t compute_l_valid	( uint l_kernel, uint64_t l_data )	{ return l_data - l_kernel + 1 	;};
    uint64_t compute_l_full	( uint l_kernel, uint64_t l_data )	{ return l_kernel + l_data - 1 	;};
    Multi_array<float,2,uint32_t> copy_ks( np_double& np_ks, uint n_prod );
    
	uint compute_l_chunk			( uint l_kernel ,  uint l_fft  )					{ return l_fft - l_kernel + 1 	;};
	uint compute_n_chunks			( uint64_t l_data , uint l_chunk )					{ return l_data/l_chunk 		;};
	uint compute_l_reste			( uint64_t l_data , uint l_chunk )					{ return l_data%l_chunk 		;};
	
	private :

	uint n_prod ;
	std::vector<ssize_t> ks_shape;
    uint l_kernel ;
    uint64_t l_data ; //
    uint64_t l_valid ;
    uint64_t l_full  ;
    Multi_array<float,2,uint32_t> ks ;
	Multi_array<float,2> quads ;
    float dt ;
    uint l_fft; // Lenght(/number of elements) of the FFT used for FFT convolution
	int n_threads ;
	uint l_chunk ; // The length of a chunk
	uint n_chunks ; // The number of chunks
	uint l_reste ; // The length of what didn't fit into chunks
	
	fftwf_plan kernel_plan;
	fftwf_plan g_plan;
	fftwf_plan h_plan;
	
	// Pointers to all the complex kernels
	Multi_array<complex_f,2,uint32_t> ks_complex; // [n_prod][frequency]
	Multi_array<float,2,uint32_t> 		gs ; // [thread_num][frequency] Catches data from data*
	Multi_array<complex_f,2,uint32_t> 	fs ; // [thread_num][frequency] Catches DFT of data
	Multi_array<complex_f,3,uint32_t> 	hs ; // [n_prod][thread_num][frequency]
	
    void checks();
	void prepare_plans();
	void destroy_plans();
	
	void prepare_kernels(np_double& ks) ;
    
    void execution_checks(np_double& ks, py::array_t<DataType,py::array::c_style>& data);

};

#include "TimeQuad_FFT_double.tpp"
#include "TimeQuad_FFT_float.tpp"