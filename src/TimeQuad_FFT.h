#pragma once

#include <omp_extra.h>

#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
namespace py = pybind11;
using namespace pybind11::literals;

#include <special_functions.h>
#include <Windowing.h>
#include <Multi_array.h>
#include <TimeQuad_algorithm.h>

#include <fftw3.h>

typedef unsigned int uint;
typedef py::array_t<double,py::array::c_style> np_double;
typedef py::array_t<complex_d,py::array::c_style> np_complex_d;

template<class Quads_Index_Type=uint>
class TimeQuad_FFT: public TimeQuad_algorithm
{
	public :
	// Contructor
	TimeQuad_FFT
	( 
		const Multi_array<double,3>& 					ks , 
		const Multi_array<double,3,Quads_Index_Type>& 	quads , 
		double dt , 
		uint  l_fft , 
		int n_threads 
	);
	// Destructor
	~TimeQuad_FFT() override;
	
	# define EXECUTE(DataType) \
		void execute( Multi_array<DataType,1,uint64_t>& data ) override ;	
	
	EXECUTE(int16_t)
							
	#undef EXECUTE
	
	// Utilities
	uint compute_n_quads			(const Multi_array<double,3>& ks )						{ return ks.get_n_k()			;};
	uint compute_n_kernels			(const Multi_array<double,3>& ks )						{ return ks.get_n_j()			;};
	uint compute_l_kernels			(const Multi_array<double,3>& ks )						{ return ks.get_n_i()			;};
	uint compute_l_chunk			( uint l_kernel ,  uint l_fft  )					{ return l_fft - l_kernel + 1 	;};
	uint compute_n_chunks			( uint64_t l_data , uint l_chunk )					{ return l_data/l_chunk 		;};
	uint compute_l_reste			( uint64_t l_data , uint l_chunk )					{ return l_data%l_chunk 		;};
	
    void update_kernels(){prepare_kernels();};
	
	private :
	// Kernels info
	uint n_quads ;
	uint n_kernels ;
	uint l_kernel ;

    double dt ;

	uint l_fft; // Lenght(/number of elements) of the FFT used for FFT convolution
	int n_threads ;
	
	uint l_chunk ; // The length of a chunk
	uint n_chunks ; // The number of chunks
	uint l_reste ; // The length of what didn't fit into chunks
	
	// Inherited arrays
	const Multi_array<double,3>& ks ;
	const Multi_array<double,3,Quads_Index_Type>& quads ; 
	
	fftw_plan kernel_plan;
	fftw_plan g_plan;
	fftw_plan h_plan;
	
	// Pointers to all the complex kernels
	Multi_array<complex_d,3> ks_complex; // [quad][n_kernel][frequency]
	
	// Memory allocated for fft_conv computing
	/* Triple pointers */
	Multi_array<double,2> 		gs ; // [thread_num][frequency] Catches data from data*
	Multi_array<complex_d,2> 	fs ; // [thread_num][frequency] Catches DFT of data
	Multi_array<complex_d,4> 	hs ; // [quads][n_kernel][thread_num][frequency]
	
	void prepare_plans();
	void destroy_plans();
	
	/* Constructor sequence */
	void prepare_kernels() ;
};

#include "../src/TimeQuad_FFT.tpp"