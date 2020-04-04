#pragma once

#include "../../includes/header_common.h"
#include "../../SM-Special_functions/special_functions.h"
#include "../../SM-Windowing/includes/Windowing.h"
#include "../../SM-Omp_extra/includes/omp_extra.h"

#include "TimeQuad_algorithm.h"

class TimeQuad_FFT: public TimeQuad_algorithm
{
	public :
	// Contructor
	TimeQuad_FFT
	( 
		const Multi_array<double,2>& ks_p , 
		const Multi_array<double,2>& ks_q ,
		const Multi_array<double,2>& ps , 
		const Multi_array<double,2>& qs ,
		uint l_kernel , uint n_kernels , uint64_t l_data , uint  l_fft , int n_threads 
	);
	// Destructor
	~TimeQuad_FFT() override;
	
	# define EXECUTE(DataType) \
		void execute( DataType* data ) override ;	
	
	EXECUTE(int16_t)
							
	#undef EXECUTE
	
	// Utilities
	inline uint compute_l_chunk( uint l_kernel ,  uint l_fft  ){ return l_fft - l_kernel + 1 ; };
	inline uint compute_n_chunks( uint64_t l_data , uint l_chunk ){ return  l_data/l_chunk ;	};
	inline uint compute_l_reste( uint64_t l_data , uint l_chunk ){ return l_data%l_chunk ; };
		
	private :
	// Kernels info
	uint l_kernel ;
	uint n_kernels ;
	
	// Acquisition info
	uint64_t l_data ;

	uint l_fft; // Lenght(/number of elements) of the FFT used for FFT convolution
	int n_threads ;
	
	uint l_chunk ; // The length of a chunk
	uint n_chunks ; // The number of chunks
	uint l_reste ; // The length of what didn't fit into chunks
	
	// Inherited arrays
	const Multi_array<double,2>& ks_p ;
	const Multi_array<double,2>& ks_q ;
	const Multi_array<double,2>& ps ; 
	const Multi_array<double,2>& qs ;
	
	fftw_plan kernel_plan;
	fftw_plan g_plan;
	fftw_plan h_plan;
	
	// Pointers to all the complex kernels
	Multi_array<complex_d,2> ks_p_complex; // [n_kernel][frequency]
	Multi_array<complex_d,2> ks_q_complex;
	
	// Memory allocated for fft_conv computing
	/* Triple pointers */
	Multi_array<double,2> gs ; // [thread_num][frequency] Catches data from data*
	Multi_array<complex_d,2> fs ; // [thread_num][frequency] Catches DFT of data
	Multi_array<complex_d,3> h_ps ; // [n_kernel][thread_num][frequency]
	Multi_array<complex_d,3> h_qs ;
	
	void prepare_plans();
	void destroy_plans();
	
	/* Constructor sequence */
	void prepare_kernels() ;
};
