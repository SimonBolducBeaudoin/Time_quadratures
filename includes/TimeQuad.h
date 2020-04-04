#pragma once

#include "../../includes/header_common.h"
#include "../../SM-Special_functions/special_functions.h"
#include "../../SM-Windowing/includes/Windowing.h"
#include "../../SM-Omp_extra/includes/omp_extra.h"

#include "../../SM-Multi_array/Multi_array.h"

#include "TimeQuad_algorithm.h"
#include "TimeQuad_FFT.h"
#include "TimeQuad_direct.h"

/*
	TODOS
	- Find a way to automatically normalized the result of the convolution product ..?
*/


/*
	The member TimeQuad_algorithm is implementing virtual functions to acheive run time polymorphism.
	See :
		- https://www.geeksforgeeks.org/polymorphism-in-c/
		- https://www.geeksforgeeks.org/virtual-function-cpp/
*/

class TimeQuad
{
	public :
		// Contructors
		/* Direct convolution */
		TimeQuad
		( 
			uint l_kernel , uint n_kernels , uint64_t l_data , double dt , double f_max_analogue , 
			double f_min_analogue , double alpha = 0.5 , int n_threads = 36 
		);  /*Windows and filter set to NULL*/
		TimeQuad
		( 
			uint l_kernel , uint n_kernels , uint64_t l_data , double dt , double f_max_analogue , 
			double f_min_analogue , Multi_array<complex_d,2> filters , Multi_array<double,2> windows , int n_threads = 36 
		);
		TimeQuad 
		( 
			uint l_kernel , uint n_kernels , uint64_t l_data , double dt , 	double f_max_analogue , 
			double f_min_analogue , np_complex_d filters , double alpha , int n_threads 
		); /*Filter given by numpy array*/
		TimeQuad 
		( 
			uint l_kernel , uint n_kernels , uint64_t l_data , double dt , 	double f_max_analogue , 
			double f_min_analogue , np_complex_d filters , 	np_double windows , int n_threads 
		); /*Windows and filter given by numpy array*/
		/* FFT convolution */
		TimeQuad
		( 
			uint l_kernel , uint n_kernels , uint64_t l_data , double dt , double f_max_analogue , 
			double f_min_analogue , double alpha , uint l_fft , int n_threads = 36 
		);
		TimeQuad
		( 
			uint l_kernel , uint n_kernels , uint64_t l_data , double dt , double f_max_analogue , double f_min_analogue , 
			Multi_array<complex_d,2> filters , Multi_array<double,2> windows , uint l_fft , int n_threads = 36 
		);
		TimeQuad 
		( 
			uint l_kernel , uint n_kernels , uint64_t l_data , double dt , 	double f_max_analogue , 
			double f_min_analogue , np_complex_d filters , double alpha , uint l_fft , int n_threads 
		); /*Filter given by numpy array*/
		TimeQuad 
		( 
			uint l_kernel , uint n_kernels , uint64_t l_data , double dt , 	double f_max_analogue , 
			double f_min_analogue , np_complex_d filters , 	np_double windows , uint l_fft , int n_threads 
		); /*Windows and filter given by numpy array*/
			
		// Destructor
		~TimeQuad();
		
		// Python getters
		np_double get_ks_p(){ return ks_p.get_py_no_copy() ;};
		np_double get_ks_q(){ return ks_q.get_py_no_copy() ;};
			// Returns only the valid part of the convolution
		np_double get_ps();
		np_double get_qs();
		
		// Utilities
		inline uint l_kernel_half_c( uint l_kernel ){ return l_kernel/2+1 ;};
		inline double fft_freq( uint i , uint l_fft , double dt ){ return ((double)i)/(dt*l_fft) ;};
		inline double compute_f_Nyquist( double dt ){ return 1.0/(2.0*dt) ; };
		inline double compute_t_max_analogue( double f_min_analogue ){ return  1.0/f_min_analogue ; };
		inline uint64_t compute_l_valid( uint l_kernel, uint64_t l_data ){ return l_data - l_kernel + 1 ;} ;
		inline uint64_t compute_l_full( uint l_kernel, uint64_t l_data ){ return l_kernel + l_data - 1 ; } ;
		
		//// C++ INTERFACE
		template<class DataType>
		void execute( DataType* data , uint64_t l_data );
		
		//// Python interface
		template<class DataType>
		void execute_py( py::array_t<DataType> data );
		
	private :
		// Kernels info
		uint l_kernel ;
		uint n_kernels ;
		
		// Acquisition info
		uint64_t l_data ; //
		double dt ; // 0.03125 [ns] 
		double f_max_analogue  ; // 10 [GHz] 
		double f_min_analogue  ; // 0.5 [GHz] 
		double f_Nyquist ; //16 [GHz] = 1/(2*dt)
		double t_max_analogue ; // 2 [ns] 1/f_min_analogue
		
		// Quadratures info
		uint64_t l_valid ;
		uint64_t l_full  ;
		
		Multi_array<complex_d,2> filters ; 
		Multi_array<double,2> windows ;
		double alpha ;
		int n_threads ;
		
		// Kernels
		Multi_array<complex_d,2> ks_p_complex ; // Manages memory for ks_p
		Multi_array<complex_d,2> ks_q_complex ;
		Multi_array<double,2> ks_p ; // Uses memory managed by ks_p_complex
		Multi_array<double,2> ks_q ;
		// Quadratures
		Multi_array<double,2> ps ; 
		Multi_array<double,2> qs ;
		
		TimeQuad_algorithm* algorithm ;
		
		fftw_plan k_foward ;
		fftw_plan k_backward ;
		
		void init_gen();
		void init_direct(); // Constructor calls this function
		void init_fft( uint l_fft ); // Constructor calls this function
		
		// init sequence
			void checks();
			void checks_n_threads();
			void checks_filters();
			void checks_windows();
			void execution_checks( uint64_t l_data );
			
			void prepare_plans_kernels();
			void make_kernels();
			// void make_quadratures();
		
		// make_kernels sequence
			void vanilla_kernels(); // Generates vanilla (analitical filter at Nyquist's frequency) k_p and k_q and outputs then in ks_p[0] and ks_q[0] 
			void normalize_for_ffts();
			void copy_vanillas(); /* Copying vanilla Kernels to all other ks_p[i] and ks_q[i]*/
			void apply_filters(); // Apply the list of n_kernels custom filters to all ks_p and ks_q
			void apply_windows(); // Apply the list of n_kernels windows or the same window to every one.
			
		// Destructor sequence
			void destroy_plans_kernels();
			/* delete algorithm_Type */
}; 
