#pragma once

#include <omp_extra.h>
#include <Multi_array.h>
#include <TimeQuad_algorithm.h>

typedef unsigned int uint;

template<class Quads_Index_Type=uint>
class TimeQuad_direct : public TimeQuad_algorithm
{
	public :
	// Contructor
	TimeQuad_direct
	( 
	const Multi_array<double,2>& ks_p , 
	const Multi_array<double,2>& ks_q ,
	const Multi_array<double,2,Quads_Index_Type>& ps , 
	const Multi_array<double,2,Quads_Index_Type>& qs ,
	uint l_kernel , uint n_kernels , uint64_t l_data , int n_threads
	);
    
    
	// Destructor
	~TimeQuad_direct(){};
	
	# define EXECUTE(DataType) \
		void execute( DataType* data ) override ;	
	
	EXECUTE(int16_t)
							
	#undef EXECUTE
		
	private :
	// Kernels info
	uint l_kernel ;
	uint n_kernels ;
	
	// Acquisition info
	uint64_t l_data ; 
	
	// Quadratures info
	uint64_t l_full ;

	int n_threads ;
	
	// Kernels
	const Multi_array<double,2>& ks_p ; 
	const Multi_array<double,2>& ks_q ;
	// Quadratures
	const Multi_array<double,2,Quads_Index_Type>& ps ; 
	const Multi_array<double,2,Quads_Index_Type>& qs ;
	
	template<class DataType>
	void conv_directe( DataType* data , double* k_p , double* k_q , double* p , double* q  );
};

#include "../src/TimeQuad_direct.tpp"