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
	const Multi_array<double,3>& ks , 
	const Multi_array<double,3,Quads_Index_Type>& quads ,
	double dt ,
	int n_threads
	);
    
	// Destructor
	~TimeQuad_direct(){};
	
	// Utilities
	uint compute_n_quads			(const Multi_array<double,3>& ks )						{ return ks.get_n_k()			;};
	uint compute_n_kernels			(const Multi_array<double,3>& ks )						{ return ks.get_n_j()			;};
	uint compute_l_kernels			(const Multi_array<double,3>& ks )						{ return ks.get_n_i()			;};
	inline uint64_t compute_l_full	(const Multi_array<double,3,Quads_Index_Type>& 	quads )	{ return quads.get_n_i()		;};
	
	# define EXECUTE(DataType) \
		void execute( Multi_array<DataType,1,uint64_t>& data ) override ;	
	
	EXECUTE(int16_t)
							
	#undef EXECUTE

    void update_kernels(){/*Pass*/};
	
	private :
	// Kernels info
	uint 		n_quads ;
	uint 		n_kernels ;
	uint 		l_kernel ;	
	uint64_t 	l_full;
	double 		dt;
	int 		n_threads ;
	
	// Kernels
	const Multi_array<double,3>& ks ; 
	// Quadratures
	const Multi_array<double,3,Quads_Index_Type>& quads ; 
	
};

#include "../src/TimeQuad_direct.tpp"