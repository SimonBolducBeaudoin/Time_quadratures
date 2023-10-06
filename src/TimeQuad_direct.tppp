//CONSTRUCTOR
template<class Quads_Index_Type>
TimeQuad_direct<Quads_Index_Type>::TimeQuad_direct
( 
	const Multi_array<double,3>& 					ks , 
	const Multi_array<double,3,Quads_Index_Type>& 	quads ,
	double dt ,
	int n_threads
)
: 
	n_quads		(compute_n_quads(ks)			),
	n_kernels	(compute_n_kernels(ks)			),
	l_kernel	(compute_l_kernels(ks)			),
	l_full		(compute_l_full(quads)			),
	dt			(dt 							),
	n_threads	(n_threads 						),
	ks			(ks								),
	quads		(quads							)
{ 
	omp_set_num_threads(n_threads); 
}

template<class Quads_Index_Type>							
void TimeQuad_direct<Quads_Index_Type>::execute(  Multi_array<int16_t,1,uint64_t>& data )
{	
	uint64_t l_data = data.get_n_i();
	
	double prefactor = dt;

	/* This is not optimized */
	for ( uint l = 0 ; l<n_quads ; l++ )
	{
		for ( uint k = 0 ; k<n_kernels ; k++ ) 
		{
			
			
			
			
			for(uint64_t i = 0 ; i < l_kernel - 1 ; i++)
			{
				for(uint64_t j = 0; j <= i  ; j++)
				{	
					quads(l,k,i) += prefactor*ks(l,k,j)*data[i - j] ;
				}
			}
			// Middle (This is also the valid part)
			uint64_t translation =  l_kernel - 1;

			/*Could be parallelized*/
			#pragma GCC ivdep
			for(uint64_t i = 0 ; i < l_data - l_kernel + 1 ; i++)
			{
				#pragma GCC ivdep
				for(uint64_t j=0; j < l_kernel ; j++)
				{
					quads(l,k,i+translation) += prefactor*data[ (l_kernel-1)-j+i ]*ks(l,k,j) ;
				}
			}
			// end
			uint64_t end_full = l_full - 1 ;
			
			for(uint64_t i=0; i < l_kernel - 1; i++)
			{
				for(uint64_t j=0; j <= i ; j++)
				{
					quads(l,k,end_full-i) += prefactor*ks(l,k,(l_kernel-1)-j)*data[ (l_data-1)-i+j ] ;
				}
			}
			
			
			
			
		}
	}
}