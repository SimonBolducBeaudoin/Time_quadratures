//CONSTRUCTOR
TimeQuad_direct::TimeQuad_direct
( 
	const Multi_array<double,2>& ks_p , const Multi_array<double,2>& ks_q ,
	const Multi_array<double,2>& ps , const Multi_array<double,2>& qs ,
	uint l_kernel , uint n_kernels , uint64_t l_data , int n_threads
)
: 
	l_kernel(l_kernel) , n_kernels(n_kernels) , l_data(l_data) ,
	l_full(l_kernel + l_data - 1) , n_threads( n_threads ) ,
	ks_p(ks_p) , ks_q(ks_q),
	ps(ps) , qs(qs)
{ omp_set_num_threads(n_threads); }

								
void TimeQuad_direct::execute( int16_t* data )
{	
	/* This is not optimized */
	for ( uint i = 0 ; i<n_kernels ; i++ ) 
	{
		conv_directe( data , ks_p[i] , ks_q[i] , ps[i] , qs[i] );
	}
}

template<class DataType>
void TimeQuad_direct::conv_directe( DataType* data , double* k_p , double* k_q , double* p , double* q  )
{	
	// Begenning
	for(uint64_t i = 0 ; i < l_kernel - 1 ; i++)
	{
		for(uint64_t j = 0; j <= i  ; j++)
		{	
			p[i] += (double)(k_p[j]* data[(i - j)]) ;
			q[i] += (double)(k_q[j]* data[(i - j)]) ;
		}
	}
	// Middle (This is also the valid part)
	uint64_t translation =  l_kernel - 1;

	/*Could be parallelized*/
	for(uint64_t i = 0 ; i < l_data - l_kernel + 1 ; i++)
	{
		for(uint64_t j=0; j < l_kernel ; j++)
		{
			p[i+translation] += data[ (l_kernel - 1) - j + i]*k_p[j] ;
			q[i+translation] += data[ (l_kernel - 1) - j + i]*k_q[j] ;
		}
	}
	// end
	uint64_t end_full = l_full -1;
	
	for(uint64_t i=0; i < l_kernel - 1; i++)
	{
		for(uint64_t j=0; j <= i ; j++)
		{
			p[end_full-i] += k_p[(l_kernel-1)-j]*data[(l_data - 1)-i+j] ;
			q[end_full-i] += k_q[(l_kernel-1)-j]*data[(l_data - 1)-i+j] ;
		}
	}
}