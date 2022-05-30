//CONSTRUCTOR

template<class Quads_Index_Type>
TimeQuad_FFT<Quads_Index_Type>::TimeQuad_FFT
(	
	const Multi_array<double,3>& 					ks , 
	const Multi_array<double,3,Quads_Index_Type>& 	quads , 
	double dt , 
	uint  l_fft , 
	int n_threads 
)
:
	n_quads		(compute_n_quads(ks))					,
	n_kernels	(compute_n_kernels(ks))					, 
	l_kernel	(compute_l_kernels(ks))					, 
	dt			(dt) 									, 
	l_fft		(l_fft)									, 
	n_threads	(n_threads)								,
	l_chunk		(compute_l_chunk(l_kernel,l_fft)) 		,
	ks			(ks) 									, 
	quads		(quads) 								,
	ks_complex	( Multi_array<complex_d,3>	(n_quads,n_kernels,(l_fft/2+1)	,fftw_malloc,fftw_free) ) 	,
	gs			( Multi_array<double,2>		(n_threads,2*(l_fft/2+1),fftw_malloc,fftw_free) ) 	,
	fs			( Multi_array<complex_d,2>	(n_threads,(l_fft/2+1)	,fftw_malloc,fftw_free) ) 	,
	hs( Multi_array<complex_d,4>			(compute_n_quads(ks),n_kernels,n_threads,(l_fft/2+1),fftw_malloc,fftw_free) )
{
	omp_set_num_threads(n_threads);
	
	prepare_plans();
	prepare_kernels();
}

// DESTRUCTOR
template<class Quads_Index_Type>
TimeQuad_FFT<Quads_Index_Type>::~TimeQuad_FFT()
{	
    destroy_plans();
	// fftw_cleanup();
}

// PREPARE_PLANS METHOD
template<class Quads_Index_Type>
void TimeQuad_FFT<Quads_Index_Type>::prepare_plans()
{   
	fftw_import_wisdom_from_filename("FFTW_Wisdom.dat");
	kernel_plan = fftw_plan_dft_r2c_1d	( l_fft , (double*)ks_complex(0,0) 	, reinterpret_cast<fftw_complex*>(ks_complex(0,0)) 	, FFTW_EXHAUSTIVE);
	g_plan = fftw_plan_dft_r2c_1d		( l_fft , gs[0] 					, reinterpret_cast<fftw_complex*>( fs[0] ) 			, FFTW_EXHAUSTIVE);
	h_plan = fftw_plan_dft_c2r_1d		( l_fft , reinterpret_cast<fftw_complex*>(hs(0,0,0)) , (double*)hs(0,0,0) 				, FFTW_EXHAUSTIVE); /* The c2r transform destroys its input array */
	fftw_export_wisdom_to_filename("FFTW_Wisdom.dat"); 
}

template<class Quads_Index_Type>
void TimeQuad_FFT<Quads_Index_Type>::destroy_plans()
{
	fftw_destroy_plan(kernel_plan); 
    fftw_destroy_plan(g_plan); 
    fftw_destroy_plan(h_plan); 
}

template<class Quads_Index_Type>
void TimeQuad_FFT<Quads_Index_Type>::prepare_kernels()
{
	/*Could be parallelized*/
	for ( uint k = 0 ; k<n_quads ; k++ ) 
	{
		for ( uint j = 0 ; j<n_kernels ; j++ ) 
		{
			/* Value assignment and zero padding */
			for( uint i = 0 ; i < l_kernel ; i++)
			{
				( (double*)ks_complex(k,j) )[i] = ks(k,j,i) ; 
			}
			for(uint i = l_kernel ; i < l_fft ; i++)
			{
				( (double*)ks_complex(k,j) )[i] = 0 ; 
			}
			fftw_execute_dft_r2c(kernel_plan, (double*)ks_complex(k,j) , reinterpret_cast<fftw_complex*>(ks_complex(k,j)) ); 
		}
	}
}

template<class Quads_Index_Type>						
void TimeQuad_FFT<Quads_Index_Type>::execute( Multi_array<int16_t,1,uint64_t>& data )
{	
	uint64_t l_data = 	data.get_n_i();
	uint n_chunks 	=	compute_n_chunks	(l_data,l_chunk);
	uint l_reste 	=	compute_l_reste		(l_data,l_chunk);

    double norm_factor = dt/l_fft; /*THIS SHOULD BE MOVED IN PREPARE_KERNELS*/
    /////////////////////
    // RESET PS AND QS //
    /////////////////////
	#pragma omp parallel
    {
        manage_thread_affinity();
        // Reset ps and qs to 0
        // Only the parts that are subject to race conditions
        /*
            Possible optimizations :
                - Inverse i and j loop
                - Use collapse of nested loops : https://pages.tacc.utexas.edu/~eijkhout/pcse/html/omp-loop.html
                - SIMD instructions : https://www.openmp.org/spec-html/5.0/openmpsu42.html
        */
        
		
        #pragma omp for simd collapse(4)
		for ( uint l = 0 ; l<n_quads ; l++ ) 
		{
			for ( uint j = 0 ; j<n_kernels ; j++ ) 
			{  
				for( uint i=0; i < n_chunks-1 ; i++ )
				{	
					// Last l_kernel-1.0 points
					// Subject to race conditions
					for( uint k=l_chunk ; k < l_fft; k++ )
					{	
						quads(l,j,i*l_chunk+k) = 0.0 ;
					}
				}	
			}
		}
    }
	if (l_reste != 0)
	{			
		// Product 
		for ( uint l = 0 ; l<n_quads ; l++ ) 
		{
			for ( uint j = 0 ; j<n_kernels ; j++ ) 
			{
				// Select only the part of the ifft that contributes to the full output length
				for( uint k = 0 ; k < l_reste + l_kernel - 1 ; k++ )
				{
					quads(l,j,n_chunks*l_chunk+k) = 0.0 ;
				}
			}
		}
	}
	///////////////////////
    // COMPUTE PS AND QS //
    ///////////////////////
	#pragma omp parallel
	{	
		manage_thread_affinity();
             
		int this_thread = omp_get_thread_num();
		
		 // Reset g to zeros.
		for( uint k=0; k < l_fft ; k++ )
		{
			gs(this_thread,k) = 0;
		}
		
	
	//// Loop on chunks ---->
		
		#pragma omp for
		for( uint i=0; i < n_chunks ; i++ )
		{
			///// THIS ONLY ONCE
			// fft_data ///
			
			for( uint j=0 ; j < l_chunk ; j++ )
			{
				gs(this_thread,j) = (double)data[i*l_chunk + j] ; // Cast data to double
			}
			
			fftw_execute_dft_r2c( g_plan, gs[this_thread] , reinterpret_cast<fftw_complex*>( fs[this_thread] ) );
			
			/////
			
			///// THIS FOR EACH KERNELS PAIRS
			for ( uint l = 0 ; l<n_quads ; l++ ) 
			{
				for ( uint j = 0 ; j<n_kernels ; j++ ) 
				{
					// Product	
					for( uint k=0 ; k < (l_fft/2+1) ; k++ )
					{	
						hs(l,j,this_thread,k) = ks_complex(l,j,k) * fs(this_thread,k);
					}
					
					
					// ifft
					fftw_execute_dft_c2r(h_plan , reinterpret_cast<fftw_complex*>(hs(l,j,this_thread)) , (double*)hs(l,j,this_thread) );   
					
					// First l_kernel-1.0 points
					// Subject to race conditions
					for( uint k=0; k < l_kernel-1 ; k++ )
					{
						#pragma omp atomic update
						quads(l,j,i*l_chunk+k) += ( (double*)hs(l,j,this_thread))[k] * norm_factor ;
					}
					// Copy result to p and q 
					// Not subject to race conditions
					for( uint k=l_kernel-1; k < l_chunk ; k++ )
					{	
						quads(l,j,i*l_chunk+k) = ( (double*)hs(l,j,this_thread))[k] * norm_factor ;
					}
					// Last l_kernel-1.0 points
					// Subject to race conditions
					for( uint k=l_chunk ; k < l_fft ; k++ )
					{
						#pragma omp atomic update
						quads(l,j,i*l_chunk+k) += ( (double*)hs(l,j,this_thread))[k] * norm_factor ;
					}
				}
			}
		}
	}
	///// The rest ---->
	if (l_reste != 0)
	{	
        // add the rest
        uint k=0 ;
		for(; k < l_reste ; k++ )
		{
			gs(0,k) = (double)data[n_chunks*l_chunk + k] ;
		}
		// make sure g only contains zeros
		for(; k < l_fft ; k++ )
		{
			gs(0,k) = 0 ;
		}
		
		fftw_execute_dft_r2c(g_plan, gs[0] , reinterpret_cast<fftw_complex*>( fs[0]) );
		
		// Product 
		for ( uint l = 0 ; l<n_quads ; l++ ) 
		{
			for ( uint j = 0 ; j<n_kernels ; j++ ) 
			{
				complex_d tmp;
				for( uint k=0; k < (l_fft/2+1) ; k++)
				{
					tmp = fs(0,k) ;
					hs(l,j,0,k) = ks_complex(l,j,k) * tmp;
				}
				
				fftw_execute_dft_c2r(h_plan , reinterpret_cast<fftw_complex*>(hs(l,j,0)) , (double*)hs(l,j,0) );  
			
				// Select only the part of the ifft that contributes to the full output length
				for( uint k = 0 ; k < l_reste + l_kernel - 1 ; k++ )
				{
					quads(l,j,n_chunks*l_chunk+k) += ( (double*)hs(l,j,0))[k] * norm_factor ;
				}
			}
		}
	}
	/////
}