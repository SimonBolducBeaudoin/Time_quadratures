//CONSTRUCTOR

TimeQuad_FFT::TimeQuad_FFT
(	
	const Multi_array<double,2>& ks_p , const Multi_array<double,2>& ks_q ,
	const Multi_array<double,2>& ps , const Multi_array<double,2>& qs ,
	uint l_kernel , uint n_kernels , uint64_t l_data , uint  l_fft , int n_threads 
)
:
	l_kernel(l_kernel) , n_kernels(n_kernels) , l_data(l_data) , l_fft(l_fft), n_threads( n_threads ),
	l_chunk( compute_l_chunk( l_kernel,l_fft ) ) ,
	n_chunks( compute_n_chunks( l_data , l_chunk ) ) ,
	l_reste( compute_l_reste( l_data , l_chunk ) ) ,
	ks_p(ks_p) , ks_q(ks_q) , ps(ps) , qs(qs) ,
	ks_p_complex( Multi_array<complex_d,2>( n_kernels , (l_fft/2+1) , fftw_malloc , fftw_free ) ) ,
	ks_q_complex( Multi_array<complex_d,2>( n_kernels , (l_fft/2+1) , fftw_malloc , fftw_free ) ) ,
	gs( Multi_array<double,2>( n_threads , 2*(l_fft/2+1) , fftw_malloc , fftw_free ) ) ,
	fs( Multi_array<complex_d,2>( n_threads , (l_fft/2+1) , fftw_malloc , fftw_free ) ) ,
	h_ps( Multi_array<complex_d,3>( n_kernels , n_threads , (l_fft/2+1) , fftw_malloc , fftw_free ) ) ,
	h_qs( Multi_array<complex_d,3>( n_kernels , n_threads , (l_fft/2+1) , fftw_malloc , fftw_free ) )
{
	omp_set_num_threads(n_threads);
	
	prepare_plans();
	prepare_kernels();
}

// DESTRUCTOR
TimeQuad_FFT::~TimeQuad_FFT()
{	
	
    destroy_plans();
	
	// fftw_cleanup();
}

// PREPARE_PLANS METHOD

void TimeQuad_FFT::prepare_plans()
{   
	fftw_import_wisdom_from_filename("FFTW_Wisdom.dat");
	kernel_plan = fftw_plan_dft_r2c_1d( l_fft , (double*)ks_p_complex[0] , reinterpret_cast<fftw_complex*>(ks_p_complex[0]) , FFTW_EXHAUSTIVE);
	g_plan = fftw_plan_dft_r2c_1d( l_fft , gs[0] , reinterpret_cast<fftw_complex*>( fs[0] ) , FFTW_EXHAUSTIVE);
	h_plan = fftw_plan_dft_c2r_1d( l_fft, reinterpret_cast<fftw_complex*>(h_ps(0,0)) , (double*)h_ps(0,0) , FFTW_EXHAUSTIVE); /* The c2r transform destroys its input array */
	fftw_export_wisdom_to_filename("FFTW_Wisdom.dat"); 
}


void TimeQuad_FFT::destroy_plans()
{
	fftw_destroy_plan(kernel_plan); 
    fftw_destroy_plan(g_plan); 
    fftw_destroy_plan(h_plan); 
}

void TimeQuad_FFT::prepare_kernels()
{
	/*Could be parallelized*/
	for ( uint j = 0 ; j<n_kernels ; j++ ) 
	{
		/* Value assignment and zero padding */
		for( uint i = 0 ; i < l_kernel ; i++)
		{
			( (double*)ks_p_complex[j] )[i] = ks_p(j,i) ; 
			( (double*)ks_q_complex[j] )[i] = ks_q(j,i) ; 
		}
		for(uint i = l_kernel ; i < l_fft ; i++)
		{
			( (double*)ks_p_complex[j] )[i] = 0 ; 
			( (double*)ks_q_complex[j] )[i] = 0 ;
		}
		fftw_execute_dft_r2c(kernel_plan, (double*)ks_p_complex[j] , reinterpret_cast<fftw_complex*>(ks_p_complex[j]) ); 
		fftw_execute_dft_r2c(kernel_plan, (double*)ks_q_complex[j] , reinterpret_cast<fftw_complex*>(ks_q_complex[j]) ); 
	}
}
						
void TimeQuad_FFT::execute( int16_t* data )
{	
    /////////////////////
    // RESET PS AND QS //
    /////////////////////
	#pragma omp parallel
    {
        manage_thread_affinity();
        // Reset ps and qs to 0
        // Only the parts that are subject to race conditions
        #pragma omp for
		for( uint i=0; i < n_chunks-1 ; i++ )
		{		
			///// THIS FOR EACH KERNELS PAIRS
			
			for ( uint j = 0 ; j<n_kernels ; j++ ) 
			{   
                // Last l_kernel-1.0 points
				// Subject to race conditions
				for( uint k=l_chunk ; k < l_chunk + l_kernel-1 ; k++ )
				{
					ps(j,i*l_chunk+k) = 0.0 ;
					qs(j,i*l_chunk+k) = 0.0 ;
				}
			}
			
		}
    }
    /////////////////////
    
    
	#pragma omp parallel
	{	
		manage_thread_affinity();
             
		int this_thread = omp_get_thread_num();
		complex_d tmp ; // Intermediate variable 
		
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
			
			for ( uint j = 0 ; j<n_kernels ; j++ ) 
			{
				// Product
				for( uint k=0 ; k < (l_fft/2+1) ; k++ )
				{	
					tmp = fs(this_thread,k) ;
					h_ps(j,this_thread,k) = ks_p_complex(j,k) * tmp;
					h_qs(j,this_thread,k) = ks_q_complex(j,k) * tmp;
				}
				
								
				// ifft
				fftw_execute_dft_c2r(h_plan , reinterpret_cast<fftw_complex*>(h_ps(j,this_thread)) , (double*)h_ps(j,this_thread) );  
				fftw_execute_dft_c2r(h_plan , reinterpret_cast<fftw_complex*>(h_qs(j,this_thread)) , (double*)h_qs(j,this_thread) ); 
				
                // First l_kernel-1.0 points
				// Subject to race conditions
				for( uint k=0; k < l_kernel-1 ; k++ )
				{
					#pragma omp atomic update
					ps(j,i*l_chunk+k) += ( (double*)h_ps(j,this_thread))[k] /l_fft ;
					#pragma omp atomic update
					qs(j,i*l_chunk+k) += ( (double*)h_qs(j,this_thread))[k] /l_fft ;
				}
				// Copy result to p and q 
                // Not subject to race conditions
				for( uint k=l_kernel-1; k < l_chunk ; k++ )
				{	
					ps(j,i*l_chunk+k) = ( (double*)h_ps(j,this_thread))[k] / l_fft ;
					qs(j,i*l_chunk+k) = ( (double*)h_qs(j,this_thread))[k] / l_fft ;
				}
                // Last l_kernel-1.0 points
				// Subject to race conditions
				for( uint k=l_chunk ; k < l_fft ; k++ )
				{
					#pragma omp atomic update
					ps(j,i*l_chunk+k) += ( (double*)h_ps(j,this_thread))[k] /l_fft ;
					#pragma omp atomic update
					qs(j,i*l_chunk+k) += ( (double*)h_qs(j,this_thread))[k] /l_fft ;
				}
			}
			
		}
		
	}
	///// The rest ---->
	if (l_reste != 0)
	{	
		// make sure g only contains zeros
		for( uint k=0; k < l_fft ; k++ )
		{
			gs(0,k) = 0 ;
		}
		// add the rest
		for( uint k=0; k < l_reste ; k++ )
		{
			gs(0,k) = (double)data[n_chunks*l_chunk + k] ;
		}
		fftw_execute_dft_r2c(g_plan, gs[0] , reinterpret_cast<fftw_complex*>( fs[0]) );
		
		
		// Product 
		for ( uint j = 0 ; j<n_kernels ; j++ ) 
		{
			complex_d tmp;
			for( uint k=0; k < (l_fft/2+1) ; k++)
			{
				tmp = fs(0,k) ;
				h_ps(j,0,k) = ks_p_complex(j,k) * tmp;
				h_qs(j,0,k) = ks_q_complex(j,k) * tmp;
			}
			
			fftw_execute_dft_c2r(h_plan , reinterpret_cast<fftw_complex*>(h_ps(j,0)) , (double*)h_ps(j,0) );  
			fftw_execute_dft_c2r(h_plan , reinterpret_cast<fftw_complex*>(h_qs(j,0)) , (double*)h_qs(j,0) ); 
		
			// Select only the part of the ifft that contributes to the full output length
            for( uint k = 0 ; k < l_reste + l_kernel - 1 ; k++ )
			{
				ps(j,n_chunks*l_chunk+k) = ( (double*)h_ps(j,0))[k] /l_fft ;
				qs(j,n_chunks*l_chunk+k) = ( (double*)h_qs(j,0))[k] /l_fft ;
			}
		}
	}
	/////
}

