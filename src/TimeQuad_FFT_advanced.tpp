//CONSTRUCTOR

template<class Quads_Index_Type>
TimeQuad_FFT_advanced<Quads_Index_Type>::TimeQuad_FFT_advanced
(	
	const Multi_array<double,2>& ks_p , const Multi_array<double,2>& ks_q ,
	const Multi_array<double,2,Quads_Index_Type>& ps , const Multi_array<double,2,Quads_Index_Type>& qs ,
	uint l_kernel , uint n_kernels , uint64_t l_data , uint  l_fft , uint howmany, int n_threads 
)
:
	l_kernel(l_kernel) , n_kernels(n_kernels) , l_data(l_data) , 
	l_fft(l_fft), l_hc(compute_l_hc(l_fft)) ,
	howmany(howmany),n_threads( n_threads ),
	l_chunk( compute_l_chunk( l_kernel,l_fft ) ) ,
	n_chunks( compute_n_chunks( l_data , l_chunk ) ) ,
	l_reste( compute_l_reste( l_data , l_chunk ) ) ,
	n_pieces( compute_n_piece	( n_chunks, howmany) ),
	ks_p(ks_p) , ks_q(ks_q) , ps(ps) , qs(qs) ,
	ks_p_complex( Multi_array<complex_d,2>( n_kernels , (l_fft/2+1) , fftw_malloc , fftw_free ) ) ,
	ks_q_complex( Multi_array<complex_d,2>( n_kernels , (l_fft/2+1) , fftw_malloc , fftw_free ) ) ,
	gs( Multi_array<double,2>( n_threads , howmany*l_fft , fftw_malloc , fftw_free ) ) ,
	fs( Multi_array<complex_d,2>( n_threads , howmany*(l_fft/2+1) , fftw_malloc , fftw_free ) ) ,
	h_ps( Multi_array<complex_d,3>( n_kernels , n_threads , howmany*(l_fft/2+1) , fftw_malloc , fftw_free ) ) ,
	h_qs( Multi_array<complex_d,3>( n_kernels , n_threads , howmany*(l_fft/2+1) , fftw_malloc , fftw_free ) )
{
	omp_set_num_threads(n_threads);
	
	/*
		check that l_fft is significantly (at least a factor 2 to be efficient) larger than l_kernel
	*/
	prepare_plans();
	prepare_kernels();
	
	// printf("l_kernel = %d \n n_kernels = %d \n l_data = %d \n l_fft = %d \n howmany = %d\n n_threads = %d\n",l_kernel,n_kernels,l_data,l_fft,howmany,n_threads);
}

// DESTRUCTOR
template<class Quads_Index_Type>
TimeQuad_FFT_advanced<Quads_Index_Type>::~TimeQuad_FFT_advanced()
{	
	
    destroy_plans();
	
	// fftw_cleanup();
}

// PREPARE_PLANS METHOD
template<class Quads_Index_Type>
void TimeQuad_FFT_advanced<Quads_Index_Type>::prepare_plans()
{   
	fftw_import_wisdom_from_filename("FFTW_Wisdom.dat");
	kernel_plan = fftw_plan_dft_r2c_1d( l_fft , (double*)ks_p_complex[0] , reinterpret_cast<fftw_complex*>(ks_p_complex[0]) , FFTW_EXHAUSTIVE);
	
	int n[] = {(int)l_fft};
	g_plan = 
		fftw_plan_many_dft_r2c
		( 
			1 , // rank
			n , //  list of dimensions 
			howmany , // howmany (to do many ffts on the same core)
			gs[0] , // input
			NULL , // inembed
			1 , // istride
			0 , // idist
			reinterpret_cast<fftw_complex*>( fs[0] ) , // output pointer
			NULL , //  onembed
			1 , // ostride
			0 , // odist
			FFTW_EXHAUSTIVE
		);
	h_plan = 
		fftw_plan_many_dft_c2r
		(
			1 , // rank
			n , //  list of dimensions 
			howmany , // howmany (to do many ffts on the same core)
			reinterpret_cast<fftw_complex*>(h_ps(0,0)) , // input
			NULL , // inembed
			1 , // istride
			0 , // idist
			(double*)h_ps(0,0) , // output pointer
			NULL , //  onembed
			1 , // ostride
			0 , // odist
			FFTW_EXHAUSTIVE
		); /* The c2r transform destroys its input array */
	fftw_export_wisdom_to_filename("FFTW_Wisdom.dat"); 
}

template<class Quads_Index_Type>
void TimeQuad_FFT_advanced<Quads_Index_Type>::destroy_plans()
{
	fftw_destroy_plan(kernel_plan); 
    fftw_destroy_plan(g_plan); 
    fftw_destroy_plan(h_plan); 
}

template<class Quads_Index_Type>
void TimeQuad_FFT_advanced<Quads_Index_Type>::prepare_kernels()
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

template<class Quads_Index_Type>						
void TimeQuad_FFT_advanced<Quads_Index_Type>::execute( int16_t* data )
{	
	// printf("line number %d in file %s\n", __LINE__, __FILE__);
    /////////////////////
    // RESET PS AND QS //
    /////////////////////
	#pragma omp parallel
    {
		// printf("line number %d in file %s\n", __LINE__, __FILE__);
        manage_thread_affinity();
        // Reset ps and qs to 0
        // Only the parts that are subject to race conditions
        /*
            Possible optimizations :
                - Inverse i and j loop (DONE)
                - Use collapse of nested loops : https://pages.tacc.utexas.edu/~eijkhout/pcse/html/omp-loop.html (DONE)
                - SIMD instructions : https://www.openmp.org/spec-html/5.0/openmpsu42.html (DONE)
				- Slipt into 2 loops so that data is more local (DONE)
        */
		// printf("line number %d in file %s\n", __LINE__, __FILE__);
		///// THIS FOR EACH KERNELS PAIRS
		/*
		#pragma omp for simd collapse(3)
		for( uint i=0; i < n_chunks-1 ; i++ )
		{		
			///// THIS FOR EACH KERNELS PAIRS	
			for ( uint j = 0 ; j<n_kernels ; j++ ) 
			{   
                // Last l_kernel-1.0 points
				// Subject to race conditions
				for( uint k=l_chunk ; k < l_fft; k++ )
				{
					ps(j,i*l_chunk+k) = 0.0 ;
					qs(j,i*l_chunk+k) = 0.0 ;
				}
			}	
		}
		*/
		
		/*
			Removed unecessary slipt of the threads
		*/
		
		int this_thread = omp_get_thread_num();
		complex_d tmp ; // Intermediate variable 
		for( uint k=0; k < howmany*l_fft ; k++ )
		{
			gs(this_thread,k) = 0;
		}
		
	//// Loop on chunks ---->
		// printf("line number %d in file %s\n", __LINE__, __FILE__);
		#pragma omp for
		for( uint i=0; i < n_pieces ; i++ )
		{
			// printf("%d/%d",i,n_pieces);
			
			for( uint j=0; j < howmany ; j++ )
			{
				for(uint k=0 ; k < l_chunk ; k++ )
				{
					gs(this_thread,l_fft*j+k) = (double)data[(i*howmany+j)*l_chunk + k] ; // Cast data to double
				}
			}
			
			
			fftw_execute_dft_r2c(g_plan,gs[this_thread],reinterpret_cast<fftw_complex*>(fs[this_thread]));
			
			/////
			///// THIS FOR EACH KERNELS PAIRS
			for ( uint j = 0 ; j<n_kernels ; j++ ) 
			{
				// Product
				for( uint k=0; k < howmany ; k++ )
				{
					for( uint l=0 ; l < l_hc ; l++ )
					{	
						tmp = fs(this_thread,k*l_hc+l) ;
						h_ps(j,this_thread,k*l_hc+l) = ks_p_complex(j,l) * tmp;
						h_qs(j,this_thread,k*l_hc+l) = ks_q_complex(j,l) * tmp;
					}
				}
				
				// ifft
				// For advanced interface out of place transform would be faster
				/*
				fftw_execute_dft_c2r(h_plan , reinterpret_cast<fftw_complex*>(h_ps(j,this_thread)) , (double*)h_ps(j,this_thread) );  
				fftw_execute_dft_c2r(h_plan , reinterpret_cast<fftw_complex*>(h_qs(j,this_thread)) , (double*)h_qs(j,this_thread) ); 
				*/
				/*
				for( uint k=0; k < howmany ; k++ )
				{
					// First l_kernel-1 points
					// Subject to race conditions
					for( uint l=0; l < l_kernel-1 ; l++ )
					{
						#pragma omp atomic update
						ps(j,(i*howmany+k)*l_chunk+l) += ((double*)h_ps(j,this_thread))[k*2*l_hc+l] /l_fft ;
						#pragma omp atomic update
						qs(j,(i*howmany+k)*l_chunk+l) += ((double*)h_qs(j,this_thread))[k*2*l_hc+l] /l_fft ;
					}
					// Copy result to p and q 
					for( uint l=l_kernel-1; l < l_chunk ; l++ )
					{	
						ps(j,(i*howmany+k)*l_chunk+l) = ((double*)h_ps(j,this_thread))[k*2*l_hc+l] / l_fft ;
						qs(j,(i*howmany+k)*l_chunk+l) = ((double*)h_qs(j,this_thread))[k*2*l_hc+l] / l_fft ;
					}
					// Last l_kernel-1.0 points
					// Subject to race conditions
					for( uint l=l_chunk ; l < l_fft ; l++ )
					{
						#pragma omp atomic update
						ps(j,(i*howmany+k)*l_chunk+l) += ((double*)h_ps(j,this_thread))[k*2*l_hc+l] /l_fft ;
						#pragma omp atomic update
						qs(j,(i*howmany+k)*l_chunk+l) += ((double*)h_qs(j,this_thread))[k*2*l_hc+l] /l_fft ;
					}
				}
				*/
			}
		}
	}
	///// The rest ---->
	/*
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
				ps(j,n_chunks*l_chunk+k) += ( (double*)h_ps(j,0))[k] /l_fft ;
				qs(j,n_chunks*l_chunk+k) += ( (double*)h_qs(j,0))[k] /l_fft ;
			}
		}
	}
	*/
	/////
	
}