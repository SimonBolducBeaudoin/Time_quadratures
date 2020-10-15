//CONSTRUCTORS

// Macros
#define INPUTS_0 double Z , double dt , uint64_t l_data , uint kernel_conf

#define INPUTS_1_cpp 	Multi_array<complex_d,2> betas , Multi_array<complex_d,1> g

#define INPUTS_1_python np_complex_d betas , np_complex_d g

#define INPUTS_2_direct double alpha 				, int n_threads

#define INPUTS_2_fft 	double alpha , uint l_fft 	, int n_threads

#define	INIT_LIST_PART1 \
	Z(Z) , dt(dt) , l_kernel(compute_l_kernel(betas)) , l_data(l_data) , kernel_conf(kernel_conf) ,\
	n_quads(compute_n_quads(kernel_conf)) , prefactor(compute_prefactor(Z)),\
	l_valid( compute_l_valid( compute_l_kernel(betas) ,l_data ) ) , l_full( compute_l_full(  compute_l_kernel(betas) ,l_data ) )
	
#define BETAS_G_INITS \
	betas( betas ) , \
	g( g ) , \
	filters( Multi_array<complex_d,2>(compute_n_kernels(betas),compute_l_hc(betas)) ),\
	alpha(alpha) 

#define BETAS_G_NUMPY_INITS \
	betas( Multi_array<complex_d,2>::numpy(betas) ) , \
	g( Multi_array<complex_d,1>::numpy(g) ) , \
	filters( Multi_array<complex_d,2>(compute_n_kernels(betas),compute_l_hc(betas)) ),\
	alpha( alpha )
		
#define	INIT_LIST_PART2 \
	n_threads(n_threads),\
	ks_complex( Multi_array<complex_d,3>( compute_n_quads(kernel_conf) , compute_n_kernels(betas) , compute_l_hc(betas) , fftw_malloc , fftw_free ) ) , \
	ks( Multi_array<double,3>( (double*)ks_complex.get_ptr() , compute_n_quads(kernel_conf) , compute_n_kernels(betas) , compute_l_kernel(betas) , compute_l_hc(betas)*sizeof(complex_d) , sizeof(double) ) ),\
	half_norms( Multi_array<double,2>( compute_n_quads(kernel_conf) , compute_n_kernels(betas) ) ),\
	quads( Multi_array<double,3,Quads_Index_Type>( compute_n_quads(kernel_conf) , compute_n_kernels(betas) , l_full ) )

/////////////////////////////////////////
// Direct convolution constructors
template<class Quads_Index_Type>
TimeQuad<Quads_Index_Type>::TimeQuad( INPUTS_0 , INPUTS_1_cpp , INPUTS_2_direct )
: 	
	INIT_LIST_PART1 , 
	BETAS_G_INITS , 
	INIT_LIST_PART2
{ 
	init_direct(); 
};

template<class Quads_Index_Type>
TimeQuad<Quads_Index_Type>::TimeQuad( INPUTS_0 , INPUTS_1_python , INPUTS_2_direct )
: 
	INIT_LIST_PART1 , 
	BETAS_G_NUMPY_INITS , 
	INIT_LIST_PART2
{ 
	init_direct(); 
};
	
/////////////////////////////////////////
// FFT convolutionc constructors
template<class Quads_Index_Type>
TimeQuad<Quads_Index_Type>::TimeQuad( INPUTS_0 , INPUTS_1_cpp , INPUTS_2_fft )
: 	
	INIT_LIST_PART1 , 
	BETAS_G_INITS , 
	INIT_LIST_PART2
{ 
	init_fft(l_fft); 
};

template<class Quads_Index_Type>
TimeQuad<Quads_Index_Type>::TimeQuad( INPUTS_0 , INPUTS_1_python , INPUTS_2_fft )
: 
	INIT_LIST_PART1 , 
	BETAS_G_NUMPY_INITS , 
	INIT_LIST_PART2
{ 
	init_fft(l_fft); 
};
/////////////////////////////////////////

// DESTRUCTOR
template<class Quads_Index_Type>
TimeQuad<Quads_Index_Type>::~TimeQuad()
{	
	destroy_plans_kernels();
	/* 
	delete algorithm object
	*/
	
	delete algorithm;
}

// INITIALISOR/STATIC METHODS
template<class Quads_Index_Type>
uint TimeQuad<Quads_Index_Type>::compute_n_quads(uint kernel_conf)
{
	if(kernel_conf==0)
	{
		return 1 ;
	}
	else if ((kernel_conf==1) || (kernel_conf==2))
	{
		return 2 ;
	}
	else
	{
		throw std::runtime_error(" Invalid kernel_conf ");
	}
}

template<class Quads_Index_Type>
np_complex_d TimeQuad<Quads_Index_Type>::compute_flat_band(uint l_hc,double dt,double f_min_analogue,double f_max_analogue,double f_Nyquist)
{
	double f[l_hc];
	uint l_kernel = 2*l_hc - 1 ;
	for ( uint i = 0 ; i<l_hc  ; i++ ){ f[i] = fft_freq( i , l_kernel , dt ); };
	Multi_array<complex_d,1> filter (l_hc);
	Tukey_modifed_Window( filter.get_ptr() , f , l_hc , f_min_analogue , f_max_analogue , f_Nyquist ) ;
	
	return filter.copy_py() ;
}

// CHECKS 
template<class Quads_Index_Type>
void TimeQuad<Quads_Index_Type>::checks()
{
	if (l_kernel %2 != 1)
	{
		throw std::runtime_error(" l_kernel is not odd dont expect this to work... ");
	} 
	if (l_kernel > l_data)
	{
		throw std::runtime_error(" l_kernel > l_data dont expect this to work... ");
	}
}

template<class Quads_Index_Type>
void TimeQuad<Quads_Index_Type>::checks_betas()
{
	if( (betas.get_n_i() != compute_l_hc_from_l_kernel(l_kernel)) )
	{
		throw std::runtime_error(" length of betas not mathching with length of kernels ");
	}
}

template<class Quads_Index_Type>
void TimeQuad<Quads_Index_Type>::checks_g()
{
	if( (betas.get_ptr() != NULL) and (g.get_n_i() != compute_l_hc_from_l_kernel(l_kernel)) )
	{
		throw std::runtime_error(" length of g not mathching with length of kernels ");
	}
}

template<class Quads_Index_Type>
void TimeQuad<Quads_Index_Type>::execution_checks( uint64_t l_data  )
{
	if ( this->l_data != l_data )
	{
		throw std::runtime_error(" data length given during execution dont match data length declared at instentiation ");
	}
}

// PREPARE_PLANS METHOD
template<class Quads_Index_Type>
void TimeQuad<Quads_Index_Type>::prepare_plans_kernels()
{   
	fftw_import_wisdom_from_filename("FFTW_Wisdom.dat");
	k_foward 	= fftw_plan_dft_r2c_1d( l_kernel 	, ks(0,0) 									, reinterpret_cast<fftw_complex*>(ks(0,0)) 	, FFTW_ESTIMATE); // FFTW_ESTIMATE
	k_backward 	= fftw_plan_dft_c2r_1d( l_kernel	, reinterpret_cast<fftw_complex*>(ks(0,0)) 	, ks(0,0) 									, FFTW_ESTIMATE); 
	fftw_export_wisdom_to_filename("FFTW_Wisdom.dat");	
}

// FRREE METHODS
template<class Quads_Index_Type>
void TimeQuad<Quads_Index_Type>::destroy_plans_kernels()
{
	fftw_destroy_plan(k_foward); 
    fftw_destroy_plan(k_backward); 	
}

template<class Quads_Index_Type>
void TimeQuad<Quads_Index_Type>::init_gen()
{
	checks();
	// checks_n_threads();
	omp_set_num_threads(n_threads);
	checks_g();
	checks_betas();
		
	prepare_plans_kernels();
	make_kernels();
}

template<class Quads_Index_Type>
void TimeQuad<Quads_Index_Type>::init_direct()
{
	init_gen();
	
	TimeQuad_direct<Quads_Index_Type>* tmp = new TimeQuad_direct<Quads_Index_Type>	
		( ks , quads , dt , n_threads  );
		
	algorithm =  tmp ;
}

template<class Quads_Index_Type>
void TimeQuad<Quads_Index_Type>::init_fft( uint l_fft )
{
	init_gen();
	
	TimeQuad_FFT<Quads_Index_Type>* tmp = new TimeQuad_FFT<Quads_Index_Type>
		( ks , quads , dt, l_fft  , n_threads );
		
	algorithm =  tmp ;
}

template<class Quads_Index_Type>
void TimeQuad<Quads_Index_Type>::make_kernels()
{
	normalize_betas(); /* Can be done first because only depend on betas */
	
	vanilla_kernels();
	normalize_for_dfts(); 
	apply_windows();
	compute_filters();
	apply_filters();
	
    half_normalization();
}

template<class Quads_Index_Type>
void TimeQuad<Quads_Index_Type>::vanilla_kp(uint quadrature_index, uint mode_index)
{
	double t ; 	/* Abscisse positif */
	double prefact ;
	double argument ;
	double tmp1;
	
	uint k = quadrature_index;
	uint j = mode_index;
	
	for (uint i = 0 ; i < l_kernel/2; i++ ) // l_kernel doit être impaire
	{
		t = ( i + 1 )*dt;
		prefact = 2.0/sqrt(t);
		argument = sqrt( 2.0*t/dt );
		/* Right part */
		tmp1 = prefact * Fresnel_Cosine_Integral( argument ) ;
		
		ks( k , j , l_kernel/2 + 1 + i ) = tmp1;
		/* Left part */
		ks( k , j , l_kernel/2 - 1 - i ) = tmp1;
	}
	/* In zero */
	ks( k , j , l_kernel/2 ) = 2.0*sqrt(2.0)/sqrt(dt);
    
    for (uint i = 0 ; i < l_kernel; i++ )
    {
        ks( k , j , i ) *= prefactor ; /* units_correction * sqrt( 2/ Zh ) */
    }
}

template<class Quads_Index_Type>
void TimeQuad<Quads_Index_Type>::vanilla_kq(uint quadrature_index, uint mode_index)
{
	double t ; 	/* Abscisse positif */
	double prefact ;
	double argument ;
	double tmp1;
	
	uint k = quadrature_index;
	uint j = mode_index;
	
	for (uint i = 0 ; i < l_kernel/2; i++ ) // l_kernel doit être impaire
	{
		t = ( i + 1 )*dt;
		prefact = 2.0/sqrt(t);
		argument = sqrt( 2.0*t/dt );
		/* Right part */
		tmp1 = prefact * Fresnel_Sine_Integral( argument ) ;
		
		ks( k , j , l_kernel/2 + 1 + i ) = tmp1;
		/* Left part */
		ks( k , j , l_kernel/2 - 1 - i ) = (-1)*tmp1;
	}
	/* In zero */
	ks( k , j , l_kernel/2 ) = 0 ;
    
    for (uint i = 0 ; i < l_kernel; i++ )
    {
        ks( k , j , i ) *= prefactor ; /* units_correction * sqrt( 2/ Zh ) */
    }
}

template<class Quads_Index_Type>
void TimeQuad<Quads_Index_Type>::vanilla_kernels()
{
	if(kernel_conf==0)
	{
		for ( uint i = 0 ; i<n_kernels ; i++ ) 
		{	
			vanilla_kq(0,i); //quadrature_index , mode_index
		} 
	}
	else if (kernel_conf==1)
	{
		for ( uint i = 0 ; i<n_kernels ; i++ ) 
		{	
			vanilla_kp(0,i); //quadrature_index , mode_index
			vanilla_kq(0,i); //quadrature_index , mode_index
		} 
	}
	// else if ((kernel_conf==2)
	// {
		// for ( uint i = 0 ; i<n_kernels ; i++ ) 
		// {	
			// vanilla_k_pi_over_4(0,i); //quadrature_index , mode_index
			// vanilla_k_3_pi_over_4(0,i); //quadrature_index , mode_index
		// }
	// }
	else
	{
		throw std::runtime_error(" Invalid kernel_conf ");
	}
}	

template<class Quads_Index_Type>
void TimeQuad<Quads_Index_Type>::normalize_for_dfts()
{	
	double fft_norm = 1.0/l_kernel ; // Normalization for FFT's
	for ( uint k = 0 ; k<n_quads ; k++ )
	{
		for ( uint j = 0 ; j<n_kernels ; j++ )
		{
			for ( uint i = 0 ; i<l_kernel ; i++ )
			{
				ks(k,j,i) *= fft_norm ;
			}
		}
	}
}	

template<class Quads_Index_Type>
void TimeQuad<Quads_Index_Type>::compute_filters()
{
	uint 	l_f 	= compute_l_hc_from_l_kernel( l_kernel ) ;
	for ( uint j = 0 ; j<n_kernels ; j++ ) 
	{
		for ( uint i = 0 ; i<l_f ; i++ )
		{
			filters(j,i) = ( betas(j,i)/g(i) ) ;
		}
	}
}

template<class Quads_Index_Type>
void TimeQuad<Quads_Index_Type>::apply_filters()
{	
	uint 	l_f 	= compute_l_hc_from_l_kernel( l_kernel ) ;
	
	/* Foward transforms */
	for ( uint k = 0 ; k<n_quads ; k++ )
	{
		for ( uint j = 0 ; j<n_kernels ; j++ )
		{
			fftw_execute_dft_r2c( k_foward, ks(k,j) , reinterpret_cast<fftw_complex*>( ks(k,j) ) ); 
		}
	}
	/* Apply betas */
	for ( uint k = 0 ; k<n_quads ; k++ )
	{
		for ( uint j = 0 ; j<n_kernels ; j++ ) 
		{
			for ( uint i = 0 ; i<l_f ; i++ )
			{
				ks_complex(k,j,i) *= filters(j,i) ;
			}
		}
	}
	/* Returning to real space */
	for ( uint k = 0 ; k<n_quads ; k++ )
	{
		for ( uint j = 0 ; j<n_kernels ; j++ ) 
		{
			/* c2r destroys its inputs array */
			fftw_execute_dft_c2r(k_backward, reinterpret_cast<fftw_complex*>(ks(k,j)) , ks(k,j) ); 
		}
	}
}

template<class Quads_Index_Type>
void TimeQuad<Quads_Index_Type>::normalize_betas()
{
	for ( uint j = 0 ; j<n_kernels ; j++ )
	{
		normalize_a_beta(j);
	}
}

template<class Quads_Index_Type>
void TimeQuad<Quads_Index_Type>::normalize_a_beta(uint mode_index)
{
	uint j = mode_index;
	
	double sum = 0 ;
	
	for ( uint i = 0 ; i<l_kernel ; i++ )
	{
		sum += std::norm(betas(j,i));
	}
	sum *= dt ; 
	sum = sqrt(sum);
	
	for ( uint i = 0 ; i<l_kernel ; i++ )
	{
		betas(j,i) /= sum ;
	}
}

template<class Quads_Index_Type>
void TimeQuad<Quads_Index_Type>::apply_windows()
{
	for ( uint k = 0 ; k<n_quads ; k++ )
	{
		for ( uint j = 0 ; j<n_kernels ; j++ ) 
		{
			Tukey_Window( ks(k,j) , alpha , l_kernel ) ;
		}
	}
}

template<class Quads_Index_Type>
void TimeQuad<Quads_Index_Type>::half_normalization()
{	
	for ( uint k = 0 ; k<n_quads ; k++ )
	{
		for ( uint j = 0 ; j<n_kernels ; j++ ) 
		{
			double tmp = 0;
			for ( uint i = 0 ; i<l_kernel ; i++ )
			{
				tmp += ks(k,j,i)*ks(k,j,i);
			}
			
			half_norms(k,j) = sqrt(tmp);
			
			for ( uint i = 0 ; i<l_kernel ; i++ )
			{
				ks(k,j,i) /= half_norms(k,j);
			}
		}
	}
}

template<class Quads_Index_Type>
void TimeQuad<Quads_Index_Type>::half_denormalization()
{	
	for ( uint k = 0 ; k<n_quads ; k++ )
	{
		for ( uint j = 0 ; j<n_kernels ; j++ ) 
		{
			for ( uint i = 0 ; i<l_kernel ; i++ )
			{
				ks(k,j,i) *= half_norms(k,j);
			}
		}
	}	
}

template<class Quads_Index_Type>
template<class DataType>
void TimeQuad<Quads_Index_Type>::execute( Multi_array<DataType,1,uint64_t>& data )
{
	execution_checks( data.get_n_i() );
	algorithm->execute( data );	
}

template<class Quads_Index_Type>
template<class DataType>
void TimeQuad<Quads_Index_Type>::execute_py( py::array_t<DataType, py::array::c_style>& np_data )
{
	Multi_array<DataType,1,uint64_t> data = Multi_array<DataType,1,uint64_t>::numpy(np_data) ;
	execute( data );
}

template<class Quads_Index_Type>
np_double TimeQuad<Quads_Index_Type>::get_quads()
{
	/*
	Pybind11 doesn't work with uint64_t 
		This coul potentially cause problems with l_data, l_valid and l_full
	*/
	// Numpy will not copy the array when using the assignement operator=
	double* ptr = quads[0] + l_kernel -1 ;
	py::capsule free_dummy(	ptr, [](void *f){;} );
	
	return py::array_t<double, py::array::c_style>
	(
		{uint(n_quads) ,uint(n_kernels), uint(l_valid) },      // shape
		{quads.get_stride_k(),quads.get_stride_j(), quads.get_stride_i() },   // C-style contiguous strides for double
		ptr  ,       // the data pointer
		free_dummy // numpy array references this parent
	);
}

// MACROS UNDEF
#undef INPUTS_0 
#undef INPUTS_1_cpp 
#undef INPUTS_1_python 
#undef INPUTS_2_direct
#undef INPUTS_2_fft
#undef BETAS_G_INITS
#undef BETAS_G_NUMPY_INITS
#undef INIT_LIST_PART1
#undef INIT_LIST_PART2
