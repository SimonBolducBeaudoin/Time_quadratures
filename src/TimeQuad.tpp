//CONSTRUCTORS

// Macros
#define KS_INPUTS uint l_kernel , uint n_kernels

#define AC_INPUTS  uint64_t l_data , double dt , double f_max_analogue , double f_min_analogue

#define KS_INITS l_kernel(l_kernel) , n_kernels(n_kernels)

#define AC_INITS l_data(l_data) , dt(dt) , f_max_analogue(f_max_analogue) , f_min_analogue(f_min_analogue) ,\
				 f_Nyquist(compute_f_Nyquist(dt)) , t_max_analogue(compute_t_max_analogue(f_min_analogue))
				 
#define QUADS_INITS l_valid( compute_l_valid( l_kernel,l_data ) ) , l_full( compute_l_full( l_kernel,l_data ) )

#define FILTERS_WINDOWS_NULL_INITS \
	filters( Multi_array<complex_d,2>((complex_d*)NULL,0,0) ) , \
	windows( Multi_array<double,2>((double*)NULL,0,0) ) , \
	alpha( alpha )

#define FILTERS_WINDOWS_INITS filters( filters ) , windows( windows ) , alpha(0.25) 

#define FILTERS_NUMPY_INITS \
	filters( Multi_array<complex_d,2>::numpy(filters) ) , \
	windows( Multi_array<double,2>((double*)NULL,0,0) ) , alpha( alpha )
	
#define FILTERS_WINDOWS_NUMPY_INITS \
	filters( Multi_array<complex_d,2>::numpy(filters) ) , \
	windows( Multi_array<double,2>::numpy(windows) ) , alpha( 0.25 )
	
#define KS_PQ_INITS \
	ks_p_complex( Multi_array<complex_d,2>( n_kernels , l_kernel_half_c(l_kernel) , fftw_malloc , fftw_free ) ) , \
	ks_q_complex( Multi_array<complex_d,2>( n_kernels , l_kernel_half_c(l_kernel) , fftw_malloc , fftw_free ) ) , \
	ks_p( Multi_array<double,2>( (double*)ks_p_complex.get() , n_kernels , l_kernel , l_kernel_half_c(l_kernel)*sizeof(complex_d) , sizeof(double) ) ) , \
	ks_q( Multi_array<double,2>( (double*)ks_q_complex.get() , n_kernels , l_kernel , l_kernel_half_c(l_kernel)*sizeof(complex_d) , sizeof(double) ) )  

#define PQS_INITS \
	ps( Multi_array<double,2>( n_kernels , l_full ) ) , \
	qs( Multi_array<double,2>( n_kernels , l_full ) ) \

/////////////////////////////////////////
// Direct convolution constructors
TimeQuad::TimeQuad(	KS_INPUTS , AC_INPUTS , double alpha , int n_threads )
	: KS_INITS , AC_INITS , QUADS_INITS , FILTERS_WINDOWS_NULL_INITS , n_threads( n_threads ) , KS_PQ_INITS , PQS_INITS
	{ init_direct(); };

TimeQuad::TimeQuad( KS_INPUTS , AC_INPUTS , Multi_array<complex_d,2> filters , Multi_array<double,2> windows , int n_threads)
	: KS_INITS , AC_INITS , QUADS_INITS , FILTERS_WINDOWS_INITS , n_threads( n_threads ) , KS_PQ_INITS , PQS_INITS
	{ init_direct(); };

TimeQuad::TimeQuad( KS_INPUTS , AC_INPUTS , np_complex_d filters , double alpha , int n_threads )
	: KS_INITS , AC_INITS , QUADS_INITS , FILTERS_NUMPY_INITS , n_threads( n_threads ) , KS_PQ_INITS , PQS_INITS
	{ init_direct(); };

TimeQuad::TimeQuad( KS_INPUTS , AC_INPUTS , np_complex_d filters , np_double windows , int n_threads )
	: KS_INITS , AC_INITS , QUADS_INITS , FILTERS_WINDOWS_NUMPY_INITS , n_threads( n_threads ) , KS_PQ_INITS , PQS_INITS
	{ init_direct(); };
	
/////////////////////////////////////////
// FFT convolutionc constructors
TimeQuad::TimeQuad(	KS_INPUTS , AC_INPUTS , double alpha , uint l_fft , int n_threads )
	: KS_INITS , AC_INITS , QUADS_INITS , FILTERS_WINDOWS_NULL_INITS , n_threads( n_threads ) , KS_PQ_INITS , PQS_INITS
	{ 
	
	init_fft(l_fft); };

TimeQuad::TimeQuad( KS_INPUTS , AC_INPUTS , Multi_array<complex_d,2> filters , Multi_array<double,2> windows , uint l_fft , int n_threads)
	: KS_INITS , AC_INITS , QUADS_INITS , FILTERS_WINDOWS_INITS , n_threads( n_threads ) , KS_PQ_INITS , PQS_INITS
	{
	
	init_fft(l_fft); };

TimeQuad::TimeQuad( KS_INPUTS , AC_INPUTS , np_complex_d filters , double alpha , uint l_fft , int n_threads )
	: KS_INITS , AC_INITS , QUADS_INITS , FILTERS_NUMPY_INITS , n_threads( n_threads ) , KS_PQ_INITS , PQS_INITS
	{  
	
	init_fft(l_fft); };
	
TimeQuad::TimeQuad( KS_INPUTS , AC_INPUTS , np_complex_d filters , np_double windows , uint l_fft , int n_threads )
	: KS_INITS , AC_INITS , QUADS_INITS , FILTERS_WINDOWS_NUMPY_INITS , n_threads( n_threads ) , KS_PQ_INITS , PQS_INITS
	{ 
	
	init_fft(l_fft); };

// DESTRUCTOR
TimeQuad::~TimeQuad()
{	
	destroy_plans_kernels();
	/* 
	delete algorithm object
	*/
	
	delete algorithm;
}

// CHECKS 
void TimeQuad::checks()
{
	if (l_kernel %2 != 1)
	{
		throw std::runtime_error(" l_kernel is not odd dont expect this to work... ");
	}
	if ( not (n_kernels  >= 1))
	{
		throw std::runtime_error(" n_kernels = 0 dont expect this to work... ");
	}
	/*
		Maybe this next condition could be mouved to the algorithm implementation ?
	*/
	if (l_kernel > l_data)
	{
		throw std::runtime_error(" l_kernel > l_data dont expect this to work... ");
	}

}

void TimeQuad::checks_n_threads()
{	
	if ( n_threads <= 0 )
	{
		throw std::runtime_error(" n_threads <= 0 dont expect this to work... ");
	}
	else if ( n_threads > physical_n_threads() )
	{
		printf("Warning : The wanted number of thread (%d) is higher than the number of physical threads (%d) in this computer. n_thread was replaced by physical_n_threads. \n", n_threads, physical_n_threads() );
		n_threads = physical_n_threads();
	}
}

void TimeQuad::checks_filters()
{
    /*No filter -> only one kernel*/
	if ( ( filters.get() == NULL ) and ( n_kernels != 1 ))
	{
		throw std::runtime_error(" If no filter is given there can be only one set of kernels.");
	}
    /*filter and n_kernels ==1   -> only one filters*/
	else if ( ( filters.get() != NULL ) and ( n_kernels == 1 ) and ( filters.get_n_j() != n_kernels ) )
	{
		throw std::runtime_error(" If a list of filters is given and n_kernels = 1 then n_filters must be equal to 1.");
	}
    /*filter and n_kernels >1   -> same number of filters or one less*/
	else if ( ( filters.get() != NULL ) and ( n_kernels > 1 ) and ( filters.get_n_j() != n_kernels ) and ( filters.get_n_j() != n_kernels - 1 )  )
	{
		throw std::runtime_error(" If a list of filters and n_kernels > 1 then n_filters must be equal to n_kernels or n_kernels - 1.");
	}
	if( (filters.get() != NULL) and (filters.get_n_i() != l_kernel_half_c(l_kernel)) )
	{
		throw std::runtime_error(" length of filters not mathching with length of kernels ");
	}
}

void TimeQuad::checks_windows()
{
	if ( ( filters.get() == NULL ) and ( windows.get() == NULL ) and not( n_kernels == 1 ))
	{
		throw std::runtime_error(" If no window is given there can be only one set of kernels.");
	}
	else if( ( windows.get() != NULL ) and (windows.get_n_j() != n_kernels) )
	{
		throw std::runtime_error("number of windows != n_kernels");
	}
	else if( (windows.get() != NULL) and (windows.get_n_i() != l_kernel) )
	{
		throw std::runtime_error(" length of windows not mathching with length of kernels ");
	}
}

void TimeQuad::execution_checks( uint64_t l_data )
{
	if ( this->l_data != l_data )
	{
		throw std::runtime_error(" data length given during execution dont match data length declared at instentiation ");
	}
}

// PREPARE_PLANS METHOD
void TimeQuad::prepare_plans_kernels()
{   
	fftw_import_wisdom_from_filename("FFTW_Wisdom.dat");
	k_foward = fftw_plan_dft_r2c_1d( l_kernel , ks_p[0] , reinterpret_cast<fftw_complex*>(ks_p[0]) , FFTW_EXHAUSTIVE); // FFTW_ESTIMATE
	k_backward = fftw_plan_dft_c2r_1d( l_kernel, reinterpret_cast<fftw_complex*>(ks_p[0]) , ks_p[0] , FFTW_EXHAUSTIVE); 
	fftw_export_wisdom_to_filename("FFTW_Wisdom.dat");
}

// FRREE METHODS
void TimeQuad::destroy_plans_kernels()
{
	fftw_destroy_plan(k_foward); 
    fftw_destroy_plan(k_backward); 	
}

void TimeQuad::init_gen()
{
	checks();
	checks_n_threads();
	omp_set_num_threads(n_threads);
	checks_filters();
	checks_windows();
	
	prepare_plans_kernels();
	make_kernels();
}

void TimeQuad::init_direct()
{
	init_gen();
	/*
	Memory allocation for algorithm object
	*/
	
	TimeQuad_direct* tmp = new TimeQuad_direct( ks_p , ks_q , ps , qs , l_kernel , n_kernels , l_data , n_threads  );
	algorithm =  tmp ;
}

void TimeQuad::init_fft( uint l_fft )
{
	init_gen();
	/*
	Memory allocation for algorithm object
	*/
	
	TimeQuad_FFT* tmp = new TimeQuad_FFT( ks_p , ks_q , ps , qs , l_kernel , n_kernels , l_data , l_fft , n_threads );
	algorithm =  tmp ;
	
}

void TimeQuad::make_kernels()
{
	vanilla_kernels();
	normalize_for_ffts();
	copy_vanillas();
	apply_filters();
	apply_windows();	
}

void TimeQuad::vanilla_kernels()
{
	double t ; 	/* Abscisse positif */
	double prefact ;
	double argument ;
	double tmp1; 
	double tmp2;
	/*Could be parallelized*/
	
	for (uint i = 0 ; i < l_kernel/2; i++ ) // l_kernel doit être impaire
	{
		t = ( i + 1 )*dt;
		prefact = 2.0/sqrt(t);
		argument = sqrt( 2.0*t/dt );
		/* Right part */
		tmp1 = prefact * Fresnel_Cosine_Integral( argument ) ;
		tmp2 = prefact * Fresnel_Sine_Integral( argument ) ;
		
		ks_p( 0 , l_kernel/2 + 1 + i ) = tmp1;
		ks_q( 0 , l_kernel/2 + 1 + i ) = tmp2;
		/* Left part */
		ks_p( 0 , l_kernel/2 - 1 - i ) = tmp1;
		ks_q( 0 , l_kernel/2 - 1 - i ) = (-1) * tmp2;
	}
	/* In zero */
	ks_p( 0 , l_kernel/2 ) = 2.0*sqrt(2.0)/sqrt(dt);
	ks_q( 0 , l_kernel/2 ) = 0 ;
	
}	

void TimeQuad::normalize_for_ffts()
{	
	double fft_norm = 1.0/l_kernel ; // Normalization for FFT's

	for ( uint j = 0 ; j<l_kernel ; j++ )
	{
		ks_p(0,j) *= fft_norm ;
		ks_q(0,j) *= fft_norm ;
	}
}	

void TimeQuad::copy_vanillas()
{
	/*Could be parallelized*/
	for ( uint i = 1 ; i<n_kernels ; i++ ) 
	{
		for ( uint j = 0 ; j<l_kernel ; j++ )
		{
			ks_p(i,j) = ks_p(0,j) ;
			ks_q(i,j) = ks_q(0,j) ;
		}
	}
}

void TimeQuad::apply_filters()
{	
	uint l_f = l_kernel_half_c( l_kernel );
	double f[l_f];
	
	/* Foward transforms */
	for ( uint i = 0 ; i<n_kernels ; i++ )
	{
		fftw_execute_dft_r2c( k_foward, ks_p[i] , reinterpret_cast<fftw_complex*>(ks_p[i]) ); 
		fftw_execute_dft_r2c( k_foward, ks_q[i] , reinterpret_cast<fftw_complex*>(ks_q[i]) ); 
	}
	
	if (filters.get() == NULL) /* No filter given ==> Only one kernel */
	{	
		/* Default filter */
		for ( uint i = 0 ; i<l_f  ; i++ ){ f[i] = fft_freq( i , l_kernel , dt ); } ;
		Tukey_modifed_Window( ks_p_complex[0] , f , l_f , f_min_analogue , f_max_analogue , f_Nyquist ) ;
		Tukey_modifed_Window( ks_q_complex[0] , f , l_f , f_min_analogue , f_max_analogue , f_Nyquist ) ;
	}
    /*filter and n_kernels ==1   -> only one filters*/
	else if ( ( n_kernels == 1 ) and ( filters.get_n_j() == 1 ) )
	{
		for ( uint j = 0 ; j<l_f ; j++ )
		{
			ks_p_complex(0,j) *= filters(0,j) ;
			ks_q_complex(0,j) *= filters(0,j) ;
		}
	}
	else if ( filters.get_n_j() == n_kernels - 1 ) /*  There one filter less than n_kernels */
	{
		/* Default filter */
		for ( uint i = 0 ; i<l_f  ; i++ ){f[i] = fft_freq( i , l_kernel , dt );} ;
		Tukey_modifed_Window( ks_p_complex[0] , f , l_f , f_min_analogue , f_max_analogue , f_Nyquist ) ;
		Tukey_modifed_Window( ks_q_complex[0] , f , l_f , f_min_analogue , f_max_analogue , f_Nyquist ) ;
		
		/* Apply costum filters */
		for ( uint i = 1 ; i<n_kernels ; i++ ) 
		{
			for ( uint j = 0 ; j<l_f ; j++ )
			{
				ks_p_complex(i,j) *= filters(i-1,j) ;
				ks_q_complex(i,j) *= filters(i-1,j) ;
			}
		}
	}
	else /*  n_filters == n_kernels */
	{
		/* Apply costum filters */
		for ( uint i = 0 ; i<n_kernels ; i++ ) 
		{
			for ( uint j = 0 ; j<l_f ; j++ )
			{
				ks_p_complex(i,j) *= filters(i,j) ;
				ks_q_complex(i,j) *= filters(i,j) ;
			}
		}
	}
	
	/* Returning to real space */
	for ( uint i = 0 ; i<n_kernels ; i++ )
	{
		/* c2r destroys its inputs array */
		fftw_execute_dft_c2r(k_backward, reinterpret_cast<fftw_complex*>(ks_p[i]) , ks_p[i] ); 
		fftw_execute_dft_c2r(k_backward, reinterpret_cast<fftw_complex*>(ks_q[i]) , ks_q[i] ); 
	}
}

void TimeQuad::apply_windows()
{
	if (windows.get() != NULL)
	{
		/* Apply custom windows */
		for ( uint i = 0 ; i<n_kernels ; i++ ) 
		{
			for ( uint j = 0 ; j<l_kernel ; j++ )
			{
				ks_p(i,j) *= windows(i,j) ;
				ks_q(i,j) *= windows(i,j) ;
			}
		}
	}
	else
	{
		/* Default window */
		for ( uint i = 0 ; i<n_kernels ; i++ ) 
		{
			Tukey_Window( ks_p[i] , alpha , l_kernel ) ;
			Tukey_Window( ks_q[i] , alpha , l_kernel ) ;
		}
	}		
}

template<class DataType>
void TimeQuad::execute( DataType* data , uint64_t l_data )
{
	
	execution_checks( l_data );
	
	algorithm->execute( data );
	
}

template<class DataType>
void TimeQuad::execute_py( py::array_t<DataType> data )
{
	py::buffer_info buf = data.request(); 

    if (buf.ndim != 1 )
    {
		throw std::runtime_error("Number of dimensions must be one");
	}
	
	execute( (DataType*)buf.ptr , buf.size );
}

np_double TimeQuad::get_ps()
{
	/*
	Pybind11 doesn't work with uint64_t 
		This coul potentially cause problems with l_data, l_valid and l_full
	*/
	// Numpy will not copy the array when using the assignement operator=
	double* ptr = ps[0] + l_kernel -1 ;
	py::capsule free_dummy(	ptr, [](void *f){;} );
	
	return py::array_t<double, py::array::c_style>
	(
		{uint(n_kernels), uint(l_valid) },      // shape
		{ps.get_stride_j(), ps.get_stride_i() },   // C-style contiguous strides for double
		ptr  ,       // the data pointer
		free_dummy // numpy array references this parent
	);
}

np_double TimeQuad::get_qs()
{
	/*
	Pybind11 doesn't work with uint64_t 
		This coul potentially cause problems with l_data, l_valid and l_full
	*/
	// Numpy will not copy the array when using the assignement operator=
	double* ptr = qs[0] + l_kernel -1 ;
	py::capsule free_dummy(	ptr, [](void *f){;} );
	
	return py::array_t<double, py::array::c_style>
	(
		{uint(n_kernels), uint(l_valid)},      // shape
		{qs.get_stride_j(), qs.get_stride_i() },   // C-style contiguous strides for double
		ptr  ,       // the data pointer
		free_dummy // numpy array references this parent
	);
}

// MACROS UNDEF
#undef KS_INPUTS 
#undef AC_INPUTS 
#undef KS_INITS 
#undef AC_INITS 
#undef QUADS_INITS 
#undef FILTERS_WINDOWS_NULL_INITS 
#undef FILTERS_NUMPY_INITS 
#undef FILTERS_WINDOWS_NUMPY_INITS 
#undef FILTERS_WINDOWS_INITS 
#undef KS_PQ_INITS 
#undef PQS_INITS 