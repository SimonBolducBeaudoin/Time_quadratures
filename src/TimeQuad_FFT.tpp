//CONSTRUCTOR
template<class Quads_Index_Type,class DataType>
TimeQuad_FFT<Quads_Index_Type,DataType>::TimeQuad_FFT
(	
	np_double ks , 
	py::array_t<DataType, py::array::c_style> data , 
	double dt , 
	uint  l_fft , 
	int n_threads 
)
:
	n_prod		(compute_n_prod(ks))					,
    ks_shape    (get_shape(ks))                         ,
	l_kernel	(compute_l_kernels(ks))					,
    l_data      (compute_l_data(data))                  ,
    l_valid( compute_l_valid( l_kernel,l_data ) ),
    l_full( compute_l_full  ( l_kernel,l_data ) ) ,
    ks( copy_ks(ks,n_prod) ) ,
    quads		( Multi_array<double,2,Quads_Index_Type >( n_prod , l_full ,fftw_malloc,fftw_free))       ,
	dt			(dt) 									, 
	l_fft		(l_fft)									, 
	n_threads	(n_threads)								,
	l_chunk		(compute_l_chunk(l_kernel,l_fft)) 		,
    ks_complex	( Multi_array<complex_d,2>	(n_prod    ,(l_fft/2+1)	,fftw_malloc,fftw_free) ) 	,
	gs			( Multi_array<double,2>		(n_threads ,2*(l_fft/2+1),fftw_malloc,fftw_free) ) 	,
	fs			( Multi_array<complex_d,2>	(n_threads ,(l_fft/2+1)	,fftw_malloc,fftw_free) ) 	,
	hs          ( Multi_array<complex_d,3>	(n_prod    ,n_threads,(l_fft/2+1),fftw_malloc,fftw_free) )
{
    checks();
	prepare_plans();
}

// DESTRUCTOR
template<class Quads_Index_Type,class DataType>
TimeQuad_FFT<Quads_Index_Type,DataType>::~TimeQuad_FFT()
{	
    destroy_plans();
	// fftw_cleanup();
}

// CHECKS 
template<class Quads_Index_Type,class DataType>
void TimeQuad_FFT<Quads_Index_Type,DataType>::checks()
{
	if (2*l_kernel-2 > l_fft)
	{
		throw std::runtime_error("l_kernel to big, you have to repsect 2*l_kernel-2 <= l_fft");
	} 
}

// PREPARE_PLANS METHOD
template<class Quads_Index_Type,class DataType>
void TimeQuad_FFT<Quads_Index_Type,DataType>::prepare_plans()
{   
	fftw_import_wisdom_from_filename("FFTW_Wisdom.dat");
	kernel_plan = fftw_plan_dft_r2c_1d	( l_fft , (double*)ks_complex(0) 	, reinterpret_cast<fftw_complex*>(ks_complex(0)) 	, FFTW_EXHAUSTIVE);
	g_plan = fftw_plan_dft_r2c_1d		( l_fft , gs[0] 					, reinterpret_cast<fftw_complex*>( fs[0] ) 			, FFTW_EXHAUSTIVE);
	h_plan = fftw_plan_dft_c2r_1d		( l_fft , reinterpret_cast<fftw_complex*>(hs(0,0)) , (double*)hs(0,0) 				, FFTW_EXHAUSTIVE); /* The c2r transform destroys its input array */
	fftw_export_wisdom_to_filename("FFTW_Wisdom.dat"); 
}

template<class Quads_Index_Type,class DataType>
void TimeQuad_FFT<Quads_Index_Type,DataType>::destroy_plans()
{
	fftw_destroy_plan(kernel_plan); 
    fftw_destroy_plan(g_plan); 
    fftw_destroy_plan(h_plan); 
}

template<class Quads_Index_Type,class DataType>
uint TimeQuad_FFT<Quads_Index_Type,DataType>::compute_n_prod(np_double& np_array) 
{
    py::buffer_info buffer = np_array.request() ;
    std::vector<ssize_t> shape = buffer.shape;
    uint64_t product = 1;
    for (int i = 0; i < buffer.ndim - 1; i++) {
        product *= shape[i];
    }
    return product;
}

template<class Quads_Index_Type,class DataType>
Multi_array<double,2> TimeQuad_FFT<Quads_Index_Type,DataType>::copy_ks( np_double& np_ks, uint n_prod )
{
    /*Only works on contiguous arrays (i.e. no holes)*/
    py::buffer_info buffer = np_ks.request() ;
    std::vector<py::ssize_t> shape = buffer.shape ; // shape copy
    std::vector<py::ssize_t> strides = buffer.strides ; // stride copy
    size_t num_bytes = shape[0]*strides[0] ; // Memory space 
    double* new_ptr = (double*) malloc( num_bytes )  ; 
	memcpy ( (void*)new_ptr, (void*)buffer.ptr, num_bytes ) ; // copying memory
    py::ssize_t strides_m1 = strides.back(); // strides[-1]
    strides.pop_back(); 
    py::ssize_t strides_m2 = strides.back(); // strides[-2]
    Multi_array<double,2> ks( new_ptr, n_prod, shape.back()/*shape[-1]*/ , strides_m2 /*strides[-2]*/ , strides_m1/*strides[-1]*/ );
    return ks;
}

template<class Quads_Index_Type,class DataType>
uint TimeQuad_FFT<Quads_Index_Type,DataType>::compute_l_kernels(np_double& np_array) 
{
    py::buffer_info buffer = np_array.request() ;
    std::vector<ssize_t> shape = buffer.shape;
    return shape[buffer.ndim-1];
}

template<class Quads_Index_Type,class DataType>
std::vector<ssize_t> TimeQuad_FFT<Quads_Index_Type,DataType>::get_shape (np_double& np_array ) 
{
    return  np_array.request().shape;
}

template<class Quads_Index_Type,class DataType>
uint64_t TimeQuad_FFT<Quads_Index_Type,DataType>::compute_l_data(py::array_t<DataType, py::array::c_style>& data) {
    py::buffer_info buffer = data.request() ;
    auto shape = buffer.shape;
    return shape[buffer.ndim-1];
}

template<class Quads_Index_Type,class DataType>
void TimeQuad_FFT<Quads_Index_Type,DataType>::prepare_kernels(np_double&  np_ks)
{
    Multi_array<double,2> ks = copy_ks(np_ks,n_prod);
	double norm_factor = dt/l_fft; /*MOVED IN PREPARE_KERNELS*/
    for ( uint j = 0 ; j<n_prod ; j++ ) 
    {
        /* Value assignment and zero padding */
        for( uint i = 0 ; i < l_kernel ; i++)
        {
            ( (double*)ks_complex(j) )[i] = ks(j,i)*norm_factor ; /*Normalisation done here*/
        }
        for(uint i = l_kernel ; i < l_fft ; i++)
        {
            ( (double*)ks_complex(j) )[i] = 0 ; 
        }
        fftw_execute_dft_r2c(kernel_plan, (double*)ks_complex(j) , reinterpret_cast<fftw_complex*>(ks_complex(j)) ); 
    }
}

template<class Quads_Index_Type,class DataType>
void TimeQuad_FFT<Quads_Index_Type,DataType>::execution_checks(np_double& ks,py::array_t<DataType, py::array::c_style>& data  )
{
	if ( this->l_data != (uint64_t)data.request().shape[data.request().ndim-1] )
	{
		throw std::runtime_error("Error: Data length does not match the length provided at construction.");
	}
    if ( this->ks_shape != ks.request().shape )
	{
		throw std::runtime_error("Error: Shape of 'ks' does not match the shape provided at construction.");
	}
}

template<class Quads_Index_Type,class DataType>
void TimeQuad_FFT<Quads_Index_Type,DataType>::execute_py(np_double& ks, py::array_t<DataType, py::array::c_style>& np_data)
{
	execution_checks( ks,np_data );
    omp_set_num_threads(n_threads); // Makes sure the declared number of thread if the same as planned
    prepare_kernels(ks);
    Multi_array<DataType,1,uint64_t> data = Multi_array<DataType,1,uint64_t>::numpy_share(np_data) ;
	execute( data );
}

template<class Quads_Index_Type,class DataType>					
void TimeQuad_FFT<Quads_Index_Type,DataType>::execute( Multi_array<DataType,1,uint64_t>& data )
{	
	uint64_t l_data = 	data.get_n_i();
	uint n_chunks 	=	compute_n_chunks	(l_data,l_chunk);
	uint l_reste 	=	compute_l_reste		(l_data,l_chunk);

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
        
        #pragma omp for simd collapse(3)
        for ( uint j = 0 ; j<n_prod ; j++ ) 
        {  
            for( uint i=0; i < n_chunks-1 ; i++ )
            {	
                // Last l_kernel-1.0 points
                // Subject to race conditions
                for( uint k=l_chunk ; k < l_fft; k++ )
                {	
                    quads(j,i*l_chunk+k) = 0.0 ;
                }
            }	
        }
    }
	if (l_reste != 0)
	{			
		// Product 
        for ( uint j = 0 ; j<n_prod ; j++ ) 
        {
            // Select only the part of the ifft that contributes to the full output length
            for( uint k = 0 ; k < l_reste + l_kernel - 1 ; k++ )
            {
                quads(j,n_chunks*l_chunk+k) = 0.0 ;
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
			
			///// FOR EACH KERNELS 
            for ( uint j = 0 ; j<n_prod ; j++ ) 
            {
                // Product	
                for( uint k=0 ; k < (l_fft/2+1) ; k++ )
                {	
                    hs(j,this_thread,k) = ks_complex(j,k) * fs(this_thread,k);
                }  
                // ifft
                fftw_execute_dft_c2r(h_plan , reinterpret_cast<fftw_complex*>(hs(j,this_thread)) , (double*)hs(j,this_thread) );   
                
                // First l_kernel-1.0 points
                // Subject to race conditions
                for( uint k=0; k < l_kernel-1 ; k++ )
                {
                    #pragma omp atomic update
                    quads(j,i*l_chunk+k) += ( (double*)hs(j,this_thread))[k] ;
                }
                // Copy result to p and q 
                // Not subject to race conditions
                for( uint k=l_kernel-1; k < l_chunk ; k++ )
                {	
                    quads(j,i*l_chunk+k) = ( (double*)hs(j,this_thread))[k] ;
                }
                // Last l_kernel-1.0 points
                // Subject to race conditions
                for( uint k=l_chunk ; k < l_fft ; k++ )
                {
                    #pragma omp atomic update
                    quads(j,i*l_chunk+k) += ( (double*)hs(j,this_thread))[k] ;
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
        for ( uint j = 0 ; j<n_prod ; j++ ) 
        {
            complex_d tmp;
            for( uint k=0; k < (l_fft/2+1) ; k++)
            {
                tmp = fs(0,k) ;
                hs(j,0,k) = ks_complex(j,k) * tmp;
            }
            
            fftw_execute_dft_c2r(h_plan , reinterpret_cast<fftw_complex*>(hs(j,0)) , (double*)hs(j,0) );  
        
            // Select only the part of the ifft that contributes to the full output length
            for( uint k = 0 ; k < l_reste + l_kernel - 1 ; k++ )
            {
                quads(j,n_chunks*l_chunk+k) += ( (double*)hs(j,0))[k] ;
            }
        }
	}
	/////
}

template<class Quads_Index_Type,class DataType>
np_double TimeQuad_FFT<Quads_Index_Type,DataType>::get_quads()
{
	/*
	Pybind11 doesn't work with uint64_t 
		This coul potentially cause problems with l_data, l_valid and l_full
	*/
	// Numpy will not copy the array when using the assignement operator=
	double* ptr = quads[0] + l_kernel -1 ;
	py::capsule free_dummy(	ptr, [](void *f){;} );
	
    std::vector<ssize_t> shape_valid = ks_shape ;
    shape_valid.pop_back() ;
    shape_valid.push_back(uint(l_valid)) ;
    std::vector<ssize_t> shape_full = ks_shape ;
    shape_full.pop_back() ;
    shape_full.push_back(uint(l_full)) ;
    
    std::vector<ssize_t> strides ;    
    for (uint i = 0; i < shape_full.size(); ++i) 
    {
        ssize_t prod = 1;
        for (uint j = i + 1; j < shape_full.size(); ++j) 
        {
            prod *= shape_full[j];
        }
    strides.push_back(prod * sizeof(double));
    }
    
	return py::array_t<double, py::array::c_style>
	(
		shape_valid,      // shape
		strides,   // C-style contiguous strides for double
		ptr  ,       // the data pointer
		free_dummy // numpy array references this parent
	);
}