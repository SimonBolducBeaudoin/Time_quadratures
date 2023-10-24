//CONSTRUCTOR
template<class BinType,class DataType>
TimeQuad_FFT_to_Hist<double,BinType,DataType>::TimeQuad_FFT_to_Hist
(	
	np_double ks , 
	py::array_t<DataType, py::array::c_style> data , 
	double dt , 
	uint  l_fft ,
    uint nofbins ,
    double max,    
	int n_threads 
)
:
	n_prod		(compute_n_prod(ks))					,
    ks_shape    (get_shape(ks))                         ,
	l_kernel	(compute_l_kernels(ks))					,
    l_data      (compute_l_data(data))                  ,
    l_valid     ( compute_l_valid( l_kernel,l_data ) ),
    l_full      ( compute_l_full  ( l_kernel,l_data ) ) ,
    ks          ( copy_ks(ks,n_prod) ) ,
    qs		    ( Multi_array<double,2,uint32_t>( n_prod , l_kernel-1 )) ,
	dt			(dt) 									, 
	l_fft		(l_fft)									, 
    nofbins     (nofbins)                               ,
    max         (max)                                   ,
    bin_width   (2.0*max/( nofbins ))                   , 
	n_threads	(n_threads)								,
	l_chunk		(compute_l_chunk(l_kernel,l_fft)) 		,
    ks_complex	( Multi_array<complex_d,2,uint32_t>	(n_prod    ,(l_fft/2+1)	) ),
	g			( Multi_array<double,1,uint32_t>    (2*(l_fft/2+1),fftw_malloc,fftw_free) ),
    g_complex   ( Multi_array<complex_d,1,uint32_t>    ( (complex_d*)g.get_ptr(), l_fft/2+1 ) ),
	hs          ( Multi_array<complex_d,2,uint32_t>	(n_prod    ,(l_fft/2+1),fftw_malloc,fftw_free) ),
    Hs          ( Multi_array<BinType,2,uint32_t> 	(n_prod    ,nofbins,fftw_malloc,fftw_free) )
{
    checks();
	prepare_plans();
    reset(); // initialize Hs memory to 0.
}

// DESTRUCTOR
template<class BinType,class DataType>
TimeQuad_FFT_to_Hist<double,BinType,DataType>::~TimeQuad_FFT_to_Hist()
{	
    destroy_plans();
}

// CHECKS 
template<class BinType,class DataType>
void TimeQuad_FFT_to_Hist<double,BinType,DataType>::checks()
{
	if (2*l_kernel-2 > l_fft)
	{
		throw std::runtime_error("l_kernel to big, you have to repsect 2*l_kernel-2 <= l_fft");
	} 
}

// PREPARE_PLANS METHOD
template<class BinType,class DataType>
void TimeQuad_FFT_to_Hist<double,BinType,DataType>::prepare_plans()
{   
	fftw_import_wisdom_from_filename("FFTW_Wisdom.dat");
	kernel_plan = fftw_plan_dft_r2c_1d	( l_fft , (double*)ks_complex(0) 	, reinterpret_cast<fftw_complex*>(ks_complex(0)) 	, FFTW_EXHAUSTIVE);
	g_plan = fftw_plan_dft_r2c_1d		( l_fft , g.get_ptr()					, reinterpret_cast<fftw_complex*>( g.get_ptr() ) 			, FFTW_EXHAUSTIVE);
	h_plan = fftw_plan_dft_c2r_1d		( l_fft , reinterpret_cast<fftw_complex*>(hs(0)) , (double*)hs(0) 				, FFTW_EXHAUSTIVE); /* The c2r transform destroys its input array */
	fftw_export_wisdom_to_filename("FFTW_Wisdom.dat"); 
}

template<class BinType,class DataType>
void TimeQuad_FFT_to_Hist<double,BinType,DataType>::destroy_plans()
{
	fftw_destroy_plan(kernel_plan); 
    fftw_destroy_plan(g_plan); 
    fftw_destroy_plan(h_plan); 
}

template<class BinType,class DataType>
uint TimeQuad_FFT_to_Hist<double,BinType,DataType>::compute_n_prod(np_double& np_array) 
{
    py::buffer_info buffer = np_array.request() ;
    std::vector<ssize_t> shape = buffer.shape;
    uint64_t product = 1;
    for (int i = 0; i < buffer.ndim - 1; i++) {
        product *= shape[i];
    }
    return product;
}

template<class BinType,class DataType>
Multi_array<double,2,uint32_t> TimeQuad_FFT_to_Hist<double,BinType,DataType>::copy_ks( np_double& np_ks, uint n_prod )
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
    Multi_array<double,2,uint32_t> ks( new_ptr, n_prod, shape.back()/*shape[-1]*/ , strides_m2 /*strides[-2]*/ , strides_m1/*strides[-1]*/ );
    return ks;
}

template<class BinType,class DataType>
uint TimeQuad_FFT_to_Hist<double,BinType,DataType>::compute_l_kernels(np_double& np_array) 
{
    py::buffer_info buffer = np_array.request() ;
    std::vector<ssize_t> shape = buffer.shape;
    return shape[buffer.ndim-1];
}

template<class BinType,class DataType>
std::vector<ssize_t> TimeQuad_FFT_to_Hist<double,BinType,DataType>::get_shape (np_double& np_array ) 
{
    return  np_array.request().shape;
}

template<class BinType,class DataType>
uint64_t TimeQuad_FFT_to_Hist<double,BinType,DataType>::compute_l_data(py::array_t<DataType, py::array::c_style>& data) {
    py::buffer_info buffer = data.request() ;
    auto shape = buffer.shape;
    return shape[buffer.ndim-1];
}

template<class BinType,class DataType>
void TimeQuad_FFT_to_Hist<double,BinType,DataType>::prepare_kernels(np_double&  np_ks)
{
    Multi_array<double,2,uint32_t> ks = copy_ks(np_ks,n_prod);
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

template<class BinType,class DataType>
void TimeQuad_FFT_to_Hist<double,BinType,DataType>::execution_checks(np_double& ks,py::array_t<DataType, py::array::c_style>& data  )
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

template<class BinType,class DataType>
void TimeQuad_FFT_to_Hist<double,BinType,DataType>::execute_py(np_double& ks, py::array_t<DataType, py::array::c_style>& np_data)
{
	execution_checks( ks,np_data );
    omp_set_num_threads(n_threads); // Makes sure the declared number of thread if the same as planned
    prepare_kernels(ks);
    Multi_array<DataType,1,uint64_t> data = Multi_array<DataType,1,uint64_t>::numpy_share(np_data) ;
	execute( data );
}

 
#define ACCUMULATE(J,DATA,L_DATA)\
{ \
	BinType* histogram_local = Hs[J] ;\
    /*#pragma omp for reduction(+:histogram_local[:nofbins])*/\
    for (uint32_t i=0; i<L_DATA; i++) \
    { \
        float_to_hist( *((double*)(((char*)DATA)+(sizeof(double)*i))) , histogram_local , max , bin_width );\
    } \
} 

template<class BinType,class DataType>					
void TimeQuad_FFT_to_Hist<double,BinType,DataType>::execute( Multi_array<DataType,1,uint64_t>& data )
{	
	uint64_t l_data = 	data.get_n_i();
	uint n_chunks 	=	compute_n_chunks	(l_data,l_chunk);
	uint l_reste 	=	compute_l_reste		(l_data,l_chunk);

	///////////////////////
    // COMPUTE PS AND QS //
    ///////////////////////
    {
        // #pragma omp single
        {
            uint i=0 ;
            uint j=0 ;
            for(; j < l_chunk ; j++ )
            {
                g(j) = (double)data[i*l_chunk + j] ;
            }
            for(; j < l_fft ; j++ )
            {
                g(j) = 0.0 ; 
            }
            fftw_execute_dft_r2c( g_plan, g.get_ptr() , reinterpret_cast<fftw_complex*>(g.get_ptr()) );
        }    
        // #pragma omp for
        for ( uint j = 0 ; j<n_prod ; j++ ) 
        {
            // #pragma omp for
            for( uint k=0 ; k < (l_fft/2+1) ; k++ )
            {	
                hs(j,k) = ks_complex(j,k) * g_complex(k);
            } 
            fftw_execute_dft_c2r(h_plan , reinterpret_cast<fftw_complex*>(hs(j)) , (double*)hs(j));
            ACCUMULATE(j,((double*)hs[j])+l_kernel-1,l_chunk-l_kernel+1);
            // #pragma omp for
            for( uint k=0; k < l_kernel-1 ; k++ )
            {
                qs(j,k) = ( (double*)hs(j))[l_chunk+k] ;
            }
        }               
    }
    for( uint i=1; i < n_chunks-1 ; i++ )
    {
        // #pragma omp single
        {
            uint j=0 ;
            for(; j < l_chunk ; j++ )
            {
                g(j) = (double)data[i*l_chunk + j] ;
            }
            for(; j < l_fft ; j++ )
            {
                g(j) = 0.0 ; 
            }
            fftw_execute_dft_r2c( g_plan, g.get_ptr() , reinterpret_cast<fftw_complex*>(g.get_ptr()) );
        }
        #pragma omp parallel
        {	
            manage_thread_affinity();
            #pragma omp for
            for ( uint j = 0 ; j<n_prod ; j++ ) 
            {
                // #pragma omp for
                for( uint k=0 ; k < (l_fft/2+1) ; k++ )
                {	
                    hs(j,k) = ks_complex(j,k) * g_complex(k);
                } 
                fftw_execute_dft_c2r(h_plan , reinterpret_cast<fftw_complex*>(hs(j)) , (double*)hs(j));
                // #pragma omp for
                for( uint k=0; k < l_kernel-1 ; k++ )
                {
                    ( (double*)hs(j))[k] += qs(j,k) ;
                }
                ACCUMULATE(j, (double*)hs[j],l_chunk);
                // #pragma omp for
                for( uint k=0; k < l_kernel-1 ; k++ )
                {
                    qs(j,k) = ( (double*)hs(j))[l_chunk+k] ;
                }
            }
        }
    }  
    {
        // #pragma omp single
        {
            uint i=n_chunks-1 ;
            uint j=0 ;
            for(; j < l_chunk ; j++ )
            {
                g(j) = (double)data[i*l_chunk + j] ;
            }
            for(; j < l_fft ; j++ )
            {
                g(j) = 0.0 ; 
            }
            fftw_execute_dft_r2c( g_plan, g.get_ptr() , reinterpret_cast<fftw_complex*>(g.get_ptr()) );
        }
        // #pragma omp for
        for ( uint j = 0 ; j<n_prod ; j++ ) 
        {
            //#pragma omp for
            for( uint k=0 ; k < (l_fft/2+1) ; k++ )
            {	
                hs(j,k) = ks_complex(j,k) * g_complex(k);
            } 
            fftw_execute_dft_c2r(h_plan , reinterpret_cast<fftw_complex*>(hs(j)) , (double*)hs(j));
            //#pragma omp for
            for( uint k=0; k < l_kernel-1 ; k++ )
            {
                ( (double*)hs(j))[k] += qs(j,k) ;
            }
            ACCUMULATE(j, (double*)hs[j], l_chunk);
            //#pragma omp for // Pour le reste s'il y en a un 
            for( uint k=0; k < l_kernel-1 ; k++ )
            {
                qs(j,k) = ( (double*)hs(j))[l_chunk+k] ;
            }                
        }
    }
	///// The rest ---->
	if (l_reste != 0)
	{	
        uint k=0 ;
		for(; k < l_reste ; k++ )
		{
			g(k) = (double)data[n_chunks*l_chunk + k] ;
		}
		// make sure g only contains zeros
		for(; k < l_fft ; k++ )
		{
			g(k) = 0 ;
		}	
		fftw_execute_dft_r2c(g_plan, g.get_ptr() , reinterpret_cast<fftw_complex*>( g_complex.get_ptr() ) );
		// Product 
        for ( uint j = 0 ; j<n_prod ; j++ ) 
        {
            for( uint k=0; k < (l_fft/2+1) ; k++)
            {
                hs(j,k) = ks_complex(j,k) * g_complex(k);
            }   
            fftw_execute_dft_c2r(h_plan , reinterpret_cast<fftw_complex*>(hs(j)) , (double*)hs(j) );  
        
            // Select only the part of the ifft that contributes to the valid lenght
            for( uint k = 0 ; k < l_reste ; k++ )
            {
                ( (double*)hs(j))[k] += qs(j,k) ;
            }
            
            { 
            uint J = j; double* DATA = (double*)hs[j]; uint32_t L_DATA = l_reste ;
            BinType* histogram_local = Hs[J] ;
            for (uint32_t i=0; i<L_DATA; i++) 
            { 
                float_to_hist( *((double*)(((char*)DATA)+(sizeof(double)*i))) , histogram_local , max , bin_width );
            } 
        } 
        }
	}
	/////
}

template<class BinType,class DataType>
inline void TimeQuad_FFT_to_Hist<double,BinType,DataType>::float_to_hist( double data, BinType* histogram , double max , double bin_width )
{ 	
    std::abs(data) >= max ? histogram[0]++ : histogram[ (unsigned int)((data+max)/(bin_width)) ]++ ;
}

template<class BinType,class DataType>
void TimeQuad_FFT_to_Hist<double,BinType,DataType>::reset()
{
    for (uint j=0; j<n_prod; j++)
    {
        for (uint i=0; i<nofbins; i++)
        {
            Hs(j,i) = 0 ;
        }
    }
}

template<class BinType,class DataType>
py::array_t<BinType,py::array::c_style>  TimeQuad_FFT_to_Hist<double,BinType,DataType>::get_Histograms_py()
{
	BinType* ptr = Hs[0] ;
	py::capsule capsule_dummy(	ptr, [](void *f){;} );
	
    std::vector<ssize_t> shape_Hs = ks_shape ;
    shape_Hs.pop_back() ;
    shape_Hs.push_back(uint(nofbins)) ;
    
    std::vector<ssize_t> strides ;    
    for (uint i = 0; i < shape_Hs.size(); ++i) 
    {
        ssize_t prod = 1;
        for (uint j = i + 1; j < shape_Hs.size(); ++j) 
        {
            prod *= shape_Hs[j];
        }
    strides.push_back(prod * sizeof(BinType));
    }
    
	return py::array_t<BinType, py::array::c_style>
	(
		shape_Hs,      // shape
		strides,   // C-style contiguous strides for double
		ptr  ,       // the data pointer
		capsule_dummy // numpy array references this parent
	);
}