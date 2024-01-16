//CONSTRUCTOR
template<class BinType,class DataType>
TimeQuad_FFT_to_Hist<float,BinType,DataType>::TimeQuad_FFT_to_Hist
(	
	np_double ks , 
	py::array_t<DataType, py::array::c_style> data , 
	float dt , 
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
	dt			(dt) 									, 
	l_fft		(l_fft)									, 
    nofbins     (nofbins)                               ,
    max         (max)                                   ,
    bin_width   (2.0*max/( nofbins ))                   , 
	n_threads	(n_threads)								,
	l_chunk		(compute_l_chunk(l_kernel,l_fft)) 		,
    n_chunks 	(compute_n_chunks	(l_data,l_chunk))   ,
	l_reste 	(compute_l_reste(l_data,l_chunk))       ,
    l_qs        (compute_l_qs(l_kernel,n_chunks))       ,
    l_qs_chunk  (l_kernel - 1 )                          ,
    ks          ( copy_ks(ks,n_prod) )                  ,
    quads		( Multi_array<float,2>( n_prod , l_qs )) ,
    ks_complex	( Multi_array<complex_f,2,uint32_t>	(n_prod   ,(l_fft/2+1)	) ),
	gs			( Multi_array<float,2,uint32_t>     ((uint)n_threads,l_fft         ,fftwf_malloc,fftwf_free) ),
    fs			( Multi_array<complex_f,2,uint32_t>	((uint)n_threads,(l_fft/2+1)  ,fftwf_malloc,fftwf_free) ),
	hs          ( Multi_array<complex_f,3,uint32_t>	((uint)n_threads,n_prod    ,(l_fft/2+1),fftwf_malloc,fftwf_free) ),
    Hs          ( Multi_array<BinType,3,uint32_t> 	((uint)n_threads,n_prod    ,nofbins,fftwf_malloc,fftwf_free) )
{
    checks();
	prepare_plans();
    reset(); // initialize Hs memory to 0.
}

// DESTRUCTOR
template<class BinType,class DataType>
TimeQuad_FFT_to_Hist<float,BinType,DataType>::~TimeQuad_FFT_to_Hist()
{	
    destroy_plans();
}

// CHECKS 
template<class BinType,class DataType>
void TimeQuad_FFT_to_Hist<float,BinType,DataType>::checks()
{
	if (2*l_kernel-2 > l_fft)
	{
		throw std::runtime_error("l_kernel to big, you have to repsect 2*l_kernel-2 <= l_fft");
	} 
}

// PREPARE_PLANS METHOD
template<class BinType,class DataType>
void TimeQuad_FFT_to_Hist<float,BinType,DataType>::prepare_plans()
{   
	fftwf_import_wisdom_from_filename("FFTWF_Wisdom.dat");
	kernel_plan = fftwf_plan_dft_r2c_1d	( l_fft , (float*)ks_complex(0) 	, reinterpret_cast<fftwf_complex*>(ks_complex(0)) 	, FFTW_EXHAUSTIVE);
	g_plan = fftwf_plan_dft_r2c_1d		( l_fft , gs[0] 					, reinterpret_cast<fftwf_complex*>( fs[0] ) 			, FFTW_EXHAUSTIVE);
	int n[] = {(int)l_fft} ;
    h_plan = fftwf_plan_many_dft_c2r( 
        1, // rank == 1D transform
        n , //  list of dimensions 
        n_prod  , // howmany (to do many ffts on the same core)
        reinterpret_cast<fftwf_complex*>(hs(0,0)), // input
        NULL , // inembed
        1 , // istride
        l_fft/2 + 1 , // idist
        (float*)hs(0,0) ,  // output
        NULL , //  onembed
        1 , // ostride
        2*(l_fft/2+1) , // odist
        FFTW_EXHAUSTIVE
        ); 
	fftwf_export_wisdom_to_filename("FFTWF_Wisdom.dat"); 
}

template<class BinType,class DataType>
void TimeQuad_FFT_to_Hist<float,BinType,DataType>::destroy_plans()
{
	fftwf_destroy_plan(kernel_plan); 
    fftwf_destroy_plan(g_plan); 
    fftwf_destroy_plan(h_plan); 
}

template<class BinType,class DataType>
uint TimeQuad_FFT_to_Hist<float,BinType,DataType>::compute_n_prod(np_double& np_array) 
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
Multi_array<float,2,uint32_t> TimeQuad_FFT_to_Hist<float,BinType,DataType>::copy_ks( np_double& np_ks, uint n_prod )
{
    /*Only works on contiguous arrays (i.e. no holes)*/
    py::buffer_info buffer = np_ks.request() ;
    std::vector<py::ssize_t> shape = buffer.shape ; // shape copy
    std::vector<py::ssize_t> strides = buffer.strides ; // stride copy
    size_t num_bytes = shape[0]*strides[0]/2 ; // Number of bytes taken by the new array
    size_t len = shape[0]*strides[0]/8 ; // Number of continguous double in ks 
    float* new_ptr = (float*) malloc( num_bytes )  ; 
    for (uint i = 0; i < len; i++) {
        new_ptr[i] = (float)((double*)(buffer.ptr))[i];
    } 
    py::ssize_t strides_m1 = strides.back()/2; // strides[-1]
    strides.pop_back(); 
    py::ssize_t strides_m2 = strides.back()/2; // strides[-2]
    Multi_array<float,2,uint32_t> ks( new_ptr, n_prod, shape.back()/*shape[-1]*/ , strides_m2 /*strides[-2]*/ , strides_m1/*strides[-1]*/ );
    return ks;
}

template<class BinType,class DataType>
uint TimeQuad_FFT_to_Hist<float,BinType,DataType>::compute_l_kernels(np_double& np_array) 
{
    py::buffer_info buffer = np_array.request() ;
    std::vector<ssize_t> shape = buffer.shape;
    return shape[buffer.ndim-1];
}

template<class BinType,class DataType>
std::vector<ssize_t> TimeQuad_FFT_to_Hist<float,BinType,DataType>::get_shape (np_double& np_array ) 
{
    return  np_array.request().shape;
}

template<class BinType,class DataType>
uint64_t TimeQuad_FFT_to_Hist<float,BinType,DataType>::compute_l_data(py::array_t<DataType, py::array::c_style>& data) {
    py::buffer_info buffer = data.request() ;
    auto shape = buffer.shape;
    return shape[buffer.ndim-1];
}

template<class BinType,class DataType>
void TimeQuad_FFT_to_Hist<float,BinType,DataType>::prepare_kernels(np_double&  np_ks)
{
    Multi_array<float,2,uint32_t> ks = copy_ks(np_ks,n_prod);
	float norm_factor = dt/l_fft; /*MOVED IN PREPARE_KERNELS*/
    for ( uint j = 0 ; j<n_prod ; j++ ) 
    {
        /* Value assignment and zero padding */
        for( uint i = 0 ; i < l_kernel ; i++)
        {
            ( (float*)ks_complex(j) )[i] = ks(j,i)*norm_factor ; /*Normalisation done here*/
        }
        for(uint i = l_kernel ; i < 2*(l_fft/2+1) ; i++) /* Also 0s the extra element */
        {
            ( (float*)ks_complex(j) )[i] = 0 ; 
        }
        fftwf_execute_dft_r2c(kernel_plan, (float*)ks_complex(j) , reinterpret_cast<fftwf_complex*>(ks_complex(j)) ); 
    }
}

template<class BinType,class DataType>
void TimeQuad_FFT_to_Hist<float,BinType,DataType>::execution_checks(np_double& ks,py::array_t<DataType, py::array::c_style>& data  )
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
void TimeQuad_FFT_to_Hist<float,BinType,DataType>::execute_py(np_double& ks, py::array_t<DataType, py::array::c_style>& np_data)
{
	execution_checks( ks,np_data );
    omp_set_num_threads(n_threads); // Makes sure the declared number of thread if the same as planned
    prepare_kernels(ks);
    Multi_array<DataType,1,uint64_t> data = Multi_array<DataType,1,uint64_t>::numpy_share(np_data) ;
	execute( data );
}

 
#define ACCUMULATE(N_TH,J,DATA,L_DATA)\
{ \
	BinType* histogram_local = Hs(N_TH,J) ;\
    for (uint32_t m=0; m<L_DATA; m++) \
    { \
        float_to_hist( *(DATA+m) , histogram_local , max , bin_width );\
    } \
} 

template<class BinType,class DataType>					
void TimeQuad_FFT_to_Hist<float,BinType,DataType>::execute( Multi_array<DataType,1,uint64_t>& data )
{	        
    #pragma omp parallel
    {
        manage_thread_affinity();
        #pragma omp for simd collapse(2) nowait
        for ( uint j = 0 ; j<n_prod ; j++ ) 
        {  
            for( uint i=0; i < (n_chunks+1)*l_qs_chunk ; i++ )
            {	
                quads(j,i) = 0.0 ;
            }	
        }
        int this_thread = omp_get_thread_num();          
        for(uint k=l_chunk; k  < l_fft ; k++ )
        {
            gs(this_thread,k) = 0;
        }
        #pragma omp barrier
		#pragma omp for
		for( uint i=0; i < n_chunks ; i++ )
		{
			for(uint j=0 ; j < l_chunk ; j++ )
			{
				gs(this_thread,j) = (float)data[i*l_chunk + j] ; 
			}			
			fftwf_execute_dft_r2c( g_plan, gs[this_thread] , reinterpret_cast<fftwf_complex*>( fs[this_thread] ) );
            for ( uint j = 0 ; j<n_prod ; j++ ) 
            {
                for( uint k=0 ; k < (l_fft/2+1) ; k++ )
                {	
                    hs(this_thread,j,k) = ks_complex(j,k) * fs(this_thread,k);
                }  
            } 
            fftwf_execute_dft_c2r(h_plan , reinterpret_cast<fftwf_complex*>(hs(this_thread)) , (float*)hs(this_thread) );   
            
            for ( uint j = 0 ; j<n_prod ; j++ ) 
            {    
                ACCUMULATE(this_thread,j, ((float*)hs(this_thread,j)) + l_qs_chunk , l_fft - 2*l_qs_chunk)
            }
            for ( uint j = 0 ; j<n_prod ; j++ ) 
            {    
                for( uint k=0; k < l_qs_chunk; k++ )
                {
                    #pragma omp atomic update
                    quads(j,i*l_qs_chunk+k) += ( (float*)hs(this_thread,j))[k] ;
                    #pragma omp atomic update
                    quads(j,(i+1)*l_qs_chunk+k) += ( (float*)hs(this_thread,j))[l_chunk+k] ;
                }
            }
		}
        #pragma omp single
        {
            if (l_reste != 0)
            {	
                uint k=0 ;
                for(; k < l_reste ; k++ )
                {
                    gs(this_thread,k) = (float)data[n_chunks*l_chunk + k] ;
                }
                
                for(; k < l_fft ; k++ )
                {
                    gs(this_thread,k) = 0 ;
                }	
                fftwf_execute_dft_r2c(g_plan, gs[this_thread] , reinterpret_cast<fftwf_complex*>( fs[this_thread]) );
                for ( uint j = 0 ; j<n_prod ; j++ ) 
                {
                    for( uint k=0; k < (l_fft/2+1) ; k++)
                    {
                        hs(this_thread,j,k) = ks_complex(j,k) * fs(this_thread,k);
                    }
                }
                fftwf_execute_dft_c2r(h_plan , reinterpret_cast<fftwf_complex*>(hs(this_thread)) , (float*)hs(this_thread) );   
                for ( uint j = 0 ; j<n_prod ; j++ ) 
                {
                    for( uint k = 0 ; k < l_kernel - 1 ; k++ )
                    {
                        ( (float*)hs(this_thread,j))[k] += quads(j,(n_chunks)*l_qs_chunk+k);
                    }
                    ACCUMULATE(this_thread,j,(float*)hs(this_thread,j),l_reste)
                }
            }
        }
        #pragma omp for simd
        for ( uint j = 0 ; j<n_prod ; j++ ) 
        {  
            ACCUMULATE(this_thread,j,quads(j)+l_qs_chunk,(n_chunks-1)*l_qs_chunk)
        }
        // reduction
        if (this_thread != 0)
        {
            for (uint j=0; j<n_prod; j++)
            {
                for (uint i=0; i<nofbins; i++)
                {
                    #pragma omp atomic update
                    Hs(0,j,i) += Hs(this_thread,j,i) ;
                    Hs(this_thread,j,i) = 0 ;
                }
            }
        }
	}
}

template<class BinType,class DataType>
inline void TimeQuad_FFT_to_Hist<float,BinType,DataType>::float_to_hist( float data, BinType* histogram , double max , double bin_width )
{ 	
    std::abs(data) >= max ? histogram[0]++ : histogram[ (unsigned int)((data+max)/(bin_width)) ]++ ;
}

template<class BinType,class DataType>
void TimeQuad_FFT_to_Hist<float,BinType,DataType>::reset()
{
    for (uint n=0; n<(uint)n_threads; n++)
    {
        for (uint j=0; j<n_prod; j++)
        {
            for (uint i=0; i<nofbins; i++)
            {
                Hs(n,j,i) = 0 ;
            }
        }
    }
}

template<class BinType,class DataType>
py::array_t<BinType,py::array::c_style>  TimeQuad_FFT_to_Hist<float,BinType,DataType>::get_Histograms_py()
{
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
    
    BinType* ptr = Hs[0] ;
    size_t num_bytes = shape_Hs[0]*strides[0];
    BinType* new_array = (BinType*)malloc( num_bytes );
    memcpy ( (void*)new_array, (void*)ptr, num_bytes ) ;
    py::capsule capsule(new_array,free);
    
	return py::array_t<BinType, py::array::c_style>
	(
		shape_Hs,      // shape
		strides,   // C-style contiguous strides 
		new_array  ,       // the data pointer
		capsule // numpy array references this parent
	);
}

template<class BinType,class DataType>
py::array_t<double> TimeQuad_FFT_to_Hist<float,BinType,DataType>::abscisse_py( double max, uint nofbins )
{
	double bin_width = 2.0*max/( nofbins );
	Multi_array<double,1> abscisse(nofbins) ;	
    for(uint64_t i = 0; i < nofbins; i++)
    {
        abscisse[i] = ( (i + 0.5)*bin_width )- max ; 
    }
	return abscisse.move_py();
}