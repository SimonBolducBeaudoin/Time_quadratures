#include<string>
#include "TimeQuad_py.h"

const char* s = 
"\t TimeQuad_FFT implements an fft convolutions algorithm parallelized with openmp."
"\t Memory allocation is done only once at construction."
"\t Multiples kernels can be applied in parallel."
"\t Class_IndexType, '_IndexType' behaves like Template Argument:\n"
"\t IndexType is the type used for indexing memory (i.e. arr[(IndexType)i])\n"
"Constructor arguments \n"
"\t ks (numpy array): Shape should be (..., l_kernels//2+1).\n"
"\t\t The kernels to be convoluted with data"
"\t data (numpy array) : .shape = (l_data,)\n"
"\t\t Same type and shape as the data array that will be called in the TimeQuad_FFT.execute(data) "
"\t dt[ns]\n l_data (uint) : The time delta between samples in seconds "
"\t l_fft (uint) : lenght of fft to be used in the concnvolution algorithm \n"
"\t n_threads (uint) : number of computing threads to be used in computations \n" ;
// CLASS MACROS
#define PY_TIME_QUAD_FFT(DataType,QuadsIndexType)\
	py::class_<TimeQuad_FFT<QuadsIndexType,DataType>>( m , "TimeQuad_FFT_"#QuadsIndexType"_"#DataType , s)\
	/*Constructor fft convolution */\
	.def\
	(\
		py::init<np_double,np_int16,double,uint,int>(),\
		"ks"_a.noconvert() 				,\
		"data"_a.noconvert() 			,\
		"dt"_a.noconvert() 			,\
		"l_fft"_a.noconvert() 			,\
		"n_threads"_a.noconvert() 		\
	) \
	.def		("quads", 					&TimeQuad_FFT<QuadsIndexType,DataType>::get_quads			)\
    .def		("execute" , &TimeQuad_FFT<QuadsIndexType,DataType>::execute_py )\
	;
	
void init_TimeQuad_FFT(py::module &m)
{
	PY_TIME_QUAD_FFT(int16_t,uint) ;
	PY_TIME_QUAD_FFT(int16_t,uint64_t) ;
}

// CLOSE MACRO SCOPES
#undef PY_TIME_QUAD_FFT
