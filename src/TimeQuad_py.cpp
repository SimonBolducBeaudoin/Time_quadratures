#include<string>
#include "TimeQuad_py.h"

const char* s = 
"\t TimeQuad_FFT implements an fft convolutions algorithm parallelized with openmp.\n"
"\t Memory allocation is done only once at construction.\n"
"\t Multiples kernels can be applied in parallel.\n"
"\t Class_IndexType, '_IndexType' behaves like Template Argument:\n"
"\t IndexType is the type used for indexing memory (i.e. arr[(IndexType)i])\n"
"Constructor arguments \n"
"\t ks (numpy array): Shape should be (..., l_kernels//2+1).\n"
"\t\t The kernels to be convoluted with data\n"
"\t data (numpy array) : .shape = (l_data,)\n"
"\t\t Same type and shape as the data array that will be called in the TimeQuad_FFT.execute(data) \n"
"\t dt[ns]\n l_data (uint) : The time delta between samples in seconds \n"
"\t l_fft (uint) : lenght of fft to be used in the concnvolution algorithm \n"
"\t n_threads (uint) : number of computing threads to be used in computations \n" ;
// CLASS MACROS
#define PY_TIME_QUAD_FFT(FloatType,DataType)\
	py::class_<TimeQuad_FFT<FloatType,DataType>>( m , "TimeQuad_FFT_"#FloatType"_"#DataType , s)\
	.def\
	(\
		py::init<np_double,np_int16,double,uint,int>(),\
		"ks"_a.noconvert() 				,\
		"data"_a.noconvert() 			,\
		"dt"_a.noconvert() 			,\
		"l_fft"_a.noconvert() 			,\
		"n_threads"_a.noconvert() 		\
	) \
	.def		("quads", 					&TimeQuad_FFT<FloatType,DataType>::get_quads			)\
    .def		("execute" , &TimeQuad_FFT<FloatType,DataType>::execute_py )\
	;
    
#define PY_TIME_QUAD_FFT_TO_HIST(FloatType,BinType,DataType)\
	py::class_<TimeQuad_FFT_to_Hist<FloatType,BinType,DataType>>( m , "TimeQuad_FFT_"#FloatType"_to_Hist_"#BinType"_"#DataType , s)\
	.def\
	(\
		py::init<np_double,np_int16,double,uint,uint,double,int>(),\
		"ks"_a.noconvert() 				,\
		"data"_a.noconvert() 			,\
		"dt"_a.noconvert() 			    ,\
		"l_fft"_a.noconvert() 			,\
        "nb_of_bins"_a.noconvert() 	    ,\
        "max"_a.noconvert() 			,\
		"n_threads"_a.noconvert() 		\
	) \
	.def		("Histograms", 					&TimeQuad_FFT_to_Hist<FloatType,BinType,DataType>::get_Histograms_py			)\
    .def("reset", &TimeQuad_FFT_to_Hist<FloatType,BinType,DataType>::reset)\
    .def		("execute" , &TimeQuad_FFT_to_Hist<FloatType,BinType,DataType>::execute_py )\
	;
	
void init_TimeQuad_FFT(py::module &m)
{
	PY_TIME_QUAD_FFT(double,int16_t) ;
    PY_TIME_QUAD_FFT(float,int16_t) ;
    PY_TIME_QUAD_FFT_TO_HIST(double,uint32_t,int16_t)
}

// CLOSE MACRO SCOPES
#undef PY_TIME_QUAD_FFT
#undef PY_TIME_QUAD_FFT_TO_HIST
