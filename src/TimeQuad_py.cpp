#include<string>
#include <TimeQuad_py.h>

const char* s = 
"Underscore in TimeQuad_IndexType behaves like a template argument \n"
"\t IndexType is the type of the de/referencing index for memory allocation \n"
"Constructor arguments \n"
"\t Z[Ohm] \n"
"\t\t Should be the gain of the sample if you're using deconvolution \n"
" dt[ns]\n (uint)l_data\n"
"\t kernel_conf \n"
"\t\t ==0 uses kq (only out of phase quadrature) \n"
"\t\t ==1 kp and kq (in phase and out of phase quadrature)\n"
"\t (numpy array) betas : .shape = (n_betas,l_kernels//2+1)\n"
"\t\t The shape of the modes\n"
"\t (numpy array) g[] : .shape = (l_kernels//2+1)\n" 
"\t\t The gain units of in bin per volt (i.e. not A/A neither V/V)\n"
"\t (float) alpha : parameter for the windowing \n"
"\t (uint)l_fft : lenght of fft to be used in the concnvolution algorithm \n"
"\t (uint)n_threads : number of computing threads to be used in computations \n" ;
// CLASS MACROS
#define PY_TIME_QUAD(DataType,QuadsIndexType)\
	py::class_<TimeQuad<QuadsIndexType>>( m , "TimeQuad_"#QuadsIndexType , s)\
	/*Constructor fft convolution */\
	.def\
	(\
		py::init\
		<\
			double,double,uint64_t,uint,\
			np_complex_d,np_complex_d,\
			double,uint,int\
		>(), \
		"Z"_a.noconvert() 				,\
		"dt"_a.noconvert() 				,\
		"l_data"_a.noconvert() 			,\
		"kernel_conf"_a.noconvert()	 	,\
		"betas"_a.noconvert() 			,\
		"g"_a.noconvert() 				,\
		"alpha"_a.noconvert()		 	,\
		"l_fft"_a.noconvert() 			,\
		"n_threads"_a.noconvert() 		\
	) \
	.def		("ks", 						&TimeQuad<QuadsIndexType>::get_ks				)\
	.def		("quads", 					&TimeQuad<QuadsIndexType>::get_quads			)\
	.def		("betas", 					&TimeQuad<QuadsIndexType>::get_betas 			)\
	.def		("g", 						&TimeQuad<QuadsIndexType>::get_g 				)\
	.def		("set_g", 				    &TimeQuad<QuadsIndexType>::set_g_py , "g"_a.noconvert() )\
	.def		("filters", 				&TimeQuad<QuadsIndexType>::get_filters 			)\
	.def		("half_norms", 				&TimeQuad<QuadsIndexType>::get_half_norms 		)\
	.def		("half_denormalization", 	&TimeQuad<QuadsIndexType>::half_denormalization	)\
	.def_static	("compute_flatband",		&TimeQuad<QuadsIndexType>::compute_flat_band, 	 \
		"l_hc"_a , "dt"_a, "f_min_analog_start"_a, "f_min_analog_stop"_a, "f_max_analog_start"_a, "f_max_analog_stop"_a			)\
    .def		("execute" , &TimeQuad<QuadsIndexType>::execute_py<DataType> )\
	\
	;
	
void init_TimeQuad(py::module &m)
{
	PY_TIME_QUAD(int16_t,uint) ;
	PY_TIME_QUAD(int16_t,uint64_t) ;
}

// CLOSE MACRO SCOPES
#undef PY_TIME_QUAD
