#include <TimeQuad_py.h>

// CLASS MACROS
#define PY_TIME_QUAD(DataType,QuadsIndexType)\
	py::class_<TimeQuad<QuadsIndexType>>( m , "TimeQuad_"#QuadsIndexType)\
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
	.def		("filters", 				&TimeQuad<QuadsIndexType>::get_filters 			)\
	.def		("half_norms", 				&TimeQuad<QuadsIndexType>::get_half_norms 		)\
	.def		("half_denormalization", 	&TimeQuad<QuadsIndexType>::half_denormalization	)\
	.def		("execute" , 				&TimeQuad<QuadsIndexType>::execute_py<DataType> )\
	.def_static("compute_flat_band",		&TimeQuad<QuadsIndexType>::compute_flat_band, 	 \
		"l_hc"_a , "dt"_a, "f_min_analogue"_a, "f_min_analogue"_a , "f_Nyquist"_a			)\
	\
	;
	
void init_TimeQuad(py::module &m)
{
	PY_TIME_QUAD(int16_t,uint) ;
	PY_TIME_QUAD(int16_t,uint64_t) ;
}

// CLOSE MACRO SCOPES
#undef PY_TIME_QUAD
