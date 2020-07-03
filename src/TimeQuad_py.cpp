#include <TimeQuad_py.h>

// CLASS MACROS
#define PY_TIME_QUAD(DataType,QuadsIndexType)\
	py::class_<TimeQuad<QuadsIndexType>>( m , "TimeQuad_"#QuadsIndexType)\
	/*Default constructor*/\
	.def\
	(\
		py::init<uint,uint,double,uint64_t,double,double,double,double,uint,int>(), \
		"l_kernel"_a.noconvert() = 257 , "n_kernels"_a.noconvert() = 1, \
		"Z"_a.noconvert() = 40.35143201333281, \
		"l_data"_a.noconvert() = (1<<16) , "dt"_a.noconvert() = 0.03125 ,\
		"f_max_analogue"_a.noconvert() = 10.0 , "f_min_analogue"_a.noconvert() = 0.5 ,\
		"alpha"_a.noconvert() = 0.5 ,\
		"l_fft"_a.noconvert() = (1<<10) , "n_threads"_a.noconvert() = 36 \
	)\
	/*Constructor with filters only */\
	.def\
	(\
		py::init\
		<\
			uint,uint,double,uint64_t,double,double,double, \
			py::array_t<complex_d,py::array::c_style>, \
			double,\
			uint,int\
		>(), \
		"l_kernel"_a.noconvert() = 257 , "n_kernels"_a.noconvert() = 1, \
		"Z"_a.noconvert() = 40.35143201333281, \
		"l_data"_a.noconvert() = (1<<16) , "dt"_a.noconvert() = 0.03125 ,\
		"f_max_analogue"_a.noconvert() = 10.0 , "f_min_analogue"_a.noconvert() = 0.5 ,\
		"filters"_a.noconvert() , "alpha"_a.noconvert() = 0.5 ,\
		"l_fft"_a.noconvert() = (1<<10) , "n_threads"_a.noconvert() = 36 \
	) \
	/*Constructor with filters and windows */\
	.def\
	(\
		py::init\
		<\
			uint,uint,double,uint64_t,double,double,double, \
			py::array_t<complex_d,py::array::c_style>, \
			py::array_t<double,py::array::c_style>,\
			uint,int\
		>(), \
		"l_kernel"_a.noconvert() = 257 , "n_kernels"_a.noconvert() = 1, \
		"Z"_a.noconvert() = 40.35143201333281, \
		"l_data"_a.noconvert() = (1<<16) , "dt"_a.noconvert() = 0.03125 ,\
		"f_max_analogue"_a.noconvert() = 10.0 , "f_min_analogue"_a.noconvert() = 0.5 ,\
		"filters"_a.noconvert() , "windows"_a.noconvert() , \
		"l_fft"_a.noconvert() = (1<<10) , "n_threads"_a.noconvert() = 36 \
	) \
	.def("ks_p", &TimeQuad<QuadsIndexType>::get_ks_p)\
	.def("ks_q", &TimeQuad<QuadsIndexType>::get_ks_q)\
	.def("ps", &TimeQuad<QuadsIndexType>::get_ps )\
	.def("qs", &TimeQuad<QuadsIndexType>::get_qs )\
	.def("half_norms_p", &TimeQuad<QuadsIndexType>::get_half_norms_p )\
	.def("half_norms_p", &TimeQuad<QuadsIndexType>::get_half_norms_p )\
	.def("half_denormalization", &TimeQuad<QuadsIndexType>::half_denormalization)\
	.def("execute" , &TimeQuad<QuadsIndexType>::execute_py<DataType> )\
	\
	;
	
void init_TimeQuad(py::module &m)
{
	PY_TIME_QUAD(int16_t,uint) ;
	PY_TIME_QUAD(int16_t,uint64_t) ;
}

// CLOSE MACRO SCOPES
#undef PY_TIME_QUAD
