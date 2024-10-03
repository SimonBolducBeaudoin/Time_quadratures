#include "TimeQuad_py.h"
#include <string>

const char *s_TQ =
    "\t TimeQuad_FFT implements an fft convolutions algorithm parallelized "
    "with openmp.\n"
    "\t Multiples kernels can be applied in parallel.\n"
    "\t Class_FloatType_DataType, where '_XType' behaves like Template "
    "Argument:\n"
    "\t\t FloatType is the typed used for FFTW (float or double)\n"
    "\t\t DataType  is the typed used for the input data (only int16 for now)\n"
    "\t ks (numpy array): kernels with shape (..., l_kernel).\n"
    "\t\t Similarly to generalized universal functions the algoritm is applied "
    "along the last axis of ks and repeated for all other dimensions.\n"
    "\t data (numpy array) : .shape = (l_data,) and dtype = int16 \n"
    "\t\t data's shape must de the same in the constructor and the "
    "TimeQuad_FFT.execute(data) method. \n"
    "\t dt[ns]\n l_data (uint) : The time delta between samples in seconds \n"
    "\t l_fft (uint) : lenght of fft to be used in the concnvolution algorithm "
    "\n"
    "\t n_threads (uint) : number of openmp threads to be used \n"
    "\t\t Memory is allocated once at construction and re-used, therefore the "
    "TimeQuad_FFT.execute(data) method, sets the number of threads at each "
    "call to ensure compatibility with memory layout. \n";

const char *s_TQtoH =
    "\t TimeQuad_FFTtoHist implements an fft convolutions followed by a "
    "deduction of the data into 1D-histograms, all parallelizedwith openmp\n"
    "\t This uses a lot less memory compared to TimeQuad_FFT. Only the first "
    "l_kernel-1 and last l_kernel-1 for each fft_chunk is stored in ram. The "
    "saving in memory cost should roughly be l_kernel/l_fft. \n"
    "\t Multiples kernels can be applied in parallel.\n"
    "\t Class_FloatType_BinType_DataType, where '_XType' behaves like Template "
    "Argument:\n"
    "\t\t FloatType is the typed used for FFTW (float or double)\n"
    "\t\t BinType   is the typed used for all histograms bins (uint32_t or "
    "uint64_t) \n"
    "\t\t DataType  is the typed used for the input data (only int16_t for "
    "now)\n"
    "\t ks (numpy array): kernels with shape (..., l_kernel).\n"
    "\t\t Similarly to generalized universal functions the algoritm is applied "
    "along the last axis of ks and repeated for all other dimensions.\n"
    "\t data (numpy array) : .shape = (l_data,) and dtype = int16 \n"
    "\t\t data's shape must de the same in the constructor and the "
    "TimeQuad_FFT.execute(data) method. \n"
    "\t dt[ns]\n l_data (uint) : The time delta between samples in seconds \n"
    "\t l_fft (uint) : lenght of fft to be used in the concnvolution algorithm "
    "\n"
    "\t n_threads (uint) : number of openmp threads to be used \n"
    "\t\t Memory is allocated once at construction and re-used, therefore the "
    "TimeQuad_FFT.execute(data) method, sets the number of threads at each "
    "call to ensure compatibility with memory layout. \n";

// CLASS MACROS
#define PY_TIME_QUAD_FFT(FloatType, DataType)                                  \
  py::class_<TimeQuad_FFT<FloatType, DataType>>(                               \
      m, "TimeQuad_FFT_" #FloatType "_" #DataType, s_TQ)                       \
      .def(py::init<np_double, np_int16, double, uint, int>(),                 \
           "ks"_a.noconvert(), "data"_a.noconvert(), "dt"_a.noconvert(),       \
           "l_fft"_a.noconvert(), "n_threads"_a.noconvert())                   \
      .def("quads", &TimeQuad_FFT<FloatType, DataType>::get_quads)             \
      .def("execute", &TimeQuad_FFT<FloatType, DataType>::execute_py);

#define PY_TIME_QUAD_FFT_TO_HIST(FloatType, BinType, DataType)                 \
  py::class_<TimeQuad_FFT_to_Hist<FloatType, BinType, DataType>>(              \
      m, "TimeQuad_FFT_" #FloatType "_to_Hist_" #BinType "_" #DataType,        \
      s_TQtoH)                                                                 \
      .def(py::init<np_double, np_int16, double, uint, uint, double, int>(),   \
           "ks"_a.noconvert(), "data"_a.noconvert(), "dt"_a.noconvert(),       \
           "l_fft"_a.noconvert(), "nb_of_bins"_a.noconvert(),                  \
           "max"_a.noconvert(), "n_threads"_a.noconvert())                     \
      .def("Histograms", &TimeQuad_FFT_to_Hist<FloatType, BinType,             \
                                               DataType>::get_Histograms_py)   \
      .def("reset",                                                            \
           &TimeQuad_FFT_to_Hist<FloatType, BinType, DataType>::reset)         \
      .def("execute",                                                          \
           &TimeQuad_FFT_to_Hist<FloatType, BinType, DataType>::execute_py)    \
      .def_static(                                                             \
          "abscisse",                                                          \
          &TimeQuad_FFT_to_Hist<FloatType, BinType, DataType>::abscisse_py,    \
          "max"_a.noconvert(), "nofbins"_a.noconvert());

void init_TimeQuad_FFT(py::module &m) {
  PY_TIME_QUAD_FFT(double, int16_t);
  PY_TIME_QUAD_FFT(float, int16_t);
  PY_TIME_QUAD_FFT_TO_HIST(double, uint64_t, int16_t);
  PY_TIME_QUAD_FFT_TO_HIST(double, uint32_t, int16_t);
  PY_TIME_QUAD_FFT_TO_HIST(float, uint64_t, int16_t);
  PY_TIME_QUAD_FFT_TO_HIST(float, uint32_t, int16_t);
}

// CLOSE MACRO SCOPES
#undef PY_TIME_QUAD_FFT
#undef PY_TIME_QUAD_FFT_TO_HIST
