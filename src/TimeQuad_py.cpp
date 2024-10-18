#include "TimeQuad_py.h"
#include <string>

const char *s_TQ =
    "\t TimeQuad_FFT implements an FFT convolution algorithm parallelized "
    "with OpenMP.\n"
    "\t Multiple kernels can be applied in parallel.\n"
    "\t Class_FloatType_DataType, where '_XType' behaves like Template "
    "Arguments:\n"
    "\t\t FloatType is the type used for FFTW (float or double).\n"
    "\t\t DataType is the type used for the input data (currently only "
    "int16).\n"
    "\t ks (numpy array): kernels with shape (..., l_kernel).\n"
    "\t\t Similar to generalized universal functions, the algorithm is applied "
    "along the last axis of ks and repeated for all other dimensions.\n"
    "\t data (numpy array): shape = (l_data,), dtype = int16.\n"
    "\t\t The shape of data must be the same in both the constructor and the "
    "TimeQuad_FFT.execute(data) method.\n"
    "\t dt (ns): Time delta between samples in seconds.\n"
    "\t l_fft (uint): Length of the FFT used in the convolution algorithm.\n"
    "\t n_threads (uint): Number of OpenMP threads to be used.\n"
    "\t\t Memory is allocated once during construction and re-used. Therefore, "
    "the "
    "number of threads is set during each call to TimeQuad_FFT.execute(data) "
    "to "
    "ensure compatibility with memory layout.\n"
    "\t Methods:\n"
    "\t\t .execute(ks, data): Takes in kernels (ks[..., l_kernel]) and data as "
    "numpy "
    "array (e.g., int16), performs the convolution, and stores the result in "
    "the "
    "'quad' attribute.\n"
    "\t\t .quads(): Returns access to the result of the convolution.\n"
    "\t\t .reset(): Resets the state of the object to allow reuse without "
    "reinstantiating.\n"
    "\t\t .abscisse(): Static method to obtain the abscissa (X-axis) array "
    "defining "
    "the histogram.";

const char *s_TQtoH =
    "\t TimeQuad_FFTtoHist implements an FFT convolution followed by a "
    "reduction of the data into 1D histograms, parallelized with OpenMP.\n"
    "\t This method uses significantly less memory compared to TimeQuad_FFT. "
    "Only the first l_kernel-1 and last l_kernel-1 of each FFT chunk are "
    "stored "
    "in RAM. The memory saving is approximately l_kernel/l_fft.\n"
    "\t Multiple kernels can be applied in parallel.\n"
    "\t Class_FloatType_BinType_DataType, where '_XType' behaves like Template "
    "Arguments:\n"
    "\t\t FloatType is the type used for FFTW (float or double).\n"
    "\t\t BinType is the type used for histogram bins (uint32_t or uint64_t).\n"
    "\t\t DataType is the type used for input data (currently only int16).\n"
    "\t ks (numpy array): kernels with shape (..., l_kernel).\n"
    "\t\t Similar to generalized universal functions, the algorithm is applied "
    "along the last axis of ks and repeated for all other dimensions.\n"
    "\t data (numpy array): shape = (l_data,), dtype = int16.\n"
    "\t\t The shape of data must be the same in both the constructor and the "
    "TimeQuad_FFT.execute(data) method.\n"
    "\t dt (ns): Time delta between samples in seconds.\n"
    "\t l_fft (uint): Length of the FFT used in the convolution algorithm.\n"
    "\t n_threads (uint): Number of OpenMP threads to be used.\n"
    "\t\t Memory is allocated once during construction and re-used. Therefore, "
    "the "
    "number of threads is set during each call to TimeQuad_FFT.execute(data) "
    "to "
    "ensure compatibility with memory layout.\n"
    "\t Methods:\n"
    "\t\t .execute(ks, data): Takes in kernels (ks[..., l_kernel]) and data as "
    "numpy "
    "array (e.g., int16), performs the convolution, and bins the result into a "
    "1D histogram.\n"
    "\t\t .Histograms(): Returns a copy of the accumulated 1D histogram.\n"
    "\t\t .reset(): Resets the state of the object to allow reuse without "
    "reinstantiating.\n"
    "\t\t .abscisse(): Static method to obtain the abscissa (X-axis) array "
    "defining the histogram.";

const char *s_TQtoH2D =
    "\t TimeQuad_FFTtoHist2D implements an FFT convolution followed by a "
    "reduction of the data into 2D histograms, parallelized with OpenMP.\n"
    "\t This method is optimized to use less memory, as only portions of each "
    "FFT chunk are stored in RAM. This results in significant memory savings "
    "compared to full FFT approaches.\n"
    "\t Multiple kernels can be applied in parallel.\n"
    "\t The second index (j in ks[...,j,i]) must be even.\n"
    "\t Hist2D will use (j,j+1) as the X-axis and Y-axis for the histogram.\n"
    "\t The internal 2D histogram returned by TimeQuad_FFTtoHist2D.Histograms "
    "has shape = (..., j//2).\n"
    "\t Class_FloatType_BinType_DataType, where '_XType' behaves like Template "
    "Arguments:\n"
    "\t\t FloatType is the type used for FFTW (float or double).\n"
    "\t\t BinType is the type used for histogram bins (uint32_t or uint64_t).\n"
    "\t\t DataType is the type used for input data (currently only int16).\n"
    "\t ks (numpy array): kernels with shape (..., l_kernel).\n"
    "\t\t Similar to generalized universal functions, the algorithm is applied "
    "along the last axis of ks and repeated for all other dimensions.\n"
    "\t data (numpy array): shape = (l_data,), dtype = int16.\n"
    "\t\t The shape of data must be the same in both the constructor and the "
    "TimeQuad_FFTtoHist2D.execute(data) method.\n"
    "\t dt (ns): Time delta between samples in seconds.\n"
    "\t l_fft (uint): Length of the FFT used in the convolution algorithm.\n"
    "\t n_threads (uint): Number of OpenMP threads to be used.\n"
    "\t\t Memory is allocated once during construction and re-used. Therefore, "
    "the "
    "number of threads is set during each call to "
    "TimeQuad_FFTtoHist2D.execute(data) "
    "to ensure compatibility with memory layout.\n"
    "\t Methods:\n"
    "\t\t .execute(ks, data): Takes in kernels (ks[..., l_kernel]) and data as "
    "numpy "
    "array (e.g., int16), performs the convolution, bins the result into a "
    "2D histogram, and discards the convolution result.\n"
    "\t\t .Histograms(): Returns a copy of the accumulated 2D histogram.\n"
    "\t\t .reset(): Resets the state of the object to allow reuse without "
    "reinstantiating.\n"
    "\t\t .abscisse(): Static method to obtain the abscissa (X and Y-axis) "
    "arrays "
    "defining the 2D histogram (square by default).";

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

#define PY_TIME_QUAD_FFT_TO_HIST2D(FloatType, BinType, DataType)               \
  py::class_<TimeQuad_FFT_to_Hist2D<FloatType, BinType, DataType>>(            \
      m, "TimeQuad_FFT_" #FloatType "_to_Hist2D_" #BinType "_" #DataType,      \
      s_TQtoH2D)                                                               \
      .def(py::init<np_double, np_int16, double, uint, uint, double, int>(),   \
           "ks"_a.noconvert(), "data"_a.noconvert(), "dt"_a.noconvert(),       \
           "l_fft"_a.noconvert(), "nb_of_bins"_a.noconvert(),                  \
           "max"_a.noconvert(), "n_threads"_a.noconvert())                     \
      .def("Histograms", &TimeQuad_FFT_to_Hist2D<FloatType, BinType,           \
                                                 DataType>::get_Histograms_py) \
      .def("reset",                                                            \
           &TimeQuad_FFT_to_Hist2D<FloatType, BinType, DataType>::reset)       \
      .def("execute",                                                          \
           &TimeQuad_FFT_to_Hist2D<FloatType, BinType, DataType>::execute_py)  \
      .def_static(                                                             \
          "abscisse",                                                          \
          &TimeQuad_FFT_to_Hist2D<FloatType, BinType, DataType>::abscisse_py,  \
          "max"_a.noconvert(), "nofbins"_a.noconvert());

void init_TimeQuad_FFT(py::module &m) {
  PY_TIME_QUAD_FFT(double, int16_t);
  PY_TIME_QUAD_FFT(float, int16_t);

  PY_TIME_QUAD_FFT_TO_HIST(double, uint64_t, int16_t);
  PY_TIME_QUAD_FFT_TO_HIST(double, uint32_t, int16_t);
  PY_TIME_QUAD_FFT_TO_HIST(float, uint64_t, int16_t);
  PY_TIME_QUAD_FFT_TO_HIST(float, uint32_t, int16_t);

  PY_TIME_QUAD_FFT_TO_HIST2D(double, uint64_t, int16_t);
  PY_TIME_QUAD_FFT_TO_HIST2D(double, uint32_t, int16_t);
  PY_TIME_QUAD_FFT_TO_HIST2D(float, uint64_t, int16_t);
  PY_TIME_QUAD_FFT_TO_HIST2D(float, uint32_t, int16_t);
}

// CLOSE MACRO SCOPES
#undef PY_TIME_QUAD_FFT
#undef PY_TIME_QUAD_FFT_TO_HIST
