#pragma once
#include <omp_extra.h>
#include <sys/types.h>

#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;
using namespace pybind11::literals;

#include <Multi_array.h>

#include <fftw3.h>

#include <iostream>

typedef unsigned int uint;
typedef std::complex<float> complex_f;
typedef std::complex<double> complex_d;
typedef py::array_t<float, py::array::c_style> np_float;
typedef py::array_t<double, py::array::c_style> np_double;
typedef py::array_t<int16_t, py::array::c_style> np_int16;
typedef py::array_t<uint32_t, py::array::c_style> np_uint32_t;
typedef py::array_t<uint64_t, py::array::c_style> np_uint64_t;
typedef py::array_t<complex_f, py::array::c_style> np_complex_f;
typedef py::array_t<complex_d, py::array::c_style> np_complex_d;

template <class FloatType, class BinType, class DataType>
class TimeQuad_FFT_to_Hist2D {
  /*Parent class of all specializations*/
  // public :
  // py::array_t<BinType,py::array::c_style> get_Histograms_py();
  // private :
  // uint n_prod ;
  // std::vector<ssize_t> ks_shape;
  // uint nofbins ;
  // Multi_array<BinType,2,uint32_t> Hs ; // [n_prod][nofbins]
  // void reset();
};

template <class BinType, class DataType>
class TimeQuad_FFT_to_Hist2D<double, BinType, DataType> {
public:
  // Contructor
  TimeQuad_FFT_to_Hist2D(np_double ks,
                         py::array_t<DataType, py::array::c_style> data,
                         double dt, uint l_fft, uint nofbins, double max,
                         int n_threads);
  // Destructor
  ~TimeQuad_FFT_to_Hist2D();

  void execute(Multi_array<DataType, 1, uint64_t> &data);
  void execute_py(np_double &ks,
                  py::array_t<DataType, py::array::c_style> &data);

  static py::array_t<double> abscisse_py(double max, uint nofbins);

  // Returns only the valid part of the convolution
  py::array_t<BinType, py::array::c_style>
  get_Histograms_py(); // Data are copied
  void reset();

  // Utilities
  uint compute_n_prod(np_double &ks);
  std::vector<ssize_t> get_shape(np_double &ks);
  uint compute_l_kernels(np_double &ks);
  uint64_t compute_l_data(py::array_t<DataType, py::array::c_style> &data);
  uint64_t compute_l_valid(uint l_kernel, uint64_t l_data) {
    return l_data - l_kernel + 1;
  };
  uint64_t compute_l_full(uint l_kernel, uint64_t l_data) {
    return l_kernel + l_data - 1;
  };
  Multi_array<double, 2, uint32_t> copy_ks(np_double &np_ks, uint n_prod);

  uint compute_l_chunk(uint l_kernel, uint l_fft) {
    return l_fft - l_kernel + 1;
  };
  uint compute_n_chunks(uint64_t l_data, uint l_chunk) {
    return l_data / l_chunk;
  };
  uint compute_l_reste(uint64_t l_data, uint l_chunk) {
    return l_data % l_chunk;
  };
  uint compute_l_qs(uint l_kernel, uint n_chunks) {
    return (n_chunks + 1) * (l_kernel - 1);
  };

private:
  uint n_prod;
  std::vector<ssize_t> ks_shape;
  uint l_kernel;
  uint64_t l_data; //
  uint64_t l_valid;
  uint64_t l_full;
  double dt;
  uint l_fft; // Lenght(/number of elements) of the FFT used for FFT convolution
  uint nofbins;
  double max;
  double bin_width;
  int n_threads;
  uint l_chunk; // The length of a chunk
  uint n_chunks;
  uint l_reste;
  uint l_qs;
  uint l_qs_chunk;

  Multi_array<double, 2, uint32_t> ks;
  Multi_array<double, 2> quads;

  fftw_plan kernel_plan;
  fftw_plan g_plan;
  fftw_plan h_plan;

  // Pointers to all the complex kernels
  Multi_array<complex_d, 2, uint32_t> ks_complex; // [n_prod][frequency]
  Multi_array<double, 2, uint32_t>
      gs; // [thread_num][frequency] Catches data from data*
  Multi_array<complex_d, 2, uint32_t>
      fs; // [thread_num][frequency] Catches DFT of data
  Multi_array<complex_d, 3, uint32_t> hs; // [n_prod][thread_num][frequency]
  Multi_array<BinType, 3, uint32_t> Hs;   // [n_prod][nofbins]

  void checks();
  void prepare_plans();
  void destroy_plans();

  void prepare_kernels(np_double &ks);
  void execution_checks(np_double &ks,
                        py::array_t<DataType, py::array::c_style> &data);

  inline void float_to_hist(double data, BinType *histogram, double max,
                            double bin_width);
};

template <class BinType, class DataType>
class TimeQuad_FFT_to_Hist2D<float, BinType, DataType> {
public:
  // Contructor
  TimeQuad_FFT_to_Hist2D(np_double ks,
                         py::array_t<DataType, py::array::c_style> data,
                         float dt, uint l_fft, uint nofbins, double max,
                         int n_threads);
  // Destructor
  ~TimeQuad_FFT_to_Hist2D();

  void execute(Multi_array<DataType, 1, uint64_t> &data);
  void execute_py(np_double &ks,
                  py::array_t<DataType, py::array::c_style> &data);

  static py::array_t<double> abscisse_py(double max, uint nofbins);

  // Returns only the valid part of the convolution
  py::array_t<BinType, py::array::c_style>
  get_Histograms_py(); // Data are copied
  void reset();

  // Utilities
  uint compute_n_prod(np_double &ks);
  std::vector<ssize_t> get_shape(np_double &ks);
  uint compute_l_kernels(np_double &ks);
  uint64_t compute_l_data(py::array_t<DataType, py::array::c_style> &data);
  uint64_t compute_l_valid(uint l_kernel, uint64_t l_data) {
    return l_data - l_kernel + 1;
  };
  uint64_t compute_l_full(uint l_kernel, uint64_t l_data) {
    return l_kernel + l_data - 1;
  };
  Multi_array<float, 2, uint32_t> copy_ks(np_double &np_ks, uint n_prod);

  uint compute_l_chunk(uint l_kernel, uint l_fft) {
    return l_fft - l_kernel + 1;
  };
  uint compute_n_chunks(uint64_t l_data, uint l_chunk) {
    return l_data / l_chunk;
  };
  uint compute_l_reste(uint64_t l_data, uint l_chunk) {
    return l_data % l_chunk;
  };
  uint compute_l_qs(uint l_kernel, uint n_chunks) {
    return (n_chunks + 1) * (l_kernel - 1);
  };

private:
  uint n_prod;
  std::vector<ssize_t> ks_shape;
  uint l_kernel;
  uint64_t l_data; //
  uint64_t l_valid;
  uint64_t l_full;
  float dt;
  uint l_fft; // Lenght(/number of elements) of the FFT used for FFT convolution
  uint nofbins;
  double max;
  double bin_width;
  int n_threads;
  uint l_chunk; // The length of a chunk
  uint n_chunks;
  uint l_reste;
  uint l_qs;
  uint l_qs_chunk;

  Multi_array<float, 2, uint32_t> ks;
  Multi_array<float, 2> quads;

  fftwf_plan kernel_plan;
  fftwf_plan g_plan;
  fftwf_plan h_plan;

  // Pointers to all the complex kernels
  Multi_array<complex_f, 2, uint32_t> ks_complex; // [n_prod][frequency]
  Multi_array<float, 2, uint32_t>
      gs; // [thread_num][frequency] Catches data from data*
  Multi_array<complex_f, 2, uint32_t>
      fs; // [thread_num][frequency] Catches DFT of data
  Multi_array<complex_f, 3, uint32_t> hs; // [n_prod][thread_num][frequency]
  Multi_array<BinType, 3, uint32_t> Hs;   // [n_prod][nofbins]

  void checks();
  void prepare_plans();
  void destroy_plans();

  void prepare_kernels(np_double &ks);
  void execution_checks(np_double &ks,
                        py::array_t<DataType, py::array::c_style> &data);

  inline void float_to_hist(float data, BinType *histogram, double max,
                            double bin_width);
};

#include "TimeQuad_FFT_double_to_Hist.tpp"
#include "TimeQuad_FFT_float_to_Hist.tpp"