// CONSTRUCTOR
template <class DataType>
TimeQuad_FFT<float, DataType>::TimeQuad_FFT(np_double ks,
                                            py::array_t<DataType, py::array::c_style> data,
                                            double dt, uint l_fft, int n_threads)
    : n_prod(compute_n_prod(ks)), ks_shape(get_shape(ks)), l_kernel(compute_l_kernels(ks)),
      l_data(compute_l_data(data)), l_valid(compute_l_valid(l_kernel, l_data)),
      l_full(compute_l_full(l_kernel, l_data)), ks(copy_ks(ks, n_prod)),
      quads(Multi_array<float, 2>(n_prod, l_full, fftwf_malloc, fftwf_free)), dt(dt), l_fft(l_fft),
      n_threads(n_threads), l_chunk(compute_l_chunk(l_kernel, l_fft)),
      ks_complex(
          Multi_array<complex_f, 2, uint32_t>(n_prod, (l_fft / 2 + 1), fftwf_malloc, fftwf_free)),
      gs(Multi_array<float, 2, uint32_t>((uint)n_threads, l_fft, fftwf_malloc, fftwf_free)),
      fs(Multi_array<complex_f, 2, uint32_t>((uint)n_threads, (l_fft / 2 + 1), fftwf_malloc,
                                             fftwf_free)),
      hs(Multi_array<complex_f, 3, uint32_t>((uint)n_threads, n_prod, (l_fft / 2 + 1), fftwf_malloc,
                                             fftwf_free)) {
    checks();
    prepare_plans();
}

// DESTRUCTOR
template <class DataType> TimeQuad_FFT<float, DataType>::~TimeQuad_FFT() {
    destroy_plans();
    // fftwf_cleanup();
}

// CHECKS
template <class DataType> void TimeQuad_FFT<float, DataType>::checks() {
    if (2 * l_kernel - 2 > l_fft) {
        throw std::runtime_error("l_kernel to big, you have to repsect 2*l_kernel-2 <= l_fft");
    }
}

// PREPARE_PLANS METHOD
template <class DataType> void TimeQuad_FFT<float, DataType>::prepare_plans() {
    fftwf_import_wisdom_from_filename("FFTWF_Wisdom.dat");
    kernel_plan =
        fftwf_plan_dft_r2c_1d(l_fft, (float *)ks_complex(0),
                              reinterpret_cast<fftwf_complex *>(ks_complex(0)), FFTW_EXHAUSTIVE);
    g_plan = fftwf_plan_dft_r2c_1d(l_fft, gs[0], reinterpret_cast<fftwf_complex *>(fs[0]),
                                   FFTW_EXHAUSTIVE);
    // h_plan = fftwf_plan_dft_c2r_1d		( l_fft ,
    // reinterpret_cast<fftwf_complex*>(hs(0)) , (float*)hs(0) 				,
    // FFTW_EXHAUSTIVE); /* The c2r transform destroys its input array */
    int n[] = {(int)l_fft};
    h_plan = fftwf_plan_many_dft_c2r(1,      // rank == 1D transform
                                     n,      //  list of dimensions
                                     n_prod, // howmany (to do many ffts on the same core)
                                     reinterpret_cast<fftwf_complex *>(hs(0, 0)), // input
                                     NULL,                                        // inembed
                                     1,                                           // istride
                                     l_fft / 2 + 1,                               // idist
                                     (float *)hs(0, 0),                           // output
                                     NULL,                                        //  onembed
                                     1,                                           // ostride
                                     2 * (l_fft / 2 + 1),                         // odist
                                     FFTW_EXHAUSTIVE);
    fftwf_export_wisdom_to_filename("FFTWF_Wisdom.dat");
}

template <class DataType> void TimeQuad_FFT<float, DataType>::destroy_plans() {
    fftwf_destroy_plan(kernel_plan);
    fftwf_destroy_plan(g_plan);
    fftwf_destroy_plan(h_plan);
}

template <class DataType> uint TimeQuad_FFT<float, DataType>::compute_n_prod(np_double &np_array) {
    py::buffer_info buffer = np_array.request();
    std::vector<ssize_t> shape = buffer.shape;
    uint64_t product = 1;
    for (int i = 0; i < buffer.ndim - 1; i++) {
        product *= shape[i];
    }
    return product;
}

template <class DataType>
Multi_array<float, 2, uint32_t> TimeQuad_FFT<float, DataType>::copy_ks(np_double &np_ks,
                                                                       uint n_prod) {
    /*Only works on contiguous arrays (i.e. no holes)*/
    py::buffer_info buffer = np_ks.request();
    std::vector<py::ssize_t> shape = buffer.shape;     // shape copy
    std::vector<py::ssize_t> strides = buffer.strides; // stride copy
    size_t num_bytes = shape[0] * strides[0] / 2;      // Number of bytes taken by the new array
    size_t len = shape[0] * strides[0] / 8;            // Number of continguous double in ks
    float *new_ptr = (float *)malloc(num_bytes);
    for (uint i = 0; i < len; i++) {
        new_ptr[i] = (float)((double *)(buffer.ptr))[i];
    }
    py::ssize_t strides_m1 = strides.back() / 2; // strides[-1]
    strides.pop_back();
    py::ssize_t strides_m2 = strides.back() / 2; // strides[-2]
    Multi_array<float, 2, uint32_t> ks(new_ptr, n_prod, shape.back() /*shape[-1]*/,
                                       strides_m2 /*strides[-2]*/, strides_m1 /*strides[-1]*/);
    return ks;
}

template <class DataType>
uint TimeQuad_FFT<float, DataType>::compute_l_kernels(np_double &np_array) {
    py::buffer_info buffer = np_array.request();
    std::vector<ssize_t> shape = buffer.shape;
    return shape[buffer.ndim - 1];
}

template <class DataType>
std::vector<ssize_t> TimeQuad_FFT<float, DataType>::get_shape(np_double &np_array) {
    return np_array.request().shape;
}

template <class DataType>
uint64_t
TimeQuad_FFT<float, DataType>::compute_l_data(py::array_t<DataType, py::array::c_style> &data) {
    py::buffer_info buffer = data.request();
    auto shape = buffer.shape;
    return shape[buffer.ndim - 1];
}

template <class DataType> void TimeQuad_FFT<float, DataType>::prepare_kernels(np_double &np_ks) {
    Multi_array<float, 2, uint32_t> ks = copy_ks(np_ks, n_prod);
    float norm_factor = dt / l_fft; /*MOVED IN PREPARE_KERNELS*/
    for (uint j = 0; j < n_prod; j++) {
        /* Value assignment and zero padding */
        for (uint i = 0; i < l_kernel; i++) {
            ((float *)ks_complex(j))[i] = ks(j, i) * norm_factor; /*Normalisation done here*/
        }
        for (uint i = l_kernel; i < 2 * (l_fft / 2 + 1);
             i++) /*Also places a 0 in the extra memory for rc2*/
        {
            ((float *)ks_complex(j))[i] = 0;
        }
        fftwf_execute_dft_r2c(kernel_plan, (float *)ks_complex(j),
                              reinterpret_cast<fftwf_complex *>(ks_complex(j)));
    }
}

template <class DataType>
void TimeQuad_FFT<float, DataType>::execution_checks(
    np_double &ks, py::array_t<DataType, py::array::c_style> &data) {
    if (this->l_data != (uint64_t)data.request().shape[data.request().ndim - 1]) {
        throw std::runtime_error("Error: Data length does not match the length "
                                 "provided at construction.");
    }
    if (this->ks_shape != ks.request().shape) {
        throw std::runtime_error("Error: Shape of 'ks' does not match the shape "
                                 "provided at construction.");
    }
}

template <class DataType>
void TimeQuad_FFT<float, DataType>::execute_py(np_double &ks,
                                               py::array_t<DataType, py::array::c_style> &np_data) {
    execution_checks(ks, np_data);
    if (omp_get_num_threads() != n_threads) // Makes sure the declared number of
                                            // thread if the same as planned
    {
        omp_set_num_threads(n_threads);
    }
    prepare_kernels(ks);
    Multi_array<DataType, 1, uint64_t> data =
        Multi_array<DataType, 1, uint64_t>::numpy_share(np_data);
    execute(data);
}

template <class DataType>
void TimeQuad_FFT<float, DataType>::execute(Multi_array<DataType, 1, uint64_t> &data) {
    uint64_t l_data = data.get_n_i();
    uint n_chunks = compute_n_chunks(l_data, l_chunk);
    uint l_reste = compute_l_reste(l_data, l_chunk);
    if (l_reste != 0) {
        for (uint j = 0; j < n_prod; j++) {
            for (uint k = 0; k < l_reste + l_kernel - 1; k++) {
                quads(j, n_chunks * l_chunk + k) = 0.0;
            }
        }
    }
#pragma omp parallel num_threads(n_threads)
    {
        manage_thread_affinity();
#pragma omp for simd collapse(3) nowait
        for (uint j = 0; j < n_prod; j++) {
            for (uint i = 0; i < n_chunks - 1; i++) {
                for (uint k = l_chunk; k < l_fft; k++) {
                    quads(j, i * l_chunk + k) = 0.0;
                }
            }
        }
#pragma omp for simd collapse(2) nowait
        for (uint th_n = 0; th_n < (uint)n_threads; th_n++) {
            for (uint k = l_chunk; k < l_fft; k++) {
                gs(th_n, k) = 0;
            }
        }
        int this_thread = omp_get_thread_num();
#pragma omp barrier

#pragma omp for
        for (uint i = 0; i < n_chunks; i++) {
            for (uint j = 0; j < l_chunk; j++) {
                gs(this_thread, j) = (float)data[i * l_chunk + j];
            }
            fftwf_execute_dft_r2c(g_plan, gs[this_thread],
                                  reinterpret_cast<fftwf_complex *>(fs[this_thread]));
            for (uint j = 0; j < n_prod; j++) {
                for (uint k = 0; k < (l_fft / 2 + 1); k++) {
                    hs(this_thread, j, k) = ks_complex(j, k) * fs(this_thread, k);
                }
            }
            fftwf_execute_dft_c2r(h_plan, reinterpret_cast<fftwf_complex *>(hs(this_thread)),
                                  (float *)hs(this_thread));
            for (uint j = 0; j < n_prod; j++) {
                for (uint k = 0; k < l_kernel - 1; k++) {
#pragma omp atomic update
                    quads(j, i * l_chunk + k) += ((float *)hs(this_thread, j))[k];
                }
                for (uint k = l_kernel - 1; k < l_chunk; k++) {
                    quads(j, i * l_chunk + k) = ((float *)hs(this_thread, j))[k];
                }
                for (uint k = l_chunk; k < l_fft; k++) {
#pragma omp atomic update
                    quads(j, i * l_chunk + k) += ((float *)hs(this_thread, j))[k];
                }
            }
        }
#pragma omp single
        {
            if (l_reste != 0) {
                uint k = 0;
                for (; k < l_reste; k++) {
                    gs(this_thread, k) = (float)data[n_chunks * l_chunk + k];
                }

                for (; k < l_fft; k++) {
                    gs(this_thread, k) = 0;
                }
                fftwf_execute_dft_r2c(g_plan, gs[this_thread],
                                      reinterpret_cast<fftwf_complex *>(fs[this_thread]));
                for (uint j = 0; j < n_prod; j++) {
                    for (uint k = 0; k < (l_fft / 2 + 1); k++) {
                        hs(this_thread, j, k) = ks_complex(j, k) * fs(this_thread, k);
                    }
                }
                fftwf_execute_dft_c2r(h_plan, reinterpret_cast<fftwf_complex *>(hs(this_thread)),
                                      (float *)hs(this_thread));
                for (uint j = 0; j < n_prod; j++) {
                    for (uint k = 0; k < l_reste + l_kernel - 1; k++) {
                        quads(j, n_chunks * l_chunk + k) += ((float *)hs(this_thread, j))[k];
                    }
                }
            }
        }
    }
}

template <class DataType> np_float TimeQuad_FFT<float, DataType>::get_quads() {
    // Numpy will not copy the array when using the assignement operator=
    float *ptr = quads[0] + l_kernel - 1;
    py::capsule free_dummy(ptr, [](void *f) { ; });

    std::vector<ssize_t> shape_valid = ks_shape;
    shape_valid.pop_back();
    shape_valid.push_back(uint(l_valid));
    std::vector<ssize_t> shape_full = ks_shape;
    shape_full.pop_back();
    shape_full.push_back(uint(l_full));

    std::vector<ssize_t> strides;
    for (uint i = 0; i < shape_full.size(); ++i) {
        ssize_t prod = 1;
        for (uint j = i + 1; j < shape_full.size(); ++j) {
            prod *= shape_full[j];
        }
        strides.push_back(prod * sizeof(float));
    }

    return py::array_t<float, py::array::c_style>(shape_valid, // shape
                                                  strides,   // C-style contiguous strides for float
                                                  ptr,       // the data pointer
                                                  free_dummy // numpy array references this parent
    );
}