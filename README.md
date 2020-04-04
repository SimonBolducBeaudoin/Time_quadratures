# Time_quadratures-OTF

The Time_quadratures-OTF library is to do fast convolution calculation (on-the-fly) on discrete voltage measurement in order to calculate time domain quadratures p(t) and q(t). Therefore kernels for the convolutions are fixed (see : Unidimensional time-domain quantum optics, Physical Review A 100, 023822 (2019)). 
 
The library is written in C++ and is using Pybin11 to provide a python interface. Version 0.1 provides the classes "Time_Quad" and "Numpy1DArray" (same name in C++ and Python).

The library is intended to be used in the following way
1. Make an instance of the Time_Quad class
   * `>>> X = Time_Quad_double_int16_t( L_kernel = 2 , L_data = 16, L_FFT = 8, dt = 0.03125, Convolution_type = 'fft');`
   * FFT calculations and outputs are of type double and  data input is of type int16_t
   * Attributes values L_kernel,  L_data  and dt can only be set at construction. They are used for memory allocation and initialization.
   * On the other hand L_FFT and Convolution_type can be change on the fly. They change the behavior of the execute function, which tells X to execute the convolution.
   * Note that the "dt" attribute is in [ns]
2. Numpy arrays are created  using the copy = 'false' argument
    * `>>> data = array(X.data, copy = False);`\
`Numpy array are created  using the copy = 'false' argument`\
`kernel_p = array(X.kernel_p, copy = False);`\
`kernel_q = array(X.kernel_q, copy = False);`\
`p_full = array(X.p_full, copy = False);`\
`q_full = array(X.q_full, copy = False);`
    * Doing so makes it so that the np.array object shares  the memory allocated by the  C++. You can read and write in that array, but you cant change its size. 
3. Write the discrete voltage measurements into the data array (here I use numpy to generate random data).
    * `>>> data[:] = normal( 0 , 2**12, L_data);`
4. ( Optopnal ) modify the convolution's execution by modifying L_FFT and Convolution_type.
    * `>>> X.L_FFT = 256;`\
`>>> X.Convolution_type = 'fft'  # or 'direct'`
    * It is the user's responsibility  to ensure that :\
    ` >>> iseven( L_kernel ) ==true ;`\
    `L_kernel < L_FFT ;`\
    `L_kernel < L_data ;`
5. Execute the convolution
    * `>>> X.execute()` 
    * The result of the convolution is then in the p_full  and q_full numpy arrays.
    * Note that full has the same meaning as in the numpy.convole function :\
" This returns the convolution at each point of overlap, with an output shape of (L_kernel+L_data-1,). At the end-points of the convolution, the signals do not overlap completely, and boundary effects may be seen."
    * A "valid" option will soon be implemented.
6. (Optional) iterate steps 3, 4 and 5 in a loop.
