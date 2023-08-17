# Time_quadratures-OTF


The Time_quadratures-OTF library is to do fast convolution calculation (on-the-fly) on discrete voltage measurement in order to calculate time domain quadratures p(t) and q(t). Therefore kernels for the convolutions are fixed (see : Unidimensional time-domain quantum optics, Physical Review A 100, 023822 (2019)).  
 
The library is written in C++ and is using Pybin11 to provide a python interface.

It implements the following computation :
$$
	x_{\text{in}+\delta,\Theta}(t) 
	=
	\underbrace
	{
		k_\Theta(t) \circledast \beta(t) \circledast g^{-1}(t)
	}_\text{numerical}
	\circledast
	\underbrace
	{ 
		g(t) \circledast v_m(t) 
	}_\text{physical}
$$

$$
	x_{\text{in}+\delta,\Theta}(t)
	=
	\underbrace
	{
		k_\Theta(t) \circledast \beta(t) 
	}_\text{Base}
	\circledast
	\underbrace
	{ 
		g^{-1}(t) \circledast g(t) \circledast v(t) 
	}_\text{gain compensation}
$$

$$
	x_{\text{in}+\delta,\Theta}(t)
	\approx
	\Delta t \mathcal{DFT}^{-1}\left\{ \frac{ \beta(f) }{ g(f) } \mathcal{DFT}[k_\Theta(t)] \mathcal{DFT}\{v_m(t)\}   \right\}
$$

Which is a valid computation for a time quadrature of base $\beta(t)$ for state that dont posses 
a phase relation ship between their various frequencies, as are chaotic states.

Kernels are computed by the class and are of the form :
	- $k_0(t) = \sqrt{ \frac{2}{Zh}} \frac{1}{|t|}$
	- $k_{\pi/2}(t) = \sqrt{ \frac{2}{Zh}} \frac{\text{signum}(t)}{|t|}$
	- $k_{\pi/4}(t) = \text{Heaviside}(t)\sqrt{ \frac{2}{Zh}} \frac{1}{|t|}$
	- $k_{3\pi/4}(t) = \text{Heaviside}(-t)\sqrt{ \frac{2}{Zh}} \frac{1}{|t|}$

Units :
	- $h$ : is a private const =  6,62607004 × 10-25 [J ns] ;
	- $Z$ [Ohm] ;
	- Time variables t and $\Delta t$ are in [ns]: 
	- Kernels have units of (V^2/Hz)^-1/2 * ns^-1 :
		- The  ns^-1 are canceled in integration since $\Delta t$ is in [ns]
	- $V$ [Volt] ;
	- $g(f)$ unitless : We assume that the user as preemtively measured the gain of the system
	- $\beta(f)$ [GHz^-1/2]
	- $x$ [] : the output quadrature has no units
	
Distinction between DFT's and Fourier transforms :
	g(f) and beta(f) are Fourier transform and not DFT's
	Recall that the Fourier transform can be approximated by $\Delta * \mathcal{DFT}[x_i]$
	
Choice of mode :
	Reccal that $\int |\beta(f)|^2 df =1$ for the definition of a mode to be properly normalized
	WARNING : If beta(f) are too monochromatic (to narrow in frequencies). 
		Their counterpart in time beta(t) will be clipped by the unavoidable windowing that comes with discretization.
		This will intern cause the real/effective beta(f) to differ from given beta(f). 
		To avoid this the given beta(f) must be wide enough so that beta(t) is near 0 
		when approching the edges of the time frame. 

The class is structured in the following way :
	It builds the wanted kernels and chooses at run time the execution algorithm based on constructor arguments. 
	Constructors default inputs :
		- Z 			: is the scalar impedance of the transmission line on which measurement are made.
		- dt 			: is the time step of the aquisition card
		- l_kernel 		: the length of a kernels in direct direct space
		- l_data 		: is the lenght of the numpy array that is going to be passed to the execution function
			- It is used to deduce the lenght of the convolution output. 
				Memory is allocated but only for the quadratures and not for the data array.
		- kernel_conf 	: determine which kernel to use
			- is either 0,1 or 2
			- 0 : only q(t) will be calculated
				$k_{\pi/2}(t) = \sqrt{ \frac{2}{Zh}} \frac{\text{signum}(t)}{|t|}$
			- 1 : p(t) and q(t) will be calculated
				$k_0(t) = \sqrt{ \frac{2}{Zh}} \frac{1}{|t|}$
				$k_{\pi/2}(t) = \sqrt{ \frac{2}{Zh}} \frac{\text{signum}(t)}{|t|}$
			- 2 (not implemented yet) : $x_{\pi/4}(t)$ and $x_{3\pi/4}(t)$ will be calculated
				$k_{\pi/4}(t) = \text{Heaviside}(t)\sqrt{ \frac{2}{Zh}} \frac{1}{|t|}$
				$k_{3\pi/4}(t) = \text{Heaviside}(-t)\sqrt{ \frac{2}{Zh}} \frac{1}{|t|}$				
		- filters(j,i) 	: 2D array of filters (beta(f)) that are defining the modes
			- Filters are half-complexe values (they assume a real input/rfft)
			- i (the fast iterating index) : frequencies index
			- j	: Filter index
		- g(f) 			: is the total gain of the system (between the sample and the virtal ideal voltmeter after the aquisition card)
		- f_max_analogue
		- f_min_analogue
		- alpha
		- n_threads : the number of thread to use for computation
	
	Constructors optionals inputs :
		- l_fft : enable fft convolution algorithm
		- howmany : enable fft advanced convolution algorithm
			
				
		
	It is host to a few important arrays
		ks is a list of the different kernels to used for the convolution
			
			It can comme 
			
