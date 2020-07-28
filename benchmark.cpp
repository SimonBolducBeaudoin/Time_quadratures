/* 
	Comment template definition before compiling benchmark
	and link to histogram.o and histogram2D.o
*/
#include <TimeQuad.h>

#include <benchmark/benchmark.h> 
#include <fftw3.h>
#include <complex>
#include <random>
// see : https://en.cppreference.com/w/cpp/numeric/random/uniform_int_distribution
#include <type_traits>
#include <stdio.h>

/*
How to use the benchmark :
	- simple call in cygwin : ./benchmark.exe will run the benchark and display the result in the console
	Options
	- Showing custom counter in individual colums : --benchmark_counters_tabular=true
	- Repetitions : --benchmark_repetitions=4
		Ex: repeats the experiment 4 times
		Statistics will automatically be showned
	- Output files : --benchmark_out=<filename> --benchmark_out_format={json|console|csv}
	- Running a subset of benchmarks : --benchmark_filter=<regex>
		Ex :
			+ Running all 1D histogram : --benchmark_filter=Histogram1D
			+ Running all 2D histogram : --benchmark_filter=Histogram2D
*/

template
<
	class DataType ,
	std::enable_if_t< std::is_integral<DataType>::value , int > = 0
>
DataType* make_rand( uint64_t size )
{
	std::random_device rd;  
    std::mt19937 gen(rd());
	std::uniform_int_distribution<int> distrib(0, (1<<sizeof(DataType)*8) -1 );
	DataType* x = (DataType*) malloc(size*sizeof(DataType));
	for(unsigned int i = 0 ; i < size ; i++)
	{
		x[i] = distrib(gen);
	}
	return x ; // Dont forget to deallocate
}

static void Bench_TimeQuad_direct(benchmark::State& state)
{
	uint l_kernel = state.range(0) ;
	uint n_kernels = 1 ;
	double Z = 1.0 ;
	uint l_data = state.range(1) ;
	double dt = 0.03125 ;
	double f_max_analogue = 10.0 ;
	double f_min_analogue = 0.5 ;
	double alpha = 0.25 ;
	int n_threads = state.range(2) ;
	
	TimeQuad<uint64_t> X(l_kernel,n_kernels,Z,l_data,dt,f_max_analogue,f_min_analogue,alpha,n_threads);
	int16_t* data = make_rand<int16_t>((uint64_t)l_data);
	for (auto _ : state)
	{	
		X.execute(data,l_data);
	}
	free(data);
	
	state.counters["Sa/Sec"] = benchmark::Counter(l_data*state.iterations(), benchmark::Counter::kIsRate);
	state.counters["l_data [Sa]"] =  l_data ;
	state.counters["l_kernel"] = l_kernel ;
	state.counters["n_threads"] = n_threads ;
}

static void Bench_TimeQuad_FFT_basic(benchmark::State& state)
{
	uint l_kernel = state.range(0) ;
	uint n_kernels = 1 ;
	double Z = 1.0 ;
	uint l_data = state.range(1) ;
	double dt = 0.03125 ;
	double f_max_analogue = 10.0 ;
	double f_min_analogue = 0.5 ;
	double alpha = 0.25 ;
	uint l_fft = state.range(2) ;
	int n_threads = state.range(3) ;
	
	TimeQuad<uint64_t> X(l_kernel,n_kernels,Z,l_data,dt,f_max_analogue,f_min_analogue,alpha,l_fft,n_threads);
	int16_t* data = make_rand<int16_t>((uint64_t)l_data);
	for (auto _ : state)
	{	
		X.execute(data,l_data);
	}
	free(data);
	
	state.counters["Sa/Sec"] = benchmark::Counter(l_data*state.iterations(), benchmark::Counter::kIsRate);
	state.counters["l_data [Sa]"] =  l_data ;
	state.counters["l_fft"] = l_fft ;
	state.counters["l_kernel"] = l_kernel ;
	state.counters["n_threads"] = n_threads ;
}

static void Bench_TimeQuad_FFT_advanced(benchmark::State& state)
{
	uint l_kernel = state.range(0) ;
	uint n_kernels = 1 ;
	double Z = 1.0 ;
	uint l_data = state.range(1) ;
	double dt = 0.03125 ;
	double f_max_analogue = 10.0 ;
	double f_min_analogue = 0.5 ;
	double alpha = 0.25 ;
	uint l_fft = state.range(2) ;
	int n_threads = state.range(3) ;
	uint howmany = state.range(4) ;
	
	TimeQuad<uint64_t> X(l_kernel,n_kernels,Z,l_data,dt,f_max_analogue,f_min_analogue,alpha,l_fft,howmany,n_threads);
	int16_t* data = make_rand<int16_t>((uint64_t)l_data);
	for (auto _ : state)
	{	
		X.execute(data,l_data);
	}
	free(data);
	
	state.counters["Sa/Sec"] = benchmark::Counter(l_data*state.iterations(), benchmark::Counter::kIsRate);
	state.counters["l_data [Sa]"] =  l_data ;
	state.counters["l_fft"] = l_fft ;
	state.counters["howmany"] = howmany ;
	state.counters["l_kernel"] = l_kernel ;
	state.counters["n_threads"] = n_threads ;
}

static void CustomArguments_direct(benchmark::internal::Benchmark* b) 
{
	int I[] = {(1<<6)+1,(1<<7)+1,(1<<8)+1};
	int J[] = {1<<20,1<<24,1<<28};
	int K[] = {1};
	for (int i = 0; i < 3; ++i) 
	for (int j = 0; j < 3; ++j) 
	for (int k = 0; k < 1; ++k) 
		b->Args({I[i],J[j],K[k]});
}

static void CustomArguments_FFT_basic(benchmark::internal::Benchmark* b) 
{
	int I[] = {(1<<6)+1,(1<<7)+1,(1<<8)+1};
	int J[] = {1<<20,1<<24,1<<28};
	int K[] = {1<<7,1<<8,1<<9,1<<10,1<<11};
	int L[] = {1,6,12};
	for (int i = 0; i < 3; ++i) 
	for (int j = 1; j < 2; ++j) 
	for (int k = i; k < 5; ++k) 
	for (int l = 2; l < 3; ++l) 
		b->Args({I[i],J[j],K[k],L[l]});
}

static void CustomArguments_FFT_advanced(benchmark::internal::Benchmark* b) 
{
	int I[] = {(1<<6)+1,(1<<7)+1,(1<<8)+1};
	int J[] = {1<<24,1<<28};
	int K[] = {1<<7,1<<8,1<<9,1<<10,1<<11};
	int L[] = {1,6,12};
	int M[] = {1,2,4,8,16,64,1<<7,1<<8,1<<10,1<<12};
	for (int i = 0; i < 3; ++i) 
	for (int j = 0; j < 1; ++j) 
	for (int k = i; k < 5; ++k) 
	for (int l = 2; l < 3; ++l) 
	for (int m = 0; m < 10; ++m) 
		b->Args({I[i],J[j],K[k],L[l],M[m]});
}

// BENCHMARK(Bench_TimeQuad_direct)	->UseRealTime()->Apply(CustomArguments_direct)				->ComputeStatistics ("max",[](const std::vector<double>&v)->double{return*(std::max_element(std::begin(v),std::end(v)));})->ComputeStatistics("min",[](const std::vector<double>&v)->double{return*(std::min_element(std::begin(v),std::end(v)));});
BENCHMARK(Bench_TimeQuad_FFT_basic)		->UseRealTime()->Apply(CustomArguments_FFT_basic)		->ComputeStatistics ("max",[](const std::vector<double>&v)->double{return*(std::max_element(std::begin(v),std::end(v)));})->ComputeStatistics("min",[](const std::vector<double>&v)->double{return*(std::min_element(std::begin(v),std::end(v)));});
// BENCHMARK(Bench_TimeQuad_FFT_advanced)	->UseRealTime()->Apply(CustomArguments_FFT_advanced)	->ComputeStatistics ("max",[](const std::vector<double>&v)->double{return*(std::max_element(std::begin(v),std::end(v)));})->ComputeStatistics("min",[](const std::vector<double>&v)->double{return*(std::min_element(std::begin(v),std::end(v)));});

BENCHMARK_MAIN();
























