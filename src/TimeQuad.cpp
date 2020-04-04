#include "../includes/TimeQuad.h"
#include "TimeQuad.tpp"

// Explicit template instanciation
// See: https://docs.microsoft.com/en-us/cpp/cpp/explicit-instantiation?view=vs-2019

template void TimeQuad::execute( int16_t* data , uint64_t l_data );
template void TimeQuad::execute_py( py::array_t<int16_t> data );