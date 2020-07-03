#include <TimeQuad.h>
#include "TimeQuad.tpp"

// Explicit template instanciation
// See: https://docs.microsoft.com/en-us/cpp/cpp/explicit-instantiation?view=vs-2019

#define TIMEQUAD(DataType,QuadsIndexType)\
template class TimeQuad<QuadsIndexType>; \
template void TimeQuad<QuadsIndexType>::execute( DataType* data , uint64_t l_data );\
template void TimeQuad<QuadsIndexType>::execute_py( py::array_t<DataType> data );

TIMEQUAD(int16_t,uint64_t);
TIMEQUAD(int16_t,uint32_t);

#undef TIMEQUAD