#include "python_submodule.h"

//Python Binding and Time_Quad class instances.
PYBIND11_MODULE(time_quadratures, m)
{
    m.doc() = " Fast caculations of time quadratrures. See markdown documentation.";
	init_TimeQuad(m);
}

