#pragma once

#include "../../includes/header_common.h"
#include "../../SM-Multi_array/Multi_array.h"

class TimeQuad_algorithm
{
	public :
	// Contructor
	TimeQuad_algorithm(){};
	// Destructor
	virtual ~TimeQuad_algorithm(){};
	
	/* template may not be virtual */
	
	# define VIRTUAL_EXECUTE(DataType) \
		virtual void execute( DataType* data ){};	
	
	VIRTUAL_EXECUTE(int16_t)
							
	#undef VIRTUAL_EXECUTE
};
