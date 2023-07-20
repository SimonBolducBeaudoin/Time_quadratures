#pragma once
#include<stdint.h>
#include <Multi_array.h>

class TimeQuad_algorithm
{
	public :
	// Contructor
	TimeQuad_algorithm(){};
	// Destructor
	virtual ~TimeQuad_algorithm(){};
	
	/* template may not be virtual */
	
	# define VIRTUAL_EXECUTE(DataType) \
		virtual void execute( Multi_array<DataType,1,uint64_t>& data ){};	
	
	VIRTUAL_EXECUTE(int16_t)
    
    virtual void  update_kernels();
							
	#undef VIRTUAL_EXECUTE
};
