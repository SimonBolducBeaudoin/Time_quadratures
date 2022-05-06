NAME = time_quadratures
PYLIB_EXT = $(if $(filter $(OS),Windows_NT),.pyd,.so)
TARGET_STATIC = lib$(NAME).a
TARGET_PYLIB = bin/$(NAME)$(PYLIB_EXT)

MULTI_ARRAY = ../Multi_array
OMP_EXTRA = ../Omp_extra
SCOPED_TIMER = ../Scoped_timer
WINDOWING = ../Windowing
SPECIAL_FUNC = ../Special_functions
NUMERICAL_INT = ../Numerical_integration
TIME_QUAD = ../Time_quadratures
LIBS = ../libs

ODIR = obj
LDIR = lib
SDIR = src

OMP_EXTRA_OBJ = $(wildcard $(OMP_EXTRA)/$(ODIR)/*.o)
WINDOWING_OBJ = $(wildcard $(WINDOWING)/$(ODIR)/*.o)
SPECIAL_FUNC_OBJ = $(wildcard $(SPECIAL_FUNC)/*$(ODIR)/*.o)
NUMERICAL_INT_OBJ = $(wildcard $(NUMERICAL_INT)/*$(ODIR)/*.o)

EXTERNAL_OBJ = $(OMP_EXTRA_OBJ) $(WINDOWING_OBJ) $(SPECIAL_FUNC_OBJ) $(NUMERICAL_INT_OBJ)
EXTERNAL_INCLUDES = -I$(MULTI_ARRAY)/$(SDIR) -I$(OMP_EXTRA)/$(SDIR) -I$(WINDOWING)/$(SDIR) \
					-I$(NUMERICAL_INT)/$(SDIR) -I$(SPECIAL_FUNC)/$(SDIR) \
					-I$(TIME_QUAD)/$(SDIR) -I$(SCOPED_TIMER)/$(SDIR) 

SRC  = $(wildcard $(SDIR)/*.cpp)
OBJ  = $(patsubst $(SDIR)/%.cpp,$(ODIR)/%.o,$(SRC))
ASS  = $(patsubst $(SDIR)/%.cpp,$(ODIR)/%.s,$(SRC))
DEPS = $(OBJ:.o=.d)

CXX = $(OS:Windows_NT=x86_64-w64-mingw32-)g++
OPTIMIZATION = -O3 -march=native
CPP_STD = -std=c++14
WARNINGS = -Wall
MINGW_COMPATIBLE = $(OS:Windows_NT=-DMS_WIN64 -D_hypot=hypot)
DEPS_FLAG = -MMD -MP

POSITION_INDEP = -fPIC
SHARED = -shared

OMP = -fopenmp -fopenmp-simd
FFTW= -lfftw3
MATH = -lm

ifeq ($(shell hostname),Simon-T14) 
	PY = $(OS:Windows_NT=/c/Anaconda3/envs/python2/)python
else
    PY = $(OS:Windows_NT=/c/Anaconda2/)python
endif

PY_INCL := $(shell $(PY) -m pybind11 --includes)

ifeq ($(shell hostname),Simon-T14) 
	PY_LINKS  = $(OS:Windows_NT=-L /c/Anaconda3/envs/python2/ -lpython27)
else
    PY_LINKS  = $(OS:Windows_NT=-L /c/Anaconda2/ -lpython27)
endif

LINKS = $(MATH) $(FFTW) $(OMP) $(PY_LINKS)
LINKING = $(CXX) $(OPTIMIZATION) $(POSITION_INDEP) $(SHARED)  -o $(TARGET_PYLIB) $(OBJ) $(LINKS) $(EXTERNAL_OBJ) $(DEPS_FLAG) $(MINGW_COMPATIBLE)
STATIC_LIB = ar cr $(TARGET_STATIC) $(OBJ) 

INCLUDES = $(OMP) $(PY_INCL) $(EXTERNAL_INCLUDES)
COMPILE = $(CXX) $(CPP_STD) $(OPTIMIZATION) $(POSITION_INDEP) $(WARNINGS) -c -o $@ $< $(INCLUDES) $(DEPS_FLAG) $(MINGW_COMPATIBLE)
ASSEMBLY = $(CXX) $(CPP_STD) $(OPTIMIZATION) $(POSITION_INDEP) $(WARNINGS) -S -o $@ $< $(INCLUDES) $(DEPS_FLAG) $(MINGW_COMPATIBLE)

LINK_BENCHMARK_CUSTOM = $(MATH) $(FFTW) $(OMP) $(PY_LINKS) $(EXTERNAL_OBJ)

LINK_BENCHMARK = \
	$(LINK_BENCHMARK_CUSTOM) \
	-L$(LIBS)/benchmark/build/src -lbenchmark -lpthread -lshlwapi

LINKING_BENCHMARK = \
	$(CXX)\
	-o $@ $< \
	-O3 -march=native \
	$(LINK_BENCHMARK)\
	$(DEPS_FLAG) $(MINGW_COMPATIBLE) \
	
INCLUDES_BENCHMARK = \
	-I $(LIBS)/benchmark/include \
	$(INCLUDES)
	
COMPILE_BENCHMARK = \
	$(CXX) $(CPP_STD) $< -O3 -march=native \
	$(INCLUDES_BENCHMARK) \
	$(DEPS_FLAG) $(MINGW_COMPATIBLE) \
	-c -o $@

python_debug_library : $(TARGET_PYLIB)

compile_objects : $(OBJ)

assembly : $(ASS)

all : $(TARGET_PYLIB) $(TARGET_STATIC) $(OBJ) $(ASS)

static_library : $(TARGET_STATIC)

benchmark : benchmark.exe

benchmark.exe : benchmark.o
	@ echo " "
	@ echo "---------Compile $@ ---------"
	$(LINKING_BENCHMARK)

benchmark.o : benchmark.cpp
	@ echo " "
	@ echo "---------Compile $@ from $< ---------"
	$(COMPILE_BENCHMARK)	

$(TARGET_PYLIB): $(OBJ)
	@ echo " "
	@ echo "---------Compile library $(TARGET_PYLIB)---------"
	$(LINKING)

$(TARGET_STATIC) : $(OBJ)
	@ echo " "
	@ echo "---------Compiling static library $(TARGET_STATIC)---------"
	$(STATIC_LIB)
	
$(ODIR)/%.o : $(SDIR)/%.cpp
	@ echo " "
	@ echo "---------Compile object $@ from $<--------"
	$(COMPILE)
	
$(ODIR)/%.s : $(SDIR)/%.cpp
	@ echo " "
	@ echo "---------Assembly $@ from $<--------"
	$(ASSEMBLY)
	
-include $(DEPS)

clean:
	@rm -f $(TARGET_PYLIB) $(TARGET_STATIC) $(OBJ) $(ASS) $(DEPS) benchmark.o benchmark.exe
	 	 
.PHONY: all , clean , python_debug_library , compile_objects , static_library , assembly , benchmark