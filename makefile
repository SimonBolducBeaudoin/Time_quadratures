# All the external objects that the current submodule depends on
# Those objects have to be up to date
tempo2 = $(wildcard ../SM-Scoped_timer/obj/*.o)
tempo3 = $(wildcard ../SM-Omp_extra/obj/*.o)
tempo4 = $(wildcard ../SM-Windowing/obj/*.o)
tempo5 = $(wildcard ../SM-Special_functions/*.o)
tempo6 = $(wildcard ../SM-Numerical_integration/*.o)
tempo7 = $(wildcard ../SM-Multi_array/*.o)
EXTERNAL_OBJ = $(tempo1) $(tempo2) $(tempo3) $(tempo4) $(tempo5) $(tempo6) $(tempo7)

TARGET_NAME = time_quadratures
TARGET_STATIC = $(TARGET_NAME).a
PYLIB_EXT = $(if $(filter $(OS),Windows_NT),.pyd,.so)
TARGET_PYLIB = ../Python/$(TARGET_NAME)$(PYLIB_EXT)

# standard subdirectories
IDIR = includes
ODIR = obj
LDIR = lib
SDIR = src

# Lits of .c and corresponding .o and .h
SRC  = $(wildcard $(SDIR)/*.cpp)
OBJ  = $(patsubst $(SDIR)/%.cpp,$(ODIR)/%.o,$(SRC))
DEPS = $(OBJ:.o=.d)
# HEAD = $(patsubst $(SDIR)/%.cpp,$(IDIR)/%.h,$(SRC))

# Toolchain, using mingw on windows
CC = $(OS:Windows_NT=x86_64-w64-mingw32-)g++

# flags
CFLAGS = -Ofast -march=native -std=c++14 -MMD -MP -Wall $(OS:Windows_NT=-DMS_WIN64 -D_hypot=hypot)
OMPFLAGS = -fopenmp -fopenmp-simd
FFTWFLAGS = -lfftw3
MATHFLAGS = -lm
SHRFLAGS = -fPIC -shared

# Python directories
PY = $(OS:Windows_NT=/c/Anaconda2/)python
ifeq ($(USERNAME),simon)
    PY = $(OS:Windows_NT=/cygdrive/c/Anaconda2/)python
endif
ifeq ($(USERNAME),Sous-sol)
    PY = $(OS:Windows_NT=/cygdrive/c/ProgramData/Anaconda2/)python
endif

# includes
PYINCL := $(shell $(PY) -m pybind11 --includes)
ifneq ($(OS),Windows_NT)
    PYINCL += -I /usr/include/python2.7/
endif

# libraries
PYLIBS = $(OS:Windows_NT=-L /c/Anaconda2/libs/ -l python27) $(PYINCL)
ifeq ($(USERNAME),simon)
    PYLIBS = $(OS:Windows_NT=-L /cygdrive/c/Anaconda2/libs/ -l python27) $(PYINCL)
endif
ifeq ($(USERNAME),Sous-sol)
    PYLIBS = $(OS:Windows_NT=-L /cygdrive/c/ProgramData/Anaconda2/libs/ -l python27) $(PYINCL) 
endif

$(TARGET_PYLIB): $(OBJ)
	@ echo " "
	@ echo "---------Compile library $(TARGET_PYLIB)---------"
	$(CC) $(SHRFLAGS) -o $(TARGET_PYLIB) $(OBJ) $(EXTERNAL_OBJ) $(CFLAGS) $(OMPFLAGS) $(FFTWFLAGS) $(MATHFLAGS) $(PYLIBS)
	
$(ODIR)/%.o : $(SDIR)/%.cpp
	@ echo " "
	@ echo "---------Compile object $@ from $<--------"
	$(CC) $(SHRFLAGS) -c -Wall -o $@ $< $(CFLAGS) $(FFTWFLAGS) $(MATHFLAGS) $(OMPFLAGS) $(PYLIBS) 

-include $(DEPS)

clean:
	@rm -f $(TARGET_PYLIB) $(TARGET_STATIC) $(OBJ)
	 	 
.PHONY: clean, dummy