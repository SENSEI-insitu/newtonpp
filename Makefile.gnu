
# CPU only for debug
#####################
# HAMR=/work/SENSEI/HAMR-cpu/
# MPI_FLAGS=-I/usr/include/mpich-x86_64 -L/usr/lib64/mpich/lib -Wl,-rpath -Wl,/usr/lib64/mpich/lib -Wl,--enable-new-dtags -lmpi
# OMP_FLAGS=
# CFLAGS=-lm -lstdc++ -Wall -Wextra -fPIE -std=c++17 -fsanitize=address
# C_OPT_FLAGS=-O0 -g -march=native -mtune=native
# CUDA_OPT_FLAGS=-O0 -g -G
# CUDA_XOPT_FLAGS=-O0,-g,-march=native,-mtune=native
# CUDA=/usr/local/cuda-12.0/lib64/
# CC=gcc
# CXX=g++

USE_SENSEI=


MPI_FLAGS=-I/usr/include/mpich-x86_64
MPI_LINK= -L/usr/lib64/mpich/lib -Wl,-rpath -Wl,/usr/lib64/mpich/lib -Wl,--enable-new-dtags -lmpi

OMP_FLAGS=-DNEWTONPP_ENABLE_OMP -DNEWTONPP_ENABLE_CUDA -fopenmp -foffload=nvptx-none -foffload-options=nvptx-none=-march=sm_75 -foffload=-lm -foffload=-latomic -fno-fast-math -fno-associative-math
OMP_LINK=-latomic -lgomp -lrt


CFLAGS=-Wall -Wextra -fPIE -std=c++17
C_OPT_FLAGS=-O3 -march=native -mtune=native
CLINK=-lm -lstdc++

CUDA=/usr/local/cuda-12.0/lib64/
CUDA_OPT_FLAGS=-O3
CUDA_XOPT_FLAGS=-O3,-march=native,-mtune=native
CUDA_LINK=

CC=`which gcc`
CXX=`which g++`

SENSEI=/work/SENSEI/sensei-svtk-install
ifneq ($(USE_SENSEI),)
	SENSEI_LINK=`${SENSEI}/bin/sensei_config --libs`
	SENSEI_FLAGS=`${SENSEI}/bin/sensei_config --cflags`
	SENSEI_PYTHON_DIR=`${SENSEI}/bin/sensei_config --python-dir`
	SENSEI_LINK += -Wl,-rpath=$(SENSEI_PYTHON_DIR)
endif

ifneq ($(USE_SENSEI),)
	HAMR=$(SENSEI)/
else
	HAMR=/work/SENSEI/HAMR-gcc_omp/
endif
HAMR_FLAGS=-I$(HAMR)/include
HAMR_LINK=$(HAMR)/lib64/libhamr.a

.PHONY: all
all : stream_compact.a newtonpp_gcc_omp

.PHONY: ics
ics: galaxy_ic plot_ic

.PHONY: clean
clean:
	rm -f *.o stream_compact.a newtonpp_gcc_omp

stream_compact.a: stream_compact.cu stream_compact.cxx
	$(CXX) $(CFLAGS) $(C_OPT_FLAGS) -fPIC stream_compact.cxx -c -o stream_compact_cpu.o
	nvcc $(CUDA_OPT_FLAGS) -arch=sm_75 -dc -o stream_compact_dc.o stream_compact.cu -Xcompiler -fPIE,$(CUDA_XOPT_FLAGS)
	nvcc $(CUDA_OPT_FLAGS) -arch=sm_75 -dlink -o stream_compact.o stream_compact_dc.o -lcudart -Xcompiler -fPIE,$(CUDA_XOPT_FLAGS)
	rm -f stream_compact.a
	ar cru stream_compact.a stream_compact_cpu.o stream_compact.o stream_compact_dc.o
	ranlib stream_compact.a

%.o: %.cpp
	$(CXX) $(OMP_FLAGS) $(MPI_FLAGS) $(CFLAGS) $(C_OPT_FLAGS) $(HAMR_FLAGS) -I./ -c $< -o $@

newtonpp_objs=domain_decomp.o initialize_file.o patch_data.o communication.o \
	 initialize_random.o patch.o patch_force.o solver.o write_vtk.o

ifneq ($(USE_SENSEI),)
newtonpp_objs += sensei_adaptor.o

sensei_adaptor.o: sensei_adaptor.cpp
	$(CXX) $(OMP_FLAGS) $(MPI_FLAGS) $(CFLAGS) $(C_OPT_FLAGS) $(HAMR_FLAGS) $(SENSEI_FLAGS) -I./ -c $< -o $@

endif


newtonpp_gcc_omp:  stream_compact.a $(newtonpp_objs) newton.cpp
	$(CXX) $(OMP_FLAGS) $(MPI_FLAGS) $(CFLAGS) $(C_OPT_FLAGS) $(HAMR_FLAGS) -I./ \
	newton.cpp $(newtonpp_objs) \
	stream_compact.a ${CUDA}/libcudart_static.a \
	$(SENSEI_LINK) $(HAMR_LINK) $(CLINK) $(MPI_LINK) $(OMP_LINK) \
	-o newtonpp_gcc_omp

galaxy_ic: galaxy_ic.cpp
	g++  $(C_OPT_FLAGS)  galaxy_ic.cpp -o galaxy_ic

plot_ic: plot_ic.cpp
	g++ $(C_OPT_FLAGS)  plot_ic.cpp -o plot_ic
