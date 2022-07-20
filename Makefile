

HAMR=/work/SENSEI/HAMR-clang_omp/
MPI_FLAGS=-I/usr/include/mpich-x86_64 -L/usr/lib64/mpich/lib -Wl,-rpath -Wl,/usr/lib64/mpich/lib -Wl,--enable-new-dtags -lmpi
OMP_FLAGS=-fopenmp -fopenmp-targets=nvptx64
CFLAGS=-lm -lstdc++ -Wall -Wextra -fPIE
C_OPT_FLAGS=-O3 -march=native -mtune=native
CUDA_OPT_FLAGS=-O3
CUDA_XOPT_FLAGS=-O3,-march=native,-mtune=native

CC=/home/bloring/work/llvm/llvm-install/bin/clang
CXX=/home/bloring/work/llvm/llvm-install/bin/clang++

.PHONY: all
all : stream_compact galaxy_ic plot_ic newtonpp

stream_compact: stream_compact.cu stream_compact.cxx
	$(CC) $(CFLAGS) $(C_OPT_FLAGS) -fPIC stream_compact.cxx -c -o stream_compact_cpu.o
	nvcc $(CUDA_OPT_FLAGS) -arch=sm_75 -dc -o stream_compact_dc.o stream_compact.cu -Xcompiler -fPIE,$(CUDA_XOPT_FLAGS)
	nvcc $(CUDA_OPT_FLAGFS) -arch=sm_75 -dlink -o stream_compact.o stream_compact_dc.o -lcudart -Xcompiler -fPIE,$(CUDA_XOPT_FLAGS)
	rm -f stream_compact.a
	ar cru stream_compact.a stream_compact_cpu.o stream_compact.o stream_compact_dc.o
	ranlib stream_compact.a

galaxy_ic: galaxy_ic.cpp
	g++  $(C_OPT_FLAGS)  galaxy_ic.cpp -o galaxy_ic

plot_ic: plot_ic.cpp
	g++ $(C_OPT_FLAGS)  plot_ic.cpp -o plot_ic

newtonpp: newton.cpp stream_compact.a
	$(CC) $(OMP_FLAGS) $(MPI_FLAGS) $(CFLAGS) $(C_OPT_FLAGS) -I ./ -I $(HAMR)/include newton.cpp -o newtonpp_omp $(HAMR)/lib/libhamr.a stream_compact.a /usr/local/cuda-11.6/lib64/libcudart_static.a

