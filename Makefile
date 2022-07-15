

HAMR=/work/SENSEI/HAMR-clang_omp/
MPI_FLAGS=-I/usr/include/mpich-x86_64 -L/usr/lib64/mpich/lib -Wl,-rpath -Wl,/usr/lib64/mpich/lib -Wl,--enable-new-dtags -lmpi
OMP_FLAGS=-fopenmp -fopenmp-targets=nvptx64
CFLAGS=-lm -lstdc++ -g -Wall -Wextra -fPIE

CC=/home/bloring/work/llvm/llvm-install/bin/clang
CXX=/home/bloring/work/llvm/llvm-install/bin/clang++

.PHONY: all
all : stream_compact galaxy_ic plot_ic newtonpp

stream_compact: stream_compact.cu
	nvcc -arch=sm_75 -dc -o stream_compact_dc.o stream_compact.cu -g -G -Xcompiler -fPIE -g
	nvcc -arch=sm_75 -dlink -o stream_compact.o stream_compact_dc.o -lcudart -g -G -Xcompiler -fPIE -g
	rm -f stream_compact.a
	ar cru stream_compact.a stream_compact.o stream_compact_dc.o
	ranlib stream_compact.a

galaxy_ic: galaxy_ic.cpp
	g++ -O0 -g3  galaxy_ic.cpp -o galaxy_ic

plot_ic: plot_ic.cpp
	g++ -O0 -g3  plot_ic.cpp -o plot_ic

newtonpp: newton.cpp stream_compact.a
	$(CC) $(OMP_FLAGS) $(MPI_FLAGS) $(CFLAGS) -I ./ -I $(HAMR)/include newton.cpp -o newtonpp_omp $(HAMR)/lib/libhamr.a stream_compact.a /usr/local/cuda-11.6/lib64/libcudart_static.a

