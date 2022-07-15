

all:
	mpicc newton.cpp -lm -lstdc++ -g -Wall -Wextra

stream_compact: stream_compact.cu
	nvcc -arch=sm_75 -dc -o stream_compact_dc.o stream_compact.cu -g -G -Xcompiler -fPIE -g
	nvcc -arch=sm_75 -dlink -o stream_compact.o stream_compact_dc.o -lcudart -g -G -Xcompiler -fPIE -g
	rm -f stream_compact.a
	ar cru stream_compact.a stream_compact.o stream_compact_dc.o
	ranlib stream_compact.a


