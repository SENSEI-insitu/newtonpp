#ifndef read_magi_h
#define read_magi_h

int read_magi(MPI_Comm comm,
    const char *h5_file, const char *sum_file,
    patch &dom, patch_data &pd);

#endif
