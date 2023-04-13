#ifndef initialize_file_h
#define initialize_file_h

#include "patch.h"
#include "patch_data.h"

#include <vector>
#include <mpi.h>

/// initialize from data in a file
int initialize_file(int argc, char **argv,
    MPI_Comm comm,
    std::vector<patch> &patches, patch_data &lpd,
    double &h, double &eps, double &nfr,
    const char *&odir, long &nits, long &io_int);

#endif
