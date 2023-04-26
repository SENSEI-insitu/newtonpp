#ifndef initialize_random_h
#define initialize_random_h

#include "patch.h"
#include "patch_data.h"

#include <vector>
#include <mpi.h>

/** generate an initial condition for nb randomly positioned bodies in a 3D
 * domain.
 */
int initialize_random(MPI_Comm comm, long nb,
    std::vector<patch> &patches, patch_data &lpd,
    double &h, double &eps, double &nfr);

#endif
