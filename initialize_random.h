#ifndef initialize_random_h
#define initialize_random_h

#include "patch.h"
#include "patch_data.h"

#include <vector>
#include <mpi.h>

/** generate an initial condition for nb randomly positioned bodies in a 3D
 * domain. A good starting point for h is 4.*24.*3600.
 */
int initialize_random(MPI_Comm comm, long nb,
    std::vector<patch> &patches, patch_data &lpd,
    double &nfr);

#endif
