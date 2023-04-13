#ifndef write_vtk_h
#define write_vtk_h

#include "patch.h"
#include "patch_data.h"
#include "patch_force.h"

#include <mpi.h>

/** write bodies and per-body forces to disk in the named directory in a VTK
 * format. Can be visualized using ParaView.
 */
void write_vtk(MPI_Comm comm, const patch_data &pd,
    const patch_force &pf, const char *dir);

/** write patches to disk in the named directory in a VTK format. Can be
 * visualized using ParaView.
 */
void write_vtk(MPI_Comm comm, const std::vector<patch> &patches, const char *dir);

#endif
