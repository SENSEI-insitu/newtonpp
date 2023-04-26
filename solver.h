#ifndef solver_h
#define solver_h

#include "patch_data.h"
#include "patch_force.h"

#include <mpi.h>
#include <vector>

/** Calculates the forces from bodies on this MPI rank. This is written to
 * handle force initialization and should always occur before accumulating
 * remote forces
 */
void forces(const patch_data &pd, patch_force &pf, double G, double eps);

/** Accumulates the forces from bodies on this MPI rank (lpd) with those from
 * another MPI rank (rpd). nf is a map that determines the required resolution.
 * eps is a softening factor.
 */
void forces(const patch_data &lpd, const patch_data &rpd,
    patch_force &pf, double G, double eps);

/** Accumulates the forces from all bodies local and remote (i.e. other MPI
 * ranks). nf is a map that determines the required resolution. eps is a
 * softening factor.
 */
void forces(MPI_Comm comm, patch_data &pd, patch_force &pf,
    double G, double eps, const std::vector<int> &nf);

/** Velocity Verlet:
 *
 * v_{n+1/2} = v_n + (h/2)*F(x_n)
 * x_{n+1} = x_n + h*v_{n+1/2}
 * v_{n+1} = v_{n+1/2} + (h/2)*F(x_{n+1})
 *
 * note: patch_forces must be pre-initialized and held in between calls. nf is
 * a map that determines the required resolution. eps is a softening factor.
 */
void velocity_verlet(MPI_Comm comm,
    patch_data &pd, patch_force &pf, double G, double h, double eps,
    const std::vector<int> &nf);

#endif
