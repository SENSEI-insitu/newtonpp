#ifndef domain_decomp_h
#define domain_decomp_h

#include "memory_management.h"
#include "patch.h"
#include "patch_data.h"
#include "patch_force.h"

#include <vector>
#include <mpi.h>


/** returns a N x N matrix with the i,j set to 1 where distance between
 * patch i and j is less than r and 0 otherwise. this identifies the patch-pairs
 * that need to be fully exchanged to caclulate forces.
 */
void near(const std::vector<patch> &p, double nfr, std::vector<int> &nf);

#pragma omp declare target
/// splits a patch into two new patches of equal size along in the named direction
void split(int dir, const double *p0_x, double *p1_x, double *p2_x);

/// return true if a point is inside a patch
int inside(const double *p, double x, double y, double z);
#pragma omp end declare target

/// splits a patch into two new patches of equal size along in the named direction
void split(int dir, const patch &p0, patch &p1, patch &p2);

/// calculates the area fraction of all patches relative to the area of the3 total domain
void area(const patch &dom, const std::vector<patch> &p, hamr::buffer<double> &area);

/// splits the domain into the requested number of sub-domains
std::vector<patch> partition(const patch &dom, size_t n_out);

/// assigns a collection of patches to MPI ranks
void assign_patches(std::vector<patch> &p, int n_ranks);

/// assigns bodies to patches
void partition(MPI_Comm comm, const std::vector<patch> &ps,
    const patch_data &pd, hamr::buffer<int> &dest);

/** given a set of bodies, per-body forces, and a mapping of bodies to MPI
 * ranks, package the bodies belonging to rank for shipment to that rank
 */
long package(const patch_data &pdi, const patch_force &pfi,
    const hamr::buffer<int> &dest, int rank, patch_data &pdo, patch_force &pfo);

/** given a mapping of bodies to MPI ranks package and ship. shipped bodies are
 * removed from the local data.
 */
void move(MPI_Comm comm, patch_data &pd, patch_force &pf, const hamr::buffer<int> &dest);
void move2(MPI_Comm comm, patch_data &pd, patch_force &pf, const hamr::buffer<int> &dest);

#endif
