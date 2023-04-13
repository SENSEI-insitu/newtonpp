#include "stream_util.h"
#include "memory_management.h"
#include "patch.h"
#include "patch_data.h"
#include "patch_force.h"
#include "domain_decomp.h"
#include "communication.h"
#include "initialize_random.h"
#include "initialize_file.h"
#include "solver.h"
#include "write_vtk.h"

#include <iostream>
#include <vector>
#include <mpi.h>

#include <chrono>
using namespace std::literals;
using timer = std::chrono::high_resolution_clock;

#if defined(ENABLE_OMP)
#pragma message("the default allocator targets the GPU")
#else
#pragma message("the default allocator targets the CPU")
#endif

#if defined(ENABLE_CUDA)
#pragma message("Stream compact on the GPU")
#else
#pragma message("Stream compact on the CPU")
#endif

#define DEBUG_IC

int main(int argc, char **argv)
{
    auto start_time = timer::now();

    int rank = 0;
    int n_ranks = 1;

    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &n_ranks);

    double h = 0.;              // time step size
    double nfr = 0.;            // distance for reduced representation
    double eps = 1e4;           // softener
    const char *odir = nullptr; // where to write results
    long nits = 0;              // number of iterations
    long io_int = 0;            // how often to write results

    // load the initial condition and initialize the bodies
    patch_data pd;
    std::vector<patch> patches;
#if defined(DEBUG_IC)
    if (initialize_random(argc, argv, comm, patches, pd, h, eps, nfr, odir, nits, io_int))
        return -1;
#else
    if (initialize_file(argc, argv, comm, patches, pd, h, eps, nfr, odir, nits, io_int))
        return -1;
#endif

    // write the domain decomp
    if (io_int)
        write_vtk(comm, patches, odir);

    // flag nearby patches
    std::vector<int> nf;
    near(patches, nfr, nf);

    // initialize forces
    patch_force pf;
    forces(comm, pd, pf, eps, nf);

    // write initial state
    if (io_int)
       write_vtk(comm, pd, pf, odir);

    if (rank == 0)
        std::cerr << " === init " << (timer::now() - start_time) / 1ms << "ms" << std::endl;

    // iterate
    long it = 0;
    while (it < nits)
    {
        auto it_time  = timer::now();

        // update bodies
        velocity_verlet(comm, pd, pf, h, eps, nf);

        // update partition
        //if (n_ranks > 1)
        {
            hamr::buffer<int> dest(def_alloc());
            partition(comm, patches, pd, dest);
            move(comm, pd, pf, dest);
        }

        // write current state
        if (io_int && (((it + 1) % io_int) == 0))
            write_vtk(comm, pd, pf, odir);

        it += 1;

        if (rank == 0)
            std::cerr << " === it " << it << " : " << (timer::now() - it_time) / 1ms << "ms" << std::endl;
    }

    MPI_Finalize();

    if (rank == 0)
        std::cerr << " === total " << (timer::now() - start_time) / 1s << "s" << std::endl;

    return 0;
}
