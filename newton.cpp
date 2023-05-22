#include "stream_util.h"
#include "memory_management.h"
#include "patch.h"
#include "patch_data.h"
#include "patch_force.h"
#include "domain_decomp.h"
#include "communication.h"
#include "initialize_random.h"
#include "read_magi.h"
#include "solver.h"
#include "write_vtk.h"
#include "command_line.h"
#if defined(NEWTONPP_ENABLE_SENSEI)
#include "insitu.h"
#endif

#include <iostream>
#include <vector>
#include <mpi.h>

#include <chrono>
using namespace std::literals;
using timer = std::chrono::high_resolution_clock;

#if defined(NEWTONPP_ENABLE_OMP)
#pragma message("the default allocator targets the GPU")
#else
#pragma message("the default allocator targets the CPU")
#endif

#if defined(NEWTONPP_ENABLE_CUDA)
#pragma message("Stream compact on the GPU")
#else
#pragma message("Stream compact on the CPU")
#endif

#if defined(NEWTONPP_GPU_DIRECT)
#pragma message("GPU direct is used for MPI communication")
#else
#pragma message("Data copy to the CPU for MPI communication")
#endif

#if defined(NEWTONPP_USE_OMP_LOOP)
#pragma message("OpenMP offload directive: target teams loop")
#else
#pragma message("OpenMP offload directive: target teams distribute parallel for")
#endif


int main(int argc, char **argv)
{
    auto start_time = timer::now();

    int rank = 0;
    int n_ranks = 1;
    MPI_Comm comm = MPI_COMM_WORLD;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &n_ranks);

    double h = 1e5;                 // time step size
    double nfr = 0.;                // distance for reduced representation
    double eps = 0.;                // the softening length
    double G = 6.67408e-11;         // the gravitational constant
    long n_its = 0;                 // number of solver steps
    long n_bodies = 0;              // number of bodies
    const char *magi_h5 = nullptr;  // where initial positions/velocities can be found
    const char *magi_sum = nullptr; // where particle types can be found
    const char *out_dir = nullptr;  // directory to write results at
    long io_int = 0;                // how often to write resutls
    const char *is_conf = nullptr;  // sensei in situ configuration file
    long is_int = 0;                // how often to invoke in situ processing

    if (parse_command_line(argc, argv, comm, G, h, eps, nfr, n_its,
        n_bodies, magi_h5, magi_sum, out_dir, io_int, is_conf, is_int))
        return -1;

    // load the initial condition and initialize the bodies
    patch_data pd;
    patch_force pf;
    std::vector<patch> patches;

#if defined(NEWTONPP_ENABLE_MAGI)
    if (magi_h5)
    {
        // load the ic positions and velocities
        patch dom;
        if (magi_h5 && read_magi(comm, magi_h5, magi_sum, dom, pd))
            return -1;

        // decompose domain
        patches = partition(dom, n_ranks);
        assign_patches(patches, n_ranks);

        // update partition
        if (n_ranks > 1)
        {
            pf.resize(pd.size());

            hamr::buffer<int> dest(def_alloc());
            partition(comm, patches, pd, dest);
            move(comm, pd, pf, dest);
        }
    }
    else
#endif
    {
        if (initialize_random(comm, n_bodies, patches, pd, h, eps, nfr))
            return -1;
    }

#if defined(NEWTONPP_ENABLE_SENSEI)
    // initialize for in-situ
    insitu_data is_data;
    if (is_conf && is_int && init_insitu(comm, is_conf, is_data))
        return -1;
#endif

    // write the domain decomp
    if (io_int)
        write_vtk(comm, patches, out_dir);

    // flag nearby patches
    std::vector<int> nf;
    near(patches, nfr, nf);

    // initialize forces
    forces(comm, pd, pf, G, eps, nf);

    // write initial state
    if (io_int)
        write_vtk(comm, pd, pf, out_dir);

#if defined(NEWTONPP_ENABLE_SENSEI)
    // process initial state
    if (is_int && is_data && update_insitu(comm, is_data, 0, 0, patches, pd, pf))
        return -1;
#endif

    if (rank == 0)
        std::cerr << " === init " << (timer::now() - start_time) / 1ms << "ms" << std::endl;

    // iterate
    long it = 0;
    while (it < n_its)
    {
        auto it_time  = timer::now();

        // update bodies
        velocity_verlet(comm, pd, pf, G, h, eps, nf);

        // update partition
        //if (n_ranks > 1)
        {
            hamr::buffer<int> dest(def_alloc());
            partition(comm, patches, pd, dest);
            move(comm, pd, pf, dest);
        }

        // write current state
        if (io_int && (((it + 1) % io_int) == 0))
            write_vtk(comm, pd, pf, out_dir);

#if defined(NEWTONPP_ENABLE_SENSEI)
        // process current state
        if (is_int && is_data && update_insitu(comm, is_data, it, it*h, patches, pd, pf))
            return -1;
#endif
        it += 1;

        if (rank == 0)
            std::cerr << " === it " << it << " : " << (timer::now() - it_time) / 1ms << "ms" << std::endl;
    }

#if defined(NEWTONPP_ENABLE_SENSEI)
    // finalize in-situ processing
    if (is_int && is_data && finalize_insitu(comm, is_data))
        return -1;
#endif

    MPI_Finalize();

    if (rank == 0)
        std::cerr << " === total " << (timer::now() - start_time) / 1s << "s" << std::endl;

    return 0;
}
