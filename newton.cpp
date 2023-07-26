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
#include "timer_stack.h"
#if defined(NEWTONPP_ENABLE_SENSEI)
#include "insitu.h"
#endif

#include <iostream>
#include <vector>
#include <mpi.h>
#if defined(_OPENMP)
#include <omp.h>
#endif

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
    int rank = 0;
    int n_ranks = 1;
    MPI_Comm comm = MPI_COMM_WORLD;

    int thread_level = 0;
    if ((MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &thread_level) != MPI_SUCCESS) ||
        (thread_level != MPI_THREAD_MULTIPLE))
    {
        std::cerr << "Error: failed to initialize MPI" << std::endl;
        return -1;
    }

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &n_ranks);

    timer_stack timer(rank == 0);
    timer.push();
    timer.push();

    double h = 1e5;                 // time step size
    double nfr = 0.;                // distance for reduced representation
    double eps = 0.;                // the softening length
    double G = 6.67408e-11;         // the gravitational constant
    long n_its = 0;                 // number of solver steps
    long n_bodies = 0;              // number of bodies
    long part_int = 4;              // how often to partition particles
    const char *magi_h5 = nullptr;  // where initial positions/velocities can be found
    const char *magi_sum = nullptr; // where particle types can be found
    const char *out_dir = nullptr;  // directory to write results at
    long io_int = 0;                // how often to write resutls
    const char *is_conf = nullptr;  // sensei in situ configuration file
    long is_int = 0;                // how often to invoke in situ processing
#if defined(_OPENMP)
    int num_devs = omp_get_num_devices();
#else
    int num_devs = 1;
#endif
    int start_dev = 0;
    int dev_stride = 1;

    if (parse_command_line(argc, argv, comm, num_devs, start_dev, dev_stride,
        G, h, eps, nfr, n_its, n_bodies, part_int, magi_h5, magi_sum, out_dir,
        io_int, is_conf, is_int))
        return -1;

#if defined(_OPENMP)
    // set the device to use
    int dev = rank % num_devs * dev_stride + start_dev;
    omp_set_default_device(dev);
#endif

    // load the initial condition and initialize the bodies
    patch_data pd;
    patch_force pf;
    std::vector<patch> patches;

    timer.push();
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
            move2(comm, pd, pf, dest);
        }
    }
    else
#endif
    {
        if (initialize_random(comm, n_bodies, patches, pd, h, eps, nfr))
            return -1;
    }
    timer.pop("read ic");

#if defined(NEWTONPP_ENABLE_SENSEI)
    // initialize for in-situ
    insitu_data is_data;
    if (is_conf && is_int)
    {
        timer.push();
        if (init_insitu(comm, is_conf, is_data))
            return -1;
        timer.pop("sensei init");
    }
#endif

    // write the domain decomp
    if (io_int)
    {
        timer.push();
        write_vtk(comm, patches, out_dir);
        timer.pop("write dom");
    }

    // flag nearby patches
    timer.push();
    std::vector<int> nf;
    near(patches, nfr, nf);
    timer.pop("build tree");

    // initialize forces
    timer.push();
    forces(comm, pd, pf, G, eps, nf);
    timer.pop("init forces");

    // write initial state
    if (io_int)
    {
        timer.push();
        write_vtk(comm, pd, pf, out_dir);
        timer.pop("write part");
    }

#if defined(NEWTONPP_ENABLE_SENSEI)
    // process initial state
    if (is_int && is_data)
    {
        timer.push();
        if (update_insitu(comm, is_data, 0, 0, patches, pd, pf))
            return -1;
        timer.pop("sensei upd");
    }
#endif
    timer.pop("=== initialization ===");

    // iterate
    long it = 0;
    while (it < n_its)
    {
#if defined(_OPENMP)
        omp_set_default_device(dev);
#endif
        // update bodies
        timer.push();
        timer.push();
        velocity_verlet(comm, pd, pf, G, h, eps, nf);
        timer.pop("integrate part");

        // update partition
        if ((it % part_int) == 0)
        {
            timer.push();
            hamr::buffer<int> dest(def_alloc());
            partition(comm, patches, pd, dest);
            timer.pop_push("partition part");
            move2(comm, pd, pf, dest);
            timer.pop("move part");
        }

        // write current state
        if (io_int && (((it + 1) % io_int) == 0))
        {
            timer.push();
            write_vtk(comm, pd, pf, out_dir);
            timer.pop("write part");
        }

#if defined(NEWTONPP_ENABLE_SENSEI)
        // process current state
        if (is_int && is_data)
        {
            timer.push();
            if (update_insitu(comm, is_data, it, it*h, patches, pd, pf))
                return -1;
            timer.pop("sensei upd");
        }
#endif
        it += 1;
        timer.pop("=== loop iteration ===");
    }

#if defined(NEWTONPP_ENABLE_SENSEI)
    // finalize in-situ processing
    timer.push();
    if (is_int && is_data && finalize_insitu(comm, is_data))
        return -1;
    timer.pop("fin sensei");
#endif

    MPI_Finalize();

    return 0;
}
