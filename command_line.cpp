#include "command_line.h"

#include <cstring>
#include <cstdlib>
#include <iostream>

// --------------------------------------------------------------------------
int parse_command_line(int argc, char **argv, MPI_Comm comm,
    int &num_devs, int &start_dev, int &dev_stride,
    double &G, double &dt, double &eps, double &theta,
    long &n_its, long &n_bodies, long &part_int, const char *&magi_h5,
    const char *&magi_sum, const char *&out_dir, long &io_int,
    const char *&is_conf, long &is_int)
{
    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    int q = 1;
    while (q < argc)
    {
        if (strcmp(argv[q], "--help") == 0)
        {
            if (rank == 0)
            {
                std::cerr << "newtonpp command line arguments:" << std::endl
                    << std::endl
                    << "    --G          : gravitational constant" << std::endl
                    << "    --dt         : time step size" << std::endl
                    << "    --eps        : softening length" << std::endl
                    << "    --theta      : threshold for reduced representation" << std::endl
                    << "    --n_its      : how many iterations to perform" << std::endl
                    << "    --n_bodies   : the total number of bodies" << std::endl
                    << "    --part_int   : how often to repartition particles" << std::endl
                    << "    --magi_h5    : MAGI file with particle positions" << std::endl
                    << "    --magi_sum   : MAGI file with component sizes" << std::endl
                    << "    --out_dir    : where to write the results" << std::endl
                    << "    --out_int    : how often to write results" << std::endl
                    << "    --sensei_xml : a sensei configuration file" << std::endl
                    << "    --sensei_int : how often to invoke in situ" << std::endl
                    << "    --num_devs   : how many devices to use per node" << std::endl
                    << "    --start_dev  : the first device to use" << std::endl
                    << "    --dev_stride : the number of devices to skip" << std::endl
                    << std::endl
                    << std::endl;
            }
            return -1;
        }
        else if ((q + 1 < argc))
        {
            if (strcmp(argv[q], "--G") == 0)
            {
                G = atof(argv[++q]);
                if (rank == 0)
                    std::cerr << " === newton++ === : the gravitational constant is " << G << std::endl;
            }
            else if (strcmp(argv[q], "--dt") == 0)
            {
                dt = atof(argv[++q]);
                if (rank == 0)
                    std::cerr << " === newton++ === : the time step is " << dt << std::endl;
            }
            else if (strcmp(argv[q], "--eps") == 0)
            {
                eps = atof(argv[++q]);
                if (rank == 0)
                    std::cerr << " === newton++ === : the softening length is " << eps << std::endl;
            }
            else if (strcmp(argv[q], "--theta") == 0)
            {
                theta = atof(argv[++q]);
                if (rank == 0)
                    std::cerr << " === newton++ === : the reduced representaion threshold is " << theta << std::endl;
            }
            else if (strcmp(argv[q], "--n_its") == 0)
            {
                n_its = atol(argv[++q]);
                if (rank == 0)
                    std::cerr << " === newton++ === : run for " << n_its << " iterations" << std::endl;
            }
            else if(strcmp(argv[q], "--n_bodies") == 0)
            {
                n_bodies = atol(argv[++q]);
                if (rank == 0)
                    std::cerr << " === newton++ === : generate " << n_bodies << " bodies total" << std::endl;
            }
            else if(strcmp(argv[q], "--part_int") == 0)
            {
                part_int = atol(argv[++q]);
                if (rank == 0)
                    std::cerr << " === newton++ === : partition particles every " << part_int << " iterations" << std::endl;
            }
            else if(strcmp(argv[q], "--magi_h5") == 0)
            {
                magi_h5 = argv[++q];
                if (rank == 0)
#if defined(NEWTONPP_ENABLE_MAGI)
                    std::cerr << "i === newton++ === : intializing postions from " << magi_h5 << std::endl;
#else
                    std::cerr << "Error: hdf5 is required for magi initial conditions" << std::endl;
                abort();
#endif
            }
            else if(strcmp(argv[q], "--magi_sum") == 0)
            {
                magi_sum = argv[++q];
                if (rank == 0)
#if defined(NEWTONPP_ENABLE_MAGI)
                    std::cerr << " === newton++ === : initializing components from " << magi_sum << std::endl;
#else
                    std::cerr << "Error: magi components disabled" << std::endl;
                abort();
#endif
            }
            else if(strcmp(argv[q], "--out_dir") == 0)
            {
                out_dir = argv[++q];
                if (rank == 0)
                    std::cerr << " === newton++ === : writing results at " << out_dir << std::endl;
            }
            else if(strcmp(argv[q], "--out_int") == 0)
            {
                io_int = atol(argv[++q]);
                if (rank == 0)
                    std::cerr << " === newton++ === : writing results every " << io_int
                        << " iterations" << std::endl;
            }
            else if(strcmp(argv[q], "--sensei_xml") == 0)
            {
                is_conf = argv[++q];
                if (rank == 0)
                    std::cerr << " === newton++ === : in-situ initialized with " << is_conf << std::endl;
            }
            else if(strcmp(argv[q], "--sensei_int") == 0)
            {
                is_int = atol(argv[++q]);
                if (rank == 0)
                    std::cerr << " === newton++ === : in-situ processing every " << is_int << " iterations" << std::endl;
            }
            else if(strcmp(argv[q], "--num_devs") == 0)
            {
                num_devs = atol(argv[++q]);
                if (rank == 0)
                    std::cerr << " === newton++ === : " << num_devs << " devices in use per node" << std::endl;
            }
            else if(strcmp(argv[q], "--start_dev") == 0)
            {
                start_dev = atol(argv[++q]);
                if (rank == 0)
                    std::cerr << " === newton++ === : " << start_dev << " is the first device" << std::endl;
            }
            else if(strcmp(argv[q], "--dev_stride") == 0)
            {
                dev_stride = atol(argv[++q]);
                if (rank == 0)
                    std::cerr << " === newton++ === : skip " << dev_stride << " devices" << std::endl;
            }
            else
            {
                if (rank == 0)
                    std::cerr << "Error: unknown argument " << argv[q]
                        << " at position " << q << std::endl;
                return -1;
            }
        }
        else
        {
            if (rank == 0)
                std::cerr << "Error: unknown argument " << argv[q]
                    << " at position " << q << std::endl;
            return -1;
        }

        q += 1;
    }

    return 0;
}
