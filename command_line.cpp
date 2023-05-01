#include "command_line.h"

#include <cstring>
#include <cstdlib>
#include <iostream>

// --------------------------------------------------------------------------
int parse_command_line(int argc, char **argv, MPI_Comm comm,
    double &G, double &dt, double &eps, double &theta,
    long &n_its, long &n_bodies, const char *&magi_file,
    const char *&out_dir, long &io_int, const char *&is_conf,
    long &is_int)
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
                    << "    --magi_file  : MAGI initialization files" << std::endl
                    << "    --out_dir    : where to write the results" << std::endl
                    << "    --out_int    : how often to write results" << std::endl
                    << "    --sensei_xml : a sensei configuration file" << std::endl
                    << "    --sensei_int : how often to invoke in situ" << std::endl
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
                    std::cerr << "the gravitational constant is " << G << std::endl;
            }
            else if (strcmp(argv[q], "--dt") == 0)
            {
                dt = atof(argv[++q]);
                if (rank == 0)
                    std::cerr << "the time step is " << dt << std::endl;
            }
            else if (strcmp(argv[q], "--eps") == 0)
            {
                eps = atof(argv[++q]);
                if (rank == 0)
                    std::cerr << "the softening length is " << eps << std::endl;
            }
            else if (strcmp(argv[q], "--theta") == 0)
            {
                theta = atof(argv[++q]);
                if (rank == 0)
                    std::cerr << "the reduced representaion threshold is " << theta << std::endl;
            }
            else if (strcmp(argv[q], "--n_its") == 0)
            {
                n_its = atol(argv[++q]);
                if (rank == 0)
                    std::cerr << "will run for " << n_its << " iterations" << std::endl;
            }
            else if(strcmp(argv[q], "--n_bodies") == 0)
            {
                n_bodies = atol(argv[++q]);
                if (rank == 0)
                    std::cerr << "will generate " << n_bodies << " bodies total" << std::endl;
            }
            else if(strcmp(argv[q], "--magi_file") == 0)
            {
                magi_file = argv[++q];
                if (rank == 0)
                    std::cerr << "initializing from " << magi_file << std::endl;
            }
            else if(strcmp(argv[q], "--out_dir") == 0)
            {
                out_dir = argv[++q];
                if (rank == 0)
                    std::cerr << "writing results at " << out_dir << std::endl;
            }
            else if(strcmp(argv[q], "--out_int") == 0)
            {
                io_int = atol(argv[++q]);
                if (rank == 0)
                    std::cerr << "writing results every " << io_int
                        << " iterations" << std::endl;
            }
            else if(strcmp(argv[q], "--sensei_xml") == 0)
            {
                is_conf = argv[++q];
                if (rank == 0)
                    std::cerr << "in-situ initialized with " << is_conf << std::endl;
            }
            else if(strcmp(argv[q], "--sensei_int") == 0)
            {
                is_int = atol(argv[++q]);
                if (rank == 0)
                    std::cerr << "in-situ processing every "
                        << is_int << " iterations" << std::endl;
            }
        }
        else
        {
            if (rank == 0)
                std::cerr << "unknown argument " << argv[q]
                    << " at position " << q << std::endl;
            return -1;
        }

        q += 1;
    }

    return 0;
}
