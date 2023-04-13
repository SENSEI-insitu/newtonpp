#include "initialize_file.h"

#include <string>
#include <unistd.h>
#include <iostream>
#include <vector>
#include <mpi.h>
#include <cstring>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>

// --------------------------------------------------------------------------
template <typename T>
int readn(int fh, T *buf, size_t n)
{
    ssize_t ierr = 0;
    ssize_t nb = n*sizeof(T);
    if ((ierr = read(fh, buf, nb)) != nb)
    {
        std::cerr << "Failed to read " << n << " elements of size "
            << sizeof(T) << std::endl << strerror(errno) << std::endl;
        return -1;
    }
    return 0;
}


// --------------------------------------------------------------------------
int initialize_file(MPI_Comm comm, const std::string &idir,
    std::vector<patch> &patches, patch_data &lpd,
    double &h, double &eps, double &nfr)
{
    int rank = 0;
    int n_ranks = 1;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &n_ranks);

    // read the set of patches
    long n_patches = 0;
    std::vector<double> tpatches;
    if (rank == 0)
    {
        // rank 0 reads and sends to all others
        std::string fn = idir + "/patches.npp";
        int h = open(fn.c_str(), O_RDONLY);
        if (h < 0)
        {
            std::cerr << "Failed to open \"" << fn << "\"" << std::endl
                << strerror(errno) << std::endl;
            return -1;
        }

        if (readn(h, &n_patches, 1 ))
        {
            std::cerr << "Failed to read \"" << fn << "\"" << std::endl;
            close(h);
            return -1;
        }

        long n_elem = 6*n_patches;
        tpatches.resize(n_elem);
        if (readn(h, tpatches.data(), n_elem))
        {
            std::cerr << "Failed to read \"" << fn << "\"" << std::endl;
            close(h);
            return -1;
        }

        close(h);

        MPI_Bcast(&n_patches, 1, MPI_LONG, 0, comm);
        MPI_Bcast(tpatches.data(), n_elem, MPI_DOUBLE, 0, comm);
    }
    else
    {
        // receive from rank 0
        MPI_Bcast(&n_patches, 1, MPI_LONG, 0, comm);

        long n_elem = 6*n_patches;
        tpatches.resize(n_elem);
        MPI_Bcast(tpatches.data(), n_elem, MPI_DOUBLE, 0, comm);
    }

    // convert to patch structures
    patches.resize(n_ranks);
    for (int i = 0; i < n_ranks; ++i)
    {
        const double *pp = &tpatches[6*i];
        patches[i] = patch(i, pp[0], pp[1], pp[2], pp[3], pp[4], pp[5]);
    }

    // check that number of ranks and patches match
    if (n_patches != n_ranks)
    {
        std::cerr << "Wrong number of patches " << n_patches
            << " for " << n_ranks << " ranks" << std::endl;
        return -1;
    }

    // read the local patch data
    {
    std::string fn = idir + "/patch_data_" + std::to_string(rank) + ".npp";
    int fh = open(fn.c_str(), O_RDONLY);
    if (fh < 0)
    {
        std::cerr << "Failed to open \"" << fn << "\"" << std::endl
            << strerror(errno) << std::endl;
        return -1;
    }

    long nbod = 0;
    if (readn(fh, &nbod, 1))
    {
        std::cerr << "Failed to read \"" << fn << "\"" << std::endl;
        close(fh);
        return -1;
    }

    if (nbod)
    {
        hamr::buffer<double> tm(cpu_alloc(), nbod);
        hamr::buffer<double> tx(cpu_alloc(), nbod);
        hamr::buffer<double> ty(cpu_alloc(), nbod);
        hamr::buffer<double> tz(cpu_alloc(), nbod);
        hamr::buffer<double> tu(cpu_alloc(), nbod);
        hamr::buffer<double> tv(cpu_alloc(), nbod);
        hamr::buffer<double> tw(cpu_alloc(), nbod);

        if (readn(fh, tm.data(), nbod) ||
            readn(fh, tx.data(), nbod) || readn(fh, ty.data(), nbod) || readn(fh, tz.data(), nbod) ||
            readn(fh, tu.data(), nbod) || readn(fh, tv.data(), nbod) || readn(fh, tw.data(), nbod))
        {
            std::cerr << "Failed to read \"" << fn << "\"" << std::endl;
            close(fh);
            return -1;
        }

        lpd.m_m = std::move(tm);
        lpd.m_x = std::move(tx);
        lpd.m_y = std::move(ty);
        lpd.m_z = std::move(tz);
        lpd.m_u = std::move(tu);
        lpd.m_v = std::move(tv);
        lpd.m_w = std::move(tw);
    }
    }

    // read parameters
    if (rank == 0)
    {
        std::string fn = idir + "/params.npp";
        FILE *fh = nullptr;

        if ((fh = fopen(fn.c_str(),"r")) == nullptr)
        {
            std::cerr << "Failed to open \"" << fn << "\"" << std::endl
                << strerror(errno) << std::endl;
        }

       if (fscanf(fh, "h = %lf, eps = %lf, nfr = %lf", &h, &eps, &nfr) != 3)
       {
           fclose(fh);
           std::cerr << "Failed to read h, eps, nfr from file " << fn << std::endl;
           return -1;
       }

       fclose(fh);
    }

    MPI_Bcast(&h, 1, MPI_DOUBLE, 0, comm);
    MPI_Bcast(&eps, 1, MPI_DOUBLE, 0, comm);
    MPI_Bcast(&nfr, 1, MPI_DOUBLE, 0, comm);

   return 0;
}

// --------------------------------------------------------------------------
int initialize_file(int argc, char **argv,
    MPI_Comm comm,
    std::vector<patch> &patches, patch_data &lpd,
    double &h, double &eps, double &nfr,
    const char *&odir, long &nits, long &io_int)
{
    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    // parse the command line
    if (argc != 5)
    {
        if (rank == 0)
            std::cerr << "usage:" << std::endl
                << "newtonpp [in dir] [out dir] [n its] [io int]" << std::endl;
        return -1;
    }

    int q = 0;
    const char *idir = argv[++q];
    odir = argv[++q];
    nits = atoi(argv[++q]);
    io_int = atoi(argv[++q]);

    // initialize
    if (initialize_file(comm, idir, patches, lpd, h, eps, nfr))
        return -1;

    long lnb = lpd.size();
    long tnb = 0;
    MPI_Reduce(&lnb, &tnb, 1, MPI_LONG, MPI_SUM, 0, comm);

    if (rank == 0)
    {
        std::cerr << "initialized " << tnb << " bodies on " << patches.size()
            << " patches. h=" << h << " eps=" << eps << " nfr=" << nfr << std::endl;
    }

    return 0;
}
