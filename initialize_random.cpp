#include "initialize_random.h"

#include "domain_decomp.h"

#include <random>
#include <vector>
#include <iostream>

// --------------------------------------------------------------------------
void initialize_random(MPI_Comm comm, long n, const patch &dom,
    const patch &p, double m0, double m1, double v0, double v1,
    patch_data &pd)
{
    const double *d_x = dom.m_x.data();
    const double *p_x = p.m_x.data();

    #pragma omp target map(tofrom:n), is_device_ptr(d_x,p_x)
    {
    // the fraction of total area covered by this patch
    double afrac = ((p_x[1] - p_x[0]) * (p_x[3] - p_x[2]) * (p_x[5] - p_x[4])) /
        ((d_x[1] - d_x[0]) * (d_x[3] - d_x[2]) * (d_x[5] - d_x[4]));

    // adjust the number of particles
    n *= afrac;
    n = (n < 1 ? 1 : n);
    }

    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    std::mt19937 gen(rank);
    std::uniform_real_distribution<double> dist(0.,1.);

    std::vector<double> rm(n);
    std::vector<double> rx(n);
    std::vector<double> ry(n);
    std::vector<double> rz(n);
    std::vector<double> rv(n);

    double *prm = rm.data();
    double *prx = rx.data();
    double *pry = ry.data();
    double *prz = rz.data();
    double *prv = rv.data();

    for (long i = 0; i < n; ++i)
    {
        prm[i] = std::max(0., std::min(1., dist(gen)));
        prx[i] = std::max(0., std::min(1., dist(gen)));
        pry[i] = std::max(0., std::min(1., dist(gen)));
        prz[i] = std::max(0., std::min(1., dist(gen)));
        prv[i] = std::max(0., std::min(1., dist(gen)));
    }

    pd.resize(n);

#if defined(USE_STRUCTURED_BINDINGS)
    auto [pd_m, pd_x, pd_y, pd_z, pd_u, pd_v, pd_w] = pd.get_data();
#else
    double *pd_m = pd.m_m.data();
    double *pd_x = pd.m_x.data();
    double *pd_y = pd.m_y.data();
    double *pd_z = pd.m_z.data();
    double *pd_u = pd.m_u.data();
    double *pd_v = pd.m_v.data();
    double *pd_w = pd.m_w.data();
#endif

#if defined(NEWTONPP_USE_OMP_LOOP)
    #pragma omp target teams loop is_device_ptr(p_x,pd_m,pd_x,pd_y,pd_z,pd_u,pd_v,pd_w), map(to:prm[0:n],prx[0:n],pry[0:n],prz[0:n],prv[0:n])
#else
    #pragma omp target teams distribute parallel for is_device_ptr(p_x,pd_m,pd_x,pd_y,pd_z,pd_u,pd_v,pd_w), map(to:prm[0:n],prx[0:n],pry[0:n],prz[0:n],prv[0:n])
#endif
    for (long i = 0; i < n; ++i)
    {
        double dm = m1 - m0;
        double dx = p_x[1] - p_x[0];
        double dy = p_x[3] - p_x[2];
        double dz = p_x[5] - p_x[4];
        double dv = v1 - v0;

        double m = dm * prm[i] + m0;
        double x = dx * prx[i] + p_x[0];
        double y = dy * pry[i] + p_x[2];
        double z = dz * prz[i] + p_x[4];
        double v = dv * prv[i] + v0;

        pd_x[i] = x;
        pd_y[i] = y;
        pd_z[i] = z;
        pd_m[i] = m;

        double r = sqrt(x*x + y*y + z*z);
        pd_u[i] = v*(-y - .1*x) / r;
        pd_v[i] = v*( x - .1*y) / r;
        pd_w[i] = 0.;
    }

    #pragma omp target is_device_ptr(pd_m,pd_x,pd_y,pd_z,pd_u,pd_v,pd_w)
    if (rank == 0)
    {
        pd_x[0] = 0.;
        pd_y[0] = 0.;
        pd_z[0] = 0.;
        pd_m[0] = 1.989e30;
        pd_u[0] = 0.;
        pd_v[0] = 0.;
        pd_w[0] = 0.;
    }
}

// --------------------------------------------------------------------------
int initialize_random(MPI_Comm comm,
    std::vector<patch> &patches, patch_data &lpd,
    double &nfr, long nb)
{
    int rank = 0;
    int n_ranks = 1;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &n_ranks);

    // initial condition
    double x0 = -5906.4e9;
    double x1 = 5906.4e9;
    double y0 = -5906.4e9;
    double y1 = 5906.4e9;
    double z0 = -5906.4e9;
    double z1 = 5906.4e9;

    double m0 = 10.0e24;
    double m1 = 100.0e24;

    double v0 = 1.0e3;
    double v1 = 10.0e3;

    double dx = x1 - x0;
    double dy = y1 - y0;
    double dz = z1 - z0;
    nfr = 2.*sqrt(dx*dx + dy*dy + dz*dz);

    // partition space
    patch dom(0, x0, x1, y0, y1, z0, z1);
    patches = partition(dom, n_ranks);

    // initialize bodies
    initialize_random(comm, nb, dom, patches[rank], m0, m1, v0, v1, lpd);

    return 0;
}

// --------------------------------------------------------------------------
int initialize_random(MPI_Comm comm, long nb,
    std::vector<patch> &patches, patch_data &lpd,
    double &nfr)
{
    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    // initialize
    if (initialize_random(comm, patches, lpd, nfr, nb))
        return -1;

    long lnb = lpd.size();
    long tnb = 0;
    MPI_Reduce(&lnb, &tnb, 1, MPI_LONG, MPI_SUM, 0, comm);

    if (rank == 0)
    {
        std::cerr << " === newton++ === : initialized " << tnb << " bodies on "
            << patches.size() << " patches. nfr=" << nfr << std::endl;
    }

    return 0;
}
