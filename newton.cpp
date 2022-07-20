#include <unistd.h>
#include <iostream>
#include <vector>
#include <random>
#include <mpi.h>
#include <cassert>
#include <math.h>
#include <deque>
#include <cstring>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>

#include <chrono>
using namespace std::literals;
using timer = std::chrono::high_resolution_clock;

// this activates the cuda optimized stream compaction
#define ENABLE_CUDA

#include "stream_compact.h"
#include "hamr_buffer.h"

hamr::buffer_allocator cpu_alloc = hamr::buffer_allocator::malloc;
hamr::buffer_allocator gpu_alloc = hamr::buffer_allocator::openmp;

#define ENABLE_OMP
#if defined(ENABLE_OMP)
hamr::buffer_allocator def_alloc = gpu_alloc;
#else
hamr::buffer_allocator def_alloc = cpu_alloc;
#endif

struct requests
{
    requests() : m_size(0) {}
    int m_size;            ///< number of requests actually made
    MPI_Request m_req[8];  ///< the requests
    std::vector<std::shared_ptr<const double>> m_data; ///< memory accessed
};


struct patch
{
    ~patch() {}
    patch() : m_owner(-1), m_x(def_alloc, 4, 0.) {}

    patch(int owner, double x0, double x1, double y0, double y1) : m_owner(owner), m_x(def_alloc, 4, {x0, x1, y0, y1}) {}

    patch(const patch &p);
    void operator=(const patch &p);

    int m_owner;
    hamr::buffer<double> m_x;
};

// --------------------------------------------------------------------------
patch::patch(const patch &p) : m_x(p.m_x)
{
    m_owner = p.m_owner;
}

// --------------------------------------------------------------------------
void patch::operator=(const patch &p)
{
    m_owner = p.m_owner;
    m_x.assign(p.m_x);
}




struct patch_data
{
    patch_data() : m_m(def_alloc), m_x(def_alloc),
        m_y(def_alloc), m_u(def_alloc), m_v(def_alloc)
    {
        #ifdef DEBUG
        std::cerr << "patch_data::patch_data " << this << std::endl;
        #endif
    }

    patch_data(const patch_data&) = delete;
    patch_data(patch_data&&) = delete;

    ~patch_data();

    void operator=(const patch_data &pd);
    void operator=(patch_data &&pd);

    long size() const { return m_m.size(); }

    void resize(long n);
    void append(const patch_data &o);

    hamr::buffer<double> m_m; ///< body mass
    hamr::buffer<double> m_x; ///< body position x
    hamr::buffer<double> m_y; ///< body position y
    hamr::buffer<double> m_u; ///< body velocity x
    hamr::buffer<double> m_v; ///< body velocity y
};

// --------------------------------------------------------------------------
patch_data::~patch_data()
{
    #ifdef DEBUG
    std::cerr << "patch_data::~patch_data " << this << std::endl;
    #endif
}

// --------------------------------------------------------------------------
void patch_data::operator=(const patch_data &pd)
{
    #ifdef DEBUG
    std::cerr << "patch_data::operator= " << this << " <-- " << &pd << std::endl;
    #endif

    m_m.assign(pd.m_m);
    m_x.assign(pd.m_x);
    m_y.assign(pd.m_y);
    m_u.assign(pd.m_u);
    m_v.assign(pd.m_v);
}

// --------------------------------------------------------------------------
void patch_data::operator=(patch_data &&pd)
{
    #ifdef DEBUG
    std::cerr << "patch_data::operator= && " << this << " <-- " << &pd << std::endl;
    #endif

    m_m = std::move(pd.m_m);
    m_x = std::move(pd.m_x);
    m_y = std::move(pd.m_y);
    m_u = std::move(pd.m_u);
    m_v = std::move(pd.m_v);
}

// --------------------------------------------------------------------------
void patch_data::resize(long n)
{
    #ifdef DEBUG
    std::cerr << "patch_data::resize " << this << std::endl;
    #endif

    m_m.resize(n);
    m_x.resize(n);
    m_y.resize(n);
    m_u.resize(n);
    m_v.resize(n);
}

// --------------------------------------------------------------------------
void patch_data::append(const patch_data &o)
{
    #ifdef DEBUG
    std::cerr << "patch_data::append " << this << std::endl;
    #endif

    m_m.append(o.m_m);
    m_x.append(o.m_x);
    m_y.append(o.m_y);
    m_u.append(o.m_u);
    m_v.append(o.m_v);
}

// --------------------------------------------------------------------------
void reduce(const patch_data &pdi, patch_data &pdo)
{
    const double *mi = pdi.m_m.data();
    const double *xi = pdi.m_x.data();
    const double *yi = pdi.m_y.data();

    pdo.resize(1);
    double *mo = pdo.m_m.data();
    double *xo = pdo.m_x.data();
    double *yo = pdo.m_y.data();

    long n = pdi.size();
    double m, x, y;
    #pragma omp target enter data map(alloc: m,x,y), map(to: n)

    #pragma omp target map(alloc: m,x,y)
    {
    m = 0.;
    x = 0.;
    y = 0.;
    }

    #pragma omp target teams distribute parallel for reduction(+: m,x,y), is_device_ptr(mi,xi,yi), map(alloc: m,x,y)
    for (long i = 0; i < n; ++i)
    {
        m += mi[i];
        x += mi[i]*xi[i];
        y += mi[i]*yi[i];
    }

    #pragma omp target is_device_ptr(mo,xo,yo), map(alloc: m,x,y)
    {
    mo[0] = m;
    xo[0] = x/m;
    yo[0] = y/m;
    }

    #pragma omp target exit data map(release: m,x,y)
}

// --------------------------------------------------------------------------
void isend_mp(MPI_Comm comm, const patch_data &pd, int dest, int tag, requests &reqs)
{
    static long n;

    reqs.m_size = 1;

    n = pd.size();
    MPI_Isend(&n, 1, MPI_LONG, dest, tag, comm, reqs.m_req);

    if (n)
    {
        reqs.m_size = 4;

        auto spm = pd.m_m.get_cpu_accessible();
        auto spx = pd.m_x.get_cpu_accessible();
        auto spy = pd.m_y.get_cpu_accessible();

        MPI_Request *req = reqs.m_req;

        MPI_Isend(spm.get(), n, MPI_DOUBLE, dest, ++tag, comm, ++req);
        MPI_Isend(spx.get(), n, MPI_DOUBLE, dest, ++tag, comm, ++req);
        MPI_Isend(spy.get(), n, MPI_DOUBLE, dest, ++tag, comm, ++req);

        reqs.m_data = {spm, spx, spy};
    }
}

// --------------------------------------------------------------------------
void recv_mp(MPI_Comm comm, patch_data &pd, int src, int tag)
{
    long n = 0;
    MPI_Recv(&n, 1, MPI_LONG, src, tag, comm, MPI_STATUS_IGNORE);

    pd.resize(n);

    if (n)
    {
        hamr::buffer<double> m(hamr::buffer_allocator::malloc, n);
        hamr::buffer<double> x(hamr::buffer_allocator::malloc, n);
        hamr::buffer<double> y(hamr::buffer_allocator::malloc, n);

        MPI_Recv(m.data(), n, MPI_DOUBLE, src, ++tag, comm, MPI_STATUS_IGNORE);
        MPI_Recv(x.data(), n, MPI_DOUBLE, src, ++tag, comm, MPI_STATUS_IGNORE);
        MPI_Recv(y.data(), n, MPI_DOUBLE, src, ++tag, comm, MPI_STATUS_IGNORE);

        pd.m_m = std::move(m);
        pd.m_x = std::move(x);
        pd.m_y = std::move(y);
    }
}


struct patch_force
{
    patch_force() : m_u(def_alloc), m_v(def_alloc)
    {
        #ifdef DEBUG
        std::cerr << "patch_force::patch_force " << this << std::endl;
        #endif
    }

    patch_force(const patch_force&) = delete;
    patch_force(patch_force&&) = delete;

    ~patch_force();

    void operator=(const patch_force &pd);
    void operator=(patch_force &&pd);

    long size() const { return m_u.size(); }

    void resize(long n);
    void append(const patch_force &o);

    hamr::buffer<double> m_u;   ///< body force x
    hamr::buffer<double> m_v;   ///< body force y
};

// --------------------------------------------------------------------------
patch_force::~patch_force()
{
    #ifdef DEBUG
    std::cerr << "patch_force::~patch_force " << this << std::endl;
    #endif
}

// --------------------------------------------------------------------------
void patch_force::operator=(const patch_force &pd)
{
    #ifdef DEBUG
    std::cerr << "patch_force::operator= " << this << " <-- " << &pd << std::endl;
    #endif

    m_u.assign(pd.m_u);
    m_v.assign(pd.m_v);
}

// --------------------------------------------------------------------------
void patch_force::operator=(patch_force &&pd)
{
    #ifdef DEBUG
    std::cerr << "patch_force::operator=&& " << this << " <-- " << &pd << std::endl;
    #endif

    m_u = std::move(pd.m_u);
    m_v = std::move(pd.m_v);
}

// --------------------------------------------------------------------------
void patch_force::resize(long n)
{
    #ifdef DEBUG
    std::cerr << "patch_force::resize " << this << std::endl;
    #endif

    m_u.resize(n);
    m_v.resize(n);
}

// --------------------------------------------------------------------------
void patch_force::append(const patch_force &o)
{
    #ifdef DEBUG
    std::cerr << "patch_force::append " << this << std::endl;
    #endif

    m_u.append(o.m_u);
    m_v.append(o.m_v);
}









/** Calculates the forces from bodies on this MPI rank. This is written to
 * handle force initialization and should always occur before accumulating
 * remote forces
 */
// --------------------------------------------------------------------------
void forces(const patch_data &pd, patch_force &pf, double eps)
{
    long n = pd.size();

    assert(pf.size() == n);

    double eps2 = eps*eps;

    const double *pd_m = pd.m_m.data();
    const double *pd_x = pd.m_x.data();
    const double *pd_y = pd.m_y.data();

    double *pf_u = pf.m_u.data();
    double *pf_v = pf.m_v.data();

    double fx,fy;
    #pragma omp target enter data map(alloc: fx,fy)

    for (long i = 0; i < n; ++i)
    {
        #pragma omp target map(alloc: fx,fy)
        {
        fx = 0.;
        fy = 0.;
        }

        #pragma omp target teams distribute parallel for reduction(+: fx,fy) is_device_ptr(pd_m,pd_x,pd_y), map(alloc: fx,fy)
        for (long j = 0; j < n; ++j)
        {
            double rx = pd_x[j] - pd_x[i];
            double ry = pd_y[j] - pd_y[i];

            double r2e2 = rx*rx + ry*ry + eps2;
            double r2e23 = r2e2*r2e2*r2e2;

            double G = 6.67408e-11;
            double mf = G*pd_m[i]*pd_m[j] / sqrt(r2e23);

            fx += (i == j ? 0. : rx*mf);
            fy += (i == j ? 0. : ry*mf);
        }

        #pragma omp target is_device_ptr(pf_u,pf_v), map(alloc: fx,fy)
        {
        pf_u[i] = fx;
        pf_v[i] = fy;
        }
    }

    #pragma omp target exit data map(release: fx,fy)
}

/** Accumulates the forces from bodies on another MPI rank.
 */
// --------------------------------------------------------------------------
void forces(const patch_data &lpd, const patch_data &rpd, patch_force &pf, double eps)
{
    long n = lpd.size();
    long m = rpd.size();

    pf.resize(n);

    double eps2 = eps*eps;

    const double *lpd_m = lpd.m_m.data();
    const double *lpd_x = lpd.m_x.data();
    const double *lpd_y = lpd.m_y.data();

    const double *rpd_m = rpd.m_m.data();
    const double *rpd_x = rpd.m_x.data();
    const double *rpd_y = rpd.m_y.data();

    double *pf_u = pf.m_u.data();
    double *pf_v = pf.m_v.data();

    double fx, fy;

    #pragma omp target enter data map(alloc: fx,fy)
    for (long i = 0; i < n; ++i)
    {
        #pragma omp target map(alloc: fx,fy)
        {
        fx = 0.;
        fy = 0.;
        }

        #pragma omp target teams distribute parallel for reduction(+: fx,fy) is_device_ptr(lpd_m,lpd_x,lpd_y, rpd_m,rpd_x,rpd_y), map(alloc: fx,fy)
        for (long j = 0; j < m; ++j)
        {
            double rx = rpd_x[j] - lpd_x[i];
            double ry = rpd_y[j] - lpd_y[i];

            double r2e2 = rx*rx + ry*ry + eps2;
            double r2e23 = r2e2*r2e2*r2e2;

            double G = 6.67408e-11;
            double mf = G*lpd_m[i]*rpd_m[j] / sqrt(r2e23);

            fx += rx*mf;
            fy += ry*mf;
        }

        #pragma omp target is_device_ptr(pf_u,pf_v), map(alloc: fx,fy)
        {
        pf_u[i] = fx;
        pf_v[i] = fy;
        }
    }

    #pragma omp target exit data map(release: fx,fy)
}


// --------------------------------------------------------------------------
void isend(MPI_Comm comm, const patch_force &pf, int dest, int tag)
{
    long n = pf.size();

    auto spf_u = pf.m_u.get_cpu_accessible();
    const double *pf_u = spf_u.get();

    auto spf_v = pf.m_v.get_cpu_accessible();
    const double *pf_v = spf_v.get();

    MPI_Send(&n, 1, MPI_LONG, dest, ++tag, comm);
    MPI_Send(pf_u, n, MPI_DOUBLE, dest, ++tag, comm);
    MPI_Send(pf_v, n, MPI_DOUBLE, dest, ++tag, comm);
}

// --------------------------------------------------------------------------
void recv(MPI_Comm comm, patch_force &pf, int src, int tag)
{
    long n = 0;
    MPI_Recv(&n, 1, MPI_LONG, src, ++tag, comm, MPI_STATUS_IGNORE);

    assert(pf.size() == n);

    hamr::buffer<double> pf_u(hamr::buffer_allocator::malloc, n);
    hamr::buffer<double> pf_v(hamr::buffer_allocator::malloc, n);

    MPI_Recv(pf_u.data(), n, MPI_DOUBLE, src, ++tag, comm, MPI_STATUS_IGNORE);
    MPI_Recv(pf_v.data(), n, MPI_DOUBLE, src, ++tag, comm, MPI_STATUS_IGNORE);

    pf.m_u = std::move(pf_u);
    pf.m_v = std::move(pf_v);
}

// --------------------------------------------------------------------------
void isend(MPI_Comm comm, const patch_data &pd,
    const patch_force &pf, int dest, int tag, requests &reqs)
{
    static long n;

    reqs.m_size = 1;
    n = pd.size();
    MPI_Isend(&n, 1, MPI_LONG, dest, tag, comm, reqs.m_req);

    if (n)
    {
        auto pd_m = pd.m_m.get_cpu_accessible();
        auto pd_x = pd.m_x.get_cpu_accessible();
        auto pd_y = pd.m_y.get_cpu_accessible();
        auto pd_u = pd.m_u.get_cpu_accessible();
        auto pd_v = pd.m_v.get_cpu_accessible();
        auto pf_u = pf.m_u.get_cpu_accessible();
        auto pf_v = pf.m_v.get_cpu_accessible();

        reqs.m_data = {pd_m, pd_x, pd_y, pd_u, pd_v, pf_u, pf_v};

        reqs.m_size = 8;
        MPI_Request *req = reqs.m_req;

        MPI_Isend(pd_m.get(), n, MPI_DOUBLE, dest, ++tag, comm, ++req);
        MPI_Isend(pd_x.get(), n, MPI_DOUBLE, dest, ++tag, comm, ++req);
        MPI_Isend(pd_y.get(), n, MPI_DOUBLE, dest, ++tag, comm, ++req);
        MPI_Isend(pd_u.get(), n, MPI_DOUBLE, dest, ++tag, comm, ++req);
        MPI_Isend(pd_v.get(), n, MPI_DOUBLE, dest, ++tag, comm, ++req);
        MPI_Isend(pf_u.get(), n, MPI_DOUBLE, dest, ++tag, comm, ++req);
        MPI_Isend(pf_v.get(), n, MPI_DOUBLE, dest, ++tag, comm, ++req);
    }
}

// --------------------------------------------------------------------------
void recv(MPI_Comm comm, patch_data &pd, patch_force &pf, int src, int tag)
{
    long n = 0;
    MPI_Recv(&n, 1, MPI_LONG, src, tag, comm, MPI_STATUS_IGNORE);

    pd.resize(0);
    pf.resize(0);

    if (n)
    {
        hamr::buffer<double> tpd_m(cpu_alloc, n);
        hamr::buffer<double> tpd_x(cpu_alloc, n);
        hamr::buffer<double> tpd_y(cpu_alloc, n);
        hamr::buffer<double> tpd_u(cpu_alloc, n);
        hamr::buffer<double> tpd_v(cpu_alloc, n);
        hamr::buffer<double> tpf_u(cpu_alloc, n);
        hamr::buffer<double> tpf_v(cpu_alloc, n);

        MPI_Recv(tpd_m.data(), n, MPI_DOUBLE, src, ++tag, comm, MPI_STATUS_IGNORE);
        MPI_Recv(tpd_x.data(), n, MPI_DOUBLE, src, ++tag, comm, MPI_STATUS_IGNORE);
        MPI_Recv(tpd_y.data(), n, MPI_DOUBLE, src, ++tag, comm, MPI_STATUS_IGNORE);
        MPI_Recv(tpd_u.data(), n, MPI_DOUBLE, src, ++tag, comm, MPI_STATUS_IGNORE);
        MPI_Recv(tpd_v.data(), n, MPI_DOUBLE, src, ++tag, comm, MPI_STATUS_IGNORE);
        MPI_Recv(tpf_u.data(), n, MPI_DOUBLE, src, ++tag, comm, MPI_STATUS_IGNORE);
        MPI_Recv(tpf_v.data(), n, MPI_DOUBLE, src, ++tag, comm, MPI_STATUS_IGNORE);

        pd.m_m = std::move(tpd_m);
        pd.m_x = std::move(tpd_x);
        pd.m_y = std::move(tpd_y);
        pd.m_u = std::move(tpd_u);
        pd.m_v = std::move(tpd_v);
        pf.m_u = std::move(tpf_u);
        pf.m_v = std::move(tpf_v);
    }
}


/** returns a N x N matrix with the i,j set to 1 where distance between
 * patch i and j is less than r and 0 otherwise. this identifies the patch-pairs
 * that need to be fully exchanged to caclulate forces.
 */
// --------------------------------------------------------------------------
void near(const std::vector<patch> &p, double nfr, std::vector<int> &nf)
{
    long n = p.size();
    long nn = n*n;

    nf.resize(nn,1);
    int *pnf = nf.data();

    #pragma omp target data map(from:pnf[0:nn])
    {
    for (long j = 0; j < n; ++j)
    {
        for (long i = 0; i < n; ++i)
        {
            const double *pj_x = p[j].m_x.data();
            const double *pi_x = p[i].m_x.data();

            #pragma omp target is_device_ptr(pj_x,pi_x)
            {
            double cxj = (pj_x[1] + pj_x[0]) / 2.;
            double cyj = (pj_x[3] + pj_x[2]) / 2.;

            double cxi = (pi_x[1] + pi_x[0]) / 2.;
            double cyi = (pi_x[3] + pi_x[2]) / 2.;

            double dx = cxi - cxj;
            double dy = cyi - cyj;

            double r = sqrt(dx*dx + dy*dy);

            pnf[j*n + i] = (r < nfr ? 1 : 0);
            }
        }
    }
    }
}

// --------------------------------------------------------------------------
void forces(MPI_Comm comm, patch_data &pd, patch_force &pf,
    double eps, const std::vector<int> &nf)
{
    int rank = 0;
    int n_ranks = 1;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &n_ranks);

    long n = pd.size();
    pf.resize(n);

    // local forces
    forces(pd, pf, eps);

    // remote forces
    for (int j = 0; j < n_ranks; ++j)
    {
        for (int i = j + 1; i < n_ranks; ++i)
        {
            requests reqs;

            if (rank == i)
            {
                // send data to j (may be reduced representation)
                if (nf[j*n_ranks +i])
                {
                    // near by send all data
                    isend_mp(comm, pd, j, 3000, reqs);
                }
                else
                {
                    // far away send reduced representation
                    patch_data rpd;
                    reduce(pd, rpd);
                    isend_mp(comm, rpd, j, 3000, reqs);
                }

                // receive data from j (may be reduced representation)
                patch_data pdj;
                recv_mp(comm, pdj, j, 4000);

                // calc force on bodies in i from bodies in j
                forces(pd, pdj, pf, eps);
            }
            else if ((rank == j) && (i != j))
            {
                // receive data from i (may be reduced representation)
                patch_data pdi;
                recv_mp(comm, pdi, i, 3000);

                // send data to i (may be reduced representation)
                if (nf[j*n_ranks +i])
                {
                    // near by send all data
                    isend_mp(comm, pd, i, 4000, reqs);
                }
                else
                {
                    // far away send reduced representation
                    patch_data rpd;
                    reduce(pd, rpd);
                    isend_mp(comm, rpd, i, 4000, reqs);
                }

                // calc force on bodies in j from bodies in i
                forces(pd, pdi, pf, eps);
            }

            // wait for sends to complete
            MPI_Waitall(reqs.m_size, reqs.m_req, MPI_STATUS_IGNORE);
        }
    }
}


// --------------------------------------------------------------------------
void append(patch_data &pdo, patch_force &pfo,
    const patch_data &pdi, const patch_force &pfi)
{
    pdo.append(pdi);
    pfo.append(pfi);
}



















/*
// --------------------------------------------------------------------------
void grow(patch &p, double amt)
{
    p.m_x[0] -= amt;
    p.m_x[1] += amt;
    p.m_x[2] -= amt;
    p.m_x[3] += amt;
}

// --------------------------------------------------------------------------
void shrink(patch &p, double amt)
{
    p.m_x[0] += amt;
    p.m_x[1] -= amt;
    p.m_x[2] += amt;
    p.m_x[3] -= amt;
}
*/
/*
// --------------------------------------------------------------------------
bool intersects(const patch &lp, const patch &rp)
{
    double lx = std::max(lp.m_x[0], rp.m_x[0]);
    double hx = std::min(lp.m_x[1], rp.m_x[1]);
    double ly = std::max(lp.m_x[2], rp.m_x[2]);
    double hy = std::min(lp.m_x[3], rp.m_x[3]);
    return (lx <= hx) && (ly <= hy);
}
*/
/*
// --------------------------------------------------------------------------
bool inside(const patch &p, double x, double y)
{
    return (x >= p.m_x[0]) && (x < p.m_x[1]) && (y >= p.m_x[2]) && (y < p.m_x[3]);
}
*/

/** finds the list of patch neighbors and returns them in pn. returns the
 * non-neighbor patches in pnn
// --------------------------------------------------------------------------
void neighbors(const std::vector<patch> &patches,
    double ofs, std::vector<std::vector<int>> &pn, std::vector<std::vector<int>> &pnn)
{
    std::vector<patch> smallp(patches);

    // make patches disjoint by shrinking a small amount
    int n = patches.size();
    for (int i = 0; i < n; ++i)
        shrink(smallp[i], ofs);

    double ofs2 = 2*ofs;

    // a neighbor list for each patch
    pn.resize(n);
    pnn.resize(n);

    for (int i = 0; i < n; ++i)
    {
        // grow active patch by 2x the small amount
        patch actp = patches[i];
        grow(actp, ofs2);

        // see which of the others this patch intersects
        for (int j = 0; j < n; ++j)
        {
            // dont test a patch against itself
            if (i == j) continue;

            const patch &spj = smallp[j];

            if (intersects(actp, spj))
            {
                // j is a neighbor of i
                pn[i].push_back(j);
            }
            else
            {
                pnn[i].push_back(j);
            }
        }
    }
}
 */


#pragma omp declare target
// --------------------------------------------------------------------------
void split(int dir, const double *p0_x, double *p1_x, double *p2_x)
{
    if (dir == 0)
    {
        p1_x[0] = p0_x[0];
        p1_x[1] = (p0_x[0] + p0_x[1]) / 2.;
        p1_x[2] = p0_x[2];
        p1_x[3] = p0_x[3];

        p2_x[0] = p1_x[1];
        p2_x[1] = p0_x[1];
        p2_x[2] = p0_x[2];
        p2_x[3] = p0_x[3];
    }
    else
    {
        p1_x[0] = p0_x[0];
        p1_x[1] = p0_x[1];
        p1_x[2] = p0_x[2];
        p1_x[3] = (p0_x[2] + p0_x[3]) / 2.;

        p2_x[0] = p0_x[0];
        p2_x[1] = p0_x[1];
        p2_x[2] = p1_x[3];
        p2_x[3] = p0_x[3];
    }
}

// --------------------------------------------------------------------------
int inside(const double *p, double x, double y)
{
    return (x >= p[0]) && (x < p[1]) && (y >= p[2]) && (y < p[3]);
}
#pragma omp end declare target

// --------------------------------------------------------------------------
void split(int dir, const patch &p0, patch &p1, patch &p2)
{
    p1.m_owner = p0.m_owner;
    p2.m_owner = p0.m_owner;

    const double *p0_x = p0.m_x.data();
    double *p1_x = p1.m_x.data();
    double *p2_x = p2.m_x.data();

    #pragma omp target is_device_ptr(p0_x,p1_x,p2_x)
    {
    split(dir, p0_x, p1_x, p2_x);
    }
}

// --------------------------------------------------------------------------
void area(const patch &dom, const std::vector<patch> &p, hamr::buffer<double> &area)
{
    long n = p.size();

    area.resize(n);
    double *pa = area.data();

    const double *dom_x = dom.m_x.data();

    // compute the total area
    double atot;
    #pragma omp target enter data map(alloc:atot)
    #pragma omp target map(alloc:atot)
    {
    atot = (dom_x[1] - dom_x[0]) * (dom_x[3] - dom_x[2]);
    }

    // compute fraction of the area covered by the patch
    for (long i = 0; i < n; ++i)
    {
        const patch &pi = p[i];
        const double *pi_x = pi.m_x.data();

        #pragma omp target map(alloc:atot), is_device_ptr(pa,pi_x,dom_x)
        {
        pa[i] = (pi_x[1] - pi_x[0]) * (pi_x[3] - pi_x[2]) / atot;
        }
    }

    #pragma omp target exit data map(release:atot)
}

// --------------------------------------------------------------------------
std::ostream &operator<<(std::ostream &os, const patch &p)
{
    auto spx = p.m_x.get_cpu_accessible();
    const double *px = spx.get();

    os << "{" << p.m_owner << " [" << px[0] << ", " << px[1] << ", "
        << px[2] << ", " << px[3] << "]}";

    return os;
}

// --------------------------------------------------------------------------
template <typename T>
std::ostream &operator<<(std::ostream &os, const std::vector<T> &v)
{
    size_t n = v.size();
    if (n)
    {
        os << v[0];
        for (size_t i = 1; i < n; ++i)
            os << ", " << v[i];
    }
    return os;
}

// --------------------------------------------------------------------------
template <typename T>
std::ostream &operator<<(std::ostream &os, const std::vector<std::vector<T>> &v)
{
    size_t n = v.size();
    for (size_t i = 0; i < n; ++i)
    {
        os << i << " : ";
        size_t m = v[i].size();
        if (m)
        {
            os << v[i][0];
            for (size_t j = 1; j < m; ++j)
                os << ", " << v[i][j];
        }
        os << std::endl;
    }
    return os;
}

// --------------------------------------------------------------------------
std::vector<patch> partition(const patch &dom, size_t n_out)
{
    std::deque<patch> patches;
    patches.push_back(dom);

    int pass = 0;
    while (patches.size() != n_out)
    {
        size_t n = patches.size();
        int dir = pass % 2;
        for (size_t i = 0; i < n; ++i)
        {
            patch p0 = patches.front();
            patches.pop_front();

            patch p1, p2;
            split(dir, p0, p1, p2);

            patches.push_back(p1);
            patches.push_back(p2);

            if (patches.size() == n_out)
                break;
        }
        pass += 1;
    }

    return std::vector<patch>(patches.begin(), patches.end());
}

// --------------------------------------------------------------------------
void assign_patches(std::vector<patch> &p, int n_ranks)
{
    long np = p.size();
    for (long i = 0; i < np; ++i)
    {
        p[i].m_owner = i % n_ranks;
    }
}



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
    double afrac = ((p_x[1] - p_x[0]) * (p_x[3] - p_x[2])) / ((d_x[1] - d_x[0]) * (d_x[3] - d_x[2]));
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
    std::vector<double> rv(n);

    double *prm = rm.data();
    double *prx = rx.data();
    double *pry = ry.data();
    double *prv = rv.data();

    for (long i = 0; i < n; ++i)
    {
        prm[i] = std::max(0., std::min(1., dist(gen)));
        prx[i] = std::max(0., std::min(1., dist(gen)));
        pry[i] = std::max(0., std::min(1., dist(gen)));
        prv[i] = std::max(0., std::min(1., dist(gen)));
    }

    pd.resize(n);
    double *pd_m = pd.m_m.data();
    double *pd_x = pd.m_x.data();
    double *pd_y = pd.m_y.data();
    double *pd_u = pd.m_u.data();
    double *pd_v = pd.m_v.data();

    #pragma omp target teams distribute parallel for is_device_ptr(p_x,pd_m,pd_x,pd_y,pd_u,pd_v), map(to:prm[0:n],prx[0:n],pry[0:n],prv[0:n])
    for (long i = 0; i < n; ++i)
    {
        double dm = m1 - m0;
        double dx = p_x[1] - p_x[0];
        double dy = p_x[3] - p_x[2];
        double dv = v1 - v0;

        double m = dm * prm[i] + m0;
        double x = dx * prx[i] + p_x[0];
        double y = dy * pry[i] + p_x[2];
        double v = dv * prv[i] + v0;

        pd_x[i] = x;
        pd_y[i] = y;
        pd_m[i] = m;

        double r = sqrt(x*x + y*y);
        pd_u[i] = v*(-y - .1*x) / r;
        pd_v[i] = v*( x - .1*y) / r;
    }

    #pragma omp target is_device_ptr(pd_m,pd_x,pd_y,pd_u,pd_v)
    if (rank == 0)
    {
        pd_x[0] = 0.;
        pd_y[0] = 0.;
        pd_m[0] = 1.989e30;
        pd_u[0] = 0.;
        pd_v[0] = 0.;
    }
}

// --------------------------------------------------------------------------
void partition(MPI_Comm comm, const std::vector<patch> &ps,
    const patch_data &pd, hamr::buffer<int> &dest)
{
    int rank = 0;
    int n_ranks = 1;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &n_ranks);

    long n = pd.size();

    // allocate space for the result
    dest.resize(n);
    int *pdest = dest.data();

    // initialize to out of bounds. if a particle is not inside any patch
    // it will be removed from the simulation
    #pragma omp target teams distribute parallel for is_device_ptr(pdest)
    for (long i = 0; i < n; ++i)
    {
        pdest[i] = -1;
    }

    // assign each body a destination
    // work patch by patch. this is inefficient since most of the time a
    // particle will stay in place. however this is far easier to implement on
    // the GPU, than working body by body, testing local patch first, and
    // falling back to neighgbors, and then to the rest.
    const double *pd_x = pd.m_x.data();
    const double *pd_y = pd.m_y.data();

    long nps = ps.size();
    for (long j = 0; j < nps; ++j)
    {
        // get the patch corners
        const double *p_x = ps[j].m_x.data();

        // test each body to see if it's inside this patch
        #pragma omp target teams distribute parallel for is_device_ptr(pd_x,pd_y,p_x,pdest)
        for (long i = 0; i < n; ++i)
        {
            if ((pdest[i] < 0) &&
               (pd_x[i] >= p_x[0]) && (pd_x[i] < p_x[1]) &&
               (pd_y[i] >= p_x[2]) && (pd_y[i] < p_x[3]))
            {
                pdest[i] = j;
            }
        }
    }
}

/* this is the more efficient CPU implementation that works body by body

/// identifes bodies inside the given patch
// --------------------------------------------------------------------------
void inside(const patch &p, const patch_data &pd, std::vector<int> &in)
{
    long n = pd.size();
    in.resize(n);

    for (long i = 0; i < n; ++i)
    {
        in[i] = (inside(p, pd.m_x[i], pd.m_y[i])) ? 1 : 0;
    }
}
// --------------------------------------------------------------------------
void partition(MPI_Comm comm, const patch &dom, const std::vector<patch> &ps,
    const std::vector<std::vector<int>> &pn, const std::vector<std::vector<int>> &pnn,
    const patch_data &pd, std::vector<int> &dest)
{
    int rank = 0;
    int n_ranks = 1;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &n_ranks);

    // assign each body a destination
    long n = pd.size();
    dest.resize(n);
    for (long i = 0; i < n; ++i)
    {
        // find the patch containing this body. in the case that the body lies
        // on a patch boundary the first patch wins.

        if (!inside(dom, pd.m_x[i], pd.m_y[i]))
        {
            // the body has moved outside of the domain. it will be removed from the sim
            dest[i] = -1;
        }
        else if (inside(ps[rank], pd.m_x[i], pd.m_y[i]))
        {
            // most likely the body stays inside the same domain.
            dest[i] = rank;
        }
        else
        {
            // next likely the body moves to the patch neighbors
            const std::vector<int> &nbs = pn[rank];
            int found = 0;
            long nn = nbs.size();
            for (long j = 0; j < nn; ++j)
            {
                if (inside(ps[nbs[j]], pd.m_x[i], pd.m_y[i]))
                {
                    dest[i] = nbs[j];
                    found = 1;
                    break;
                }
            }

            // last check the body moves to non-neighbor patches
            if (!found)
            {
                const std::vector<int> &nnbs = pnn[rank];
                long nnn = nnbs.size();
                for (long j = 0; j < nnn; ++j)
                {
                    if (inside(ps[nnbs[j]], pd.m_x[i], pd.m_y[i]))
                    {
                        dest[i] = nnbs[j];
                        found = 1;
                        break;
                    }
                }
            }

            if (!found)
            {
                // here we have a bug.
                std::cerr << "ERROR: body " << i << " was not inside any patch" << std::endl;
                abort();
            }
        }
    }
}

// CPU version
// --------------------------------------------------------------------------
void stream_compact(const patch_data &pdi, const patch_force &pfi,
    const std::vector<long> &ids, const std::vector<int> &mask,
    patch_data &pdo, patch_force &pfo)
{
    long ni = pdi.size();
    for (long i = 0; i < ni; ++i)
    {
        if (mask[i])
        {
            long q = ids[i];

            pdo.m_x[q] = pdi.m_x[i];
            pdo.m_y[q] = pdi.m_y[i];
            pdo.m_m[q] = pdi.m_m[i];
            pdo.m_u[q] = pdi.m_u[i];
            pdo.m_v[q] = pdi.m_v[i];

            pfo.m_u[q] = pfi.m_u[i];
            pfo.m_v[q] = pfi.m_v[i];
        }
    }
}


// CPU implementation
// --------------------------------------------------------------------------
void package(const patch_data &pdi, const patch_force &pfi,
    const std::vector<int> &dest, int rank, patch_data &pdo, patch_force &pfo)
{
    long ni = pdi.size();

    std::vector<int> mask; // set to 1 if the body should be coppied
    std::vector<long> ids; // the destination index of the copy

    ids.resize(ni);
    mask.resize(ni);

    long no = 0;
    for (long i = 0; i < ni; ++i)
    {
        // exclusive scan
        ids[i] = no;

        // mask
        int m = (dest[i] == rank ? 1 : 0);
        mask[i] = m;

        // sum
        no += m;
    }

    pdo.resize(no);
    pfo.resize(no);

    stream_compact(pdi, pfi, ids, mask, pdo, pfo);
}
    */

// --------------------------------------------------------------------------
void package(const patch_data &pdi, const patch_force &pfi,
    const hamr::buffer<int> &dest, int rank, patch_data &pdo, patch_force &pfo)
{
    long ni = pdi.size();

    hamr::buffer<int> mask(def_alloc, ni, 0);
    int *pmask = mask.data();

    const int *pdest = dest.data();

    // ideintify the bodies that are owned by the specified rank
    #pragma omp target teams distribute parallel for is_device_ptr(pmask,pdest)
    for (long i = 0; i < ni; ++i)
    {
        pmask[i] = (pdest[i] == rank ? 1 : 0);
    }

    // allocate enough space to package all, adjust to actual size later
    pdo.resize(ni);
    pfo.resize(ni);

    // copy them
    int no = 0;

#if defined(ENABLE_CUDA)
    int threads_per_block = 512;
    cuda::stream_compact(pdo.m_m.data(), pdo.m_x.data(), pdo.m_y.data(),
        pdo.m_u.data(), pdo.m_v.data(), pfo.m_u.data(), pfo.m_v.data(), no,
        pdi.m_m.data(), pdi.m_x.data(), pdi.m_y.data(), pdi.m_u.data(),
        pdi.m_v.data(), pfi.m_u.data(), pfi.m_v.data(), pmask, ni,
        threads_per_block);
#else
    cpu::stream_compact(pdo.m_m.data(), pdo.m_x.data(), pdo.m_y.data(),
        pdo.m_u.data(), pdo.m_v.data(), pfo.m_u.data(), pfo.m_v.data(), no,
        pdi.m_m.data(), pdi.m_x.data(), pdi.m_y.data(), pdi.m_u.data(),
        pdi.m_v.data(), pfi.m_u.data(), pfi.m_v.data(), pmask, ni);
#endif

    // adjust size to reflect contents
    pdo.resize(no);
    pfo.resize(no);
}


// --------------------------------------------------------------------------
void move(MPI_Comm comm, patch_data &pd, patch_force &pf, const hamr::buffer<int> &dest)
{
    int rank = 0;
    int n_ranks = 1;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &n_ranks);

    patch_data pdo;
    patch_force pfo;

    // local bodies
    package(pd, pf, dest, rank, pdo, pfo);

    // isend/recv bodies
    for (int j = 0; j < n_ranks; ++j)
    {
        for (int i = j + 1; i < n_ranks; ++i)
        {
            requests reqs;

            if (rank == i)
            {
                // send data to j
                patch_data pdi;
                patch_force pfi;
                package(pd, pf, dest, j, pdi, pfi);
                isend(comm, pdi, pfi, j, 5000, reqs);

                // receive data from j
                patch_data pdj;
                patch_force pfj;
                recv(comm, pdj, pfj, j, 6000);

                // add to the output
                append(pdo, pfo, pdj, pfj);
            }
            else if (rank == j)
            {
                // receive data from i
                patch_data pdi;
                patch_force pfi;
                recv(comm, pdi, pfi, i, 5000);

                // send data to i
                patch_data pdj;
                patch_force pfj;
                package(pd, pf, dest, i, pdj, pfj);
                isend(comm, pdj, pfj, i, 6000, reqs);

                // add to the output
                append(pdo, pfo, pdi, pfi);
            }

            // wait for sends to complete
            MPI_Waitall(reqs.m_size, reqs.m_req, MPI_STATUS_IGNORE);
        }
    }

    pd = std::move(pdo);
    pf = std::move(pfo);
}


/** identifes particles outside the given patch 
// --------------------------------------------------------------------------
void outside(const patch &p, const patch_data &pd, std::vector<long> &oid)
{
    long n = pd.size();
    for (long i = 0; i < n; ++i)
    {
        if (!inside(p, pd.m_x[i], pd.m_y[i]))
        {
            oid.push_back(i);
        }
    }
}*/

/** identifes particles inside the given patch 
// --------------------------------------------------------------------------
void inside(const patch &p, const patch_data &pd, std::vector<long> &iid)
{
    long n = pd.size();
    for (long i = 0; i < n; ++i)
    {
        if (!inside(p, pd.m_x[i], pd.m_y[i]))
        {
            oid.push_back(i);
        }
    }
}*/

/** Velocity Verlet:
 *
 * v_{n+1/2} = v_n + (h/2)*F(x_n)
 * x_{n+1} = x_n + h*v_{n+1/2}
 * v_{n+1} = v_{n+1/2} + (h/2)*F(x_{n+1})
 *
 *  note: patch_forces must be pre-initialized and held in between calls.
 */
// --------------------------------------------------------------------------
void velocity_verlet(MPI_Comm comm,
    patch_data &pd, patch_force &pf, double h, double eps,
    const std::vector<int> &nf)
{
    double h2 = h/2.;
    long n = pd.size();

    double *pd_m = pd.m_m.data();
    double *pd_x = pd.m_x.data();
    double *pd_y = pd.m_y.data();
    double *pd_u = pd.m_u.data();
    double *pd_v = pd.m_v.data();
    double *pf_u = pf.m_u.data();
    double *pf_v = pf.m_v.data();

    #pragma omp target teams distribute parallel for is_device_ptr(pd_m,pd_x,pd_y,pd_u,pd_v,pf_u,pf_v)
    for (long i = 0; i < n; ++i)
    {
        // half step velocity
        pd_u[i] = pd_u[i] + h2 * pf_u[i] / pd_m[i];
        pd_v[i] = pd_v[i] + h2 * pf_v[i] / pd_m[i];

        // full step position
        pd_x[i] = pd_x[i] + h * pd_u[i];
        pd_y[i] = pd_y[i] + h * pd_v[i];
    }

    // update the forces
    forces(comm, pd, pf, eps, nf);

    #pragma omp target teams distribute parallel for is_device_ptr(pd_m,pd_u,pd_v,pf_u,pf_v)
    for (long i = 0; i < n; ++i)
    {
        // half step velocity
        pd_u[i] = pd_u[i] + h2 * pf_u[i] / pd_m[i];
        pd_v[i] = pd_v[i] + h2 * pf_v[i] / pd_m[i];
    }
}

// -------------------------------------------------------------------------
const char *fmt_fname(MPI_Comm comm, const char *dir, const char *name)
{
    static long fid = 0;
    static char fname[1024] = {'\0'};

    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    sprintf(fname, "%s/%s%04d_%06ld.vtk", dir, name, rank, fid);

    fid += 1;

    return fname;
}

// --------------------------------------------------------------------------
void write(MPI_Comm comm, const patch_data &pd, const patch_force &pf, const char *dir)
{
    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    auto spd_m = pd.m_m.get_cpu_accessible();
    auto spd_x = pd.m_x.get_cpu_accessible();
    auto spd_y = pd.m_y.get_cpu_accessible();
    auto spd_u = pd.m_u.get_cpu_accessible();
    auto spd_v = pd.m_v.get_cpu_accessible();
    auto spf_u = pf.m_u.get_cpu_accessible();
    auto spf_v = pf.m_v.get_cpu_accessible();

    const double *pd_m = spd_m.get();
    const double *pd_x = spd_x.get();
    const double *pd_y = spd_y.get();
    const double *pd_u = spd_u.get();
    const double *pd_v = spd_v.get();
    const double *pf_u = spd_u.get();
    const double *pf_v = spd_v.get();

    // package in the vtk layout
    long n = pd.size();

    std::vector<double> x(3*n);
    std::vector<double> m(n);
    std::vector<double> vu(n);
    std::vector<double> vv(n);
    std::vector<double> fu(n);
    std::vector<double> fv(n);
    std::vector<int> r(n);
    std::vector<int> id(2*n);

    for (long i = 0; i < n; ++i)
    {
        long ii = 3*i;
        x[ii    ] = pd_x[i];
        x[ii + 1] = pd_y[i];
        x[ii + 2] = 0.;

        m[i] = pd_m[i];
        vu[i] = pd_u[i];
        vv[i] = pd_v[i];
        fu[i] = pf_u[i];
        fv[i] = pf_v[i];
        r[i] = rank;

        ii = 2*i;
        id[ii    ] = 1;
        id[ii + 1] = i;
    }

    // convert to big endian (required by vtk)
    uint64_t *px = (uint64_t*)x.data();
    for (size_t i = 0; i < x.size(); ++i)
        px[i] = __builtin_bswap64(px[i]);

    uint64_t *pm = (uint64_t*)m.data();
    for (size_t i = 0; i < m.size(); ++i)
        pm[i] = __builtin_bswap64(pm[i]);

    uint64_t *pvu = (uint64_t*)vu.data();
    for (size_t i = 0; i < vu.size(); ++i)
        pvu[i] = __builtin_bswap64(pvu[i]);

    uint64_t *pvv = (uint64_t*)vv.data();
    for (size_t i = 0; i < vv.size(); ++i)
        pvv[i] = __builtin_bswap64(pvv[i]);

    uint64_t *pfu = (uint64_t*)fu.data();
    for (size_t i = 0; i < fu.size(); ++i)
        pfu[i] = __builtin_bswap64(pfu[i]);

    uint64_t *pfv = (uint64_t*)fv.data();
    for (size_t i = 0; i < vv.size(); ++i)
        pfv[i] = __builtin_bswap64(pfv[i]);

    uint32_t *pr = (uint32_t*)r.data();
    for (size_t i = 0; i < r.size(); ++i)
        pr[i] = __builtin_bswap32(pr[i]);

    uint32_t *pid = (uint32_t*)id.data();
    for (size_t i = 0; i < id.size(); ++i)
        pid[i] = __builtin_bswap32(pid[i]);

    // write the file in vtk format
    const char *fn = fmt_fname(comm, dir, "bodies");
    FILE *fh = fopen(fn, "w");
    if (!fh)
    {
        std::cerr << "Error: failed to open \"" << fn << "\"" << std::endl;
        return;
    }


    // write the file in vtk format
    fprintf(fh, "# vtk DataFile Version 2.0\n"
                "newtonpp\n"
                "BINARY\n"
                "DATASET POLYDATA\n"
                "POINTS %ld double\n", n);

    fwrite(x.data(), sizeof(double), x.size(), fh);

    fprintf(fh, "VERTICES %ld %ld\n", n, 2*n);

    fwrite(id.data(), sizeof(int), id.size(), fh);

    fprintf(fh, "POINT_DATA %ld\n"
                "SCALARS rank int 1\n"
                "LOOKUP_TABLE default\n", n);

    fwrite(r.data(), sizeof(int), r.size(), fh);

    fprintf(fh, "SCALARS m double 1\n"
                "LOOKUP_TABLE default\n");

    fwrite(m.data(), sizeof(double), m.size(), fh);

    fprintf(fh, "SCALARS vu double 1\n"
                "LOOKUP_TABLE default\n");

    fwrite(vu.data(), sizeof(double), vu.size(), fh);

    fprintf(fh, "SCALARS vv double 1\n"
                "LOOKUP_TABLE default\n");

    fwrite(vv.data(), sizeof(double), vv.size(), fh);

    fprintf(fh, "SCALARS fu double 1\n"
                "LOOKUP_TABLE default\n");

    fwrite(fu.data(), sizeof(double), fu.size(), fh);

    fprintf(fh, "SCALARS fv double 1\n"
                "LOOKUP_TABLE default\n");

    fwrite(fv.data(), sizeof(double), fv.size(), fh);


    fclose(fh);
}

// --------------------------------------------------------------------------
void write(MPI_Comm comm, const std::vector<patch> &patches, const char *dir)
{
    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    if (rank != 0)
        return;

    // package in the vtk layout
    long n = patches.size();
    std::vector<double> x(12*n);
    std::vector<int> id(5*n);
    std::vector<int> r(n);
    for (long i = 0; i < n; ++i)
    {
        const patch &pi = patches[i];

        auto spi_x = pi.m_x.get_cpu_accessible();
        const double *pi_x = spi_x.get();

        long ii = 12*i;
        x[ii    ] = pi_x[0];
        x[ii + 1] = pi_x[2];
        x[ii + 2] = 0.;

        ii += 3;
        x[ii    ] = pi_x[1];
        x[ii + 1] = pi_x[2];
        x[ii + 2] = 0.;

        ii += 3;
        x[ii    ] = pi_x[1];
        x[ii + 1] = pi_x[3];
        x[ii + 2] = 0.;

        ii += 3;
        x[ii    ] = pi_x[0];
        x[ii + 1] = pi_x[3];
        x[ii + 2] = 0.;

        ii = 5*i;
        long pid = 4*i;
        id[ii    ] = 4;
        id[ii + 1] = pid;
        id[ii + 2] = pid + 1;
        id[ii + 3] = pid + 2;
        id[ii + 4] = pid + 3;

        r[i] = i;
    }

    // convert to big endian (required by vtk)
    uint64_t *px = (uint64_t*)x.data();
    for (long i = 0; i < 12*n; ++i)
        px[i] = __builtin_bswap64(px[i]);

    uint32_t *pid = (uint32_t*)id.data();
    for (long i = 0; i < 5*n; ++i)
        pid[i] = __builtin_bswap32(pid[i]);

    uint32_t *pr = (uint32_t*)r.data();
    for (long i = 0; i < n; ++i)
        pr[i] = __builtin_bswap32(pr[i]);

    // write the file in vtk format
    const char *fn = fmt_fname(comm, dir, "patches");
    FILE *fh = fopen(fn, "w");
    if (!fh)
    {
        std::cerr << "Error: failed to open \"" << fn << "\"" << std::endl;
        return;
    }

    fprintf(fh, "# vtk DataFile Version 2.0\n"
                "newtonpp\n"
                "BINARY\n"
                "DATASET POLYDATA\n"
                "POINTS %ld double\n", 4*n);

    fwrite(x.data(), sizeof(double), x.size(), fh);

    fprintf(fh, "POLYGONS %ld %ld\n", 4*n, 5*n);

    fwrite(id.data(), sizeof(int), id.size(), fh);

    fprintf(fh, "CELL_DATA %ld\n"
                "SCALARS rank int 1\n"
                "LOOKUP_TABLE default\n", n);

    fwrite(r.data(), sizeof(int), r.size(), fh);

    fclose(fh);
}

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

        long nelem = 4*n_patches;
        tpatches.resize(nelem);
        if (readn(h, tpatches.data(), nelem))
        {
            std::cerr << "Failed to read \"" << fn << "\"" << std::endl;
            close(h);
            return -1;
        }

        close(h);

        MPI_Bcast(&n_patches, 1, MPI_LONG, 0, comm);
        MPI_Bcast(tpatches.data(), n_patches, MPI_DOUBLE, 0, comm);
    }
    else
    {
        // receive from rank 0
        MPI_Bcast(&n_patches, 1, MPI_LONG, 0, comm);

        tpatches.resize(4*n_patches);
        MPI_Bcast(tpatches.data(), n_patches, MPI_DOUBLE, 0, comm);
    }

    // convert to patch structures
    patches.resize(n_ranks);
    for (int i = 0; i < n_ranks; ++i)
    {
        const double *pp = &tpatches[4*i];
        patches[i] = patch(i, pp[0], pp[1], pp[2], pp[3]);
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
        hamr::buffer<double> tm(cpu_alloc, nbod);
        hamr::buffer<double> tx(cpu_alloc, nbod);
        hamr::buffer<double> ty(cpu_alloc, nbod);
        hamr::buffer<double> tu(cpu_alloc, nbod);
        hamr::buffer<double> tv(cpu_alloc, nbod);

        if (readn(fh, tm.data(), nbod) ||
            readn(fh, tx.data(), nbod) || readn(fh, ty.data(), nbod) ||
            readn(fh, tu.data(), nbod) || readn(fh, tv.data(), nbod))
        {
            std::cerr << "Failed to read \"" << fn << "\"" << std::endl;
            close(fh);
            return -1;
        }

        lpd.m_m = std::move(tm);
        lpd.m_x = std::move(tx);
        lpd.m_y = std::move(ty);
        lpd.m_u = std::move(tu);
        lpd.m_v = std::move(tv);
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
int initialize_random(MPI_Comm comm,
    std::vector<patch> &patches, patch_data &lpd,
    double &h, double &eps, double &nfr)
{
    int rank = 0;
    int n_ranks = 1;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &n_ranks);

    // initial condition
    long nb = 100;

    double x0 = -5906.4e9;
    double x1 = 5906.4e9;
    double y0 = -5906.4e9;
    double y1 = 5906.4e9;

    double m0 = 10.0e24;
    double m1 = 100.0e24;

    double v0 = 1.0e3;
    double v1 = 10.0e3;

    h = 4.*24.*3600.;
    eps = 0.;

    double dx = x1 - x0;
    double dy = y1 - y0;
    nfr = 2.*sqrt(dx*dx + dy*dy);

    // partition space
    patch dom(0, x0, x1, y0, y1);
    patches = partition(dom, n_ranks);

    // initialize bodies
    initialize_random(comm, nb, dom, patches[rank], m0, m1, v0, v1, lpd);

    return 0;
}

// --------------------------------------------------------------------------
int initialize(MPI_Comm comm, const std::string &idir,
    std::vector<patch> &patches, patch_data &lpd,
    double &h, double &eps, double &nfr)
{
    int rank = 0;
    MPI_Comm_rank(comm, &rank);

#if defined(DEBUG_IC)
    initialize_random(comm, patches, lpd, h, eps, nfr);
#else
    if (initialize_file(comm, idir, patches, lpd, h, eps, nfr))
        return -1;
#endif

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






int main(int argc, char **argv)
{
    auto start_time = timer::now();

    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Init(&argc, &argv);

    int rank = 0;
    int n_ranks = 1;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &n_ranks);

    // parse the command line
    if (argc != 5)
    {
        std::cerr << "usage:" << std::endl
            << "newtonpp [in dir] [out dir] [n its] [io int]" << std::endl;
        return -1;
    }

    int q = 0;
    const char *idir = argv[++q];
    const char *odir = argv[++q];

    long nits = atoi(argv[++q]);
    long io_int = atoi(argv[++q]);

    double h = 0.;   // time step size
    double nfr = 0.; // distance for reduced representation
    double eps = 0.; // softener

    // load the initial condition and initialize the bodies
    patch_data pd;
    std::vector<patch> patches;

    if (initialize(comm, idir, patches, pd, h, eps, nfr))
        return -1;

    // write the domain decomp
    if (io_int)
        write(comm, patches, odir);

    // flag nearby patches
    std::vector<int> nf;
    near(patches, nfr, nf);

    // initialize forces
    patch_force pf;
    forces(comm, pd, pf, eps, nf);

    // write initial state
    if (io_int)
       write(comm, pd, pf, odir);

    if (rank == 0)
        std::cerr << " === init " << (timer::now() - start_time) / 1s << "s" << std::endl;

    // iterate
    long it = 0;
    while (it < nits)
    {
        auto it_time  = timer::now();

        // update bodies
        velocity_verlet(comm, pd, pf, h, eps, nf);

        // update partition
        if (n_ranks > 1)
        {
            hamr::buffer<int> dest(def_alloc);
            partition(comm, patches, pd, dest);
            move(comm, pd, pf, dest);
        }

        // write current state
        if (io_int && (((it + 1) % io_int) == 0))
            write(comm, pd, pf, odir);

        it += 1;

        if (rank == 0)
            std::cerr << " === it " << it << " : " << (timer::now() - it_time) / 1s << "s" << std::endl;

    }

    MPI_Finalize();

    if (rank == 0)
        std::cerr << " === total " << (timer::now() - start_time) / 1s << "s" << std::endl;

    return 0;
}
