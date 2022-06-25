#include <unistd.h>
#include <iostream>
#include <vector>
#include <random>
#include <mpi.h>
#include <cassert>
#include <math.h>
#include <deque>
#include <cstring>

struct patch
{
    ~patch() {}
    patch() : m_owner(-1), m_x{0.,0.,0.,0.} {}
    patch(int owner, double x0, double x1, double y0, double y1) : m_owner(owner), m_x{x0, x1, y0, y1} {}

    patch(const patch &p);
    void operator=(const patch &p);

    int m_owner;
    double m_x[4];
};


// --------------------------------------------------------------------------
patch::patch(const patch &p)
{
    m_owner = p.m_owner;
    m_x[0] = p.m_x[0];
    m_x[1] = p.m_x[1];
    m_x[2] = p.m_x[2];
    m_x[3] = p.m_x[3];
}

// --------------------------------------------------------------------------
void patch::operator=(const patch &p)
{
    m_owner = p.m_owner;
    m_x[0] = p.m_x[0];
    m_x[1] = p.m_x[1];
    m_x[2] = p.m_x[2];
    m_x[3] = p.m_x[3];
}




struct patch_data
{
    patch_data() : m_size(0), m_m(nullptr),
        m_x(nullptr), m_y(nullptr), m_u(nullptr), m_v(nullptr) {}

    patch_data(const patch_data&) = delete;
    patch_data(patch_data&&) = delete;

    ~patch_data();

    void operator=(const patch_data &pd);
    void operator=(patch_data &&pd);

    long size() const { return m_size; }

    void dealloc();
    void alloc(long n);
    void resize(long n);
    void append(const patch_data &o);

    long m_size;

    double *m_m; ///< body mass
    double *m_x; ///< body position x
    double *m_y; ///< body position y
    double *m_u; ///< body velocity x
    double *m_v; ///< body velocity y
};

// --------------------------------------------------------------------------
patch_data::~patch_data()
{
    dealloc();
}

// --------------------------------------------------------------------------
void patch_data::operator=(const patch_data &pd)
{

    if (m_size < pd.m_size)
    {
        dealloc();
        alloc(pd.m_size);
    }
    else
    {
        m_size = pd.m_size;
    }

    long nb = m_size*sizeof(double);

    memcpy(m_m, pd.m_m, nb);
    memcpy(m_x, pd.m_x, nb);
    memcpy(m_y, pd.m_y, nb);
    memcpy(m_u, pd.m_u, nb);
    memcpy(m_v, pd.m_v, nb);
}

// --------------------------------------------------------------------------
void patch_data::operator=(patch_data &&pd)
{
    dealloc();

    m_size = pd.m_size;

    m_m = pd.m_m;
    m_x = pd.m_x;
    m_y = pd.m_y;
    m_u = pd.m_u;
    m_v = pd.m_v;

    pd.m_m = nullptr;
    pd.m_x = nullptr;
    pd.m_y = nullptr;
    pd.m_u = nullptr;
    pd.m_v = nullptr;

    pd.m_size = 0;
}

// --------------------------------------------------------------------------
void patch_data::alloc(long n)
{
    m_size = n;

    size_t nb = n*sizeof(double);

    m_m = (double*)malloc(nb);
    m_x = (double*)malloc(nb);
    m_y = (double*)malloc(nb);
    m_u = (double*)malloc(nb);
    m_v = (double*)malloc(nb);
}


// --------------------------------------------------------------------------
void patch_data::dealloc()
{
    free(m_m);
    free(m_x);
    free(m_y);
    free(m_u);
    free(m_v);

    m_size = 0;

    m_m = nullptr;
    m_x = nullptr;
    m_y = nullptr;
    m_u = nullptr;
    m_v = nullptr;
}

// --------------------------------------------------------------------------
void patch_data::resize(long n)
{
    if (n == 0)
    {
        dealloc();
    }
    else if (n > m_size)
    {
        size_t nbnew = n*sizeof(double);

        double *tmp_m = (double*)malloc(nbnew);
        double *tmp_x = (double*)malloc(nbnew);
        double *tmp_y = (double*)malloc(nbnew);
        double *tmp_u = (double*)malloc(nbnew);
        double *tmp_v = (double*)malloc(nbnew);

        size_t nbold = m_size*sizeof(double);

        memcpy(tmp_m, m_m, nbold);
        memcpy(tmp_x, m_x, nbold);
        memcpy(tmp_y, m_y, nbold);
        memcpy(tmp_u, m_u, nbold);
        memcpy(tmp_v, m_v, nbold);

        dealloc();

        m_m = tmp_m;
        m_x = tmp_x;
        m_y = tmp_y;
        m_u = tmp_u;
        m_v = tmp_v;
    }

    m_size = n;
}

// --------------------------------------------------------------------------
void patch_data::append(const patch_data &o)
{
    long n = size();
    long no = o.size();

    resize(n + no);

    for (long i = 0; i < no; ++i)
    {
        m_m[n + i] = o.m_m[i];
        m_x[n + i] = o.m_x[i];
        m_y[n + i] = o.m_y[i];
        m_u[n + i] = o.m_u[i];
        m_v[n + i] = o.m_v[i];
    }
}





// --------------------------------------------------------------------------
void reduce(const patch_data &pdi, patch_data &pdo)
{
    double m = 0.;
    double x = 0.;
    double y = 0.;

    long n = pdi.size();
    for (long i = 0; i < n; ++i)
    {
        m += pdi.m_m[i];
        x += pdi.m_m[i]*pdi.m_x[i];
        y += pdi.m_m[i]*pdi.m_y[i];
    }

    x /= m;
    y /= m;

    pdo.resize(1);

    pdo.m_m[0] = m;
    pdo.m_x[0] = x;
    pdo.m_y[0] = y;
}

// --------------------------------------------------------------------------
void isend_mp(MPI_Comm comm, const patch_data &pd, int dest, int tag, MPI_Request reqs[4], int &nreq)
{
    static long n;

    nreq = 1;
    n = pd.size();
    MPI_Isend(&n, 1, MPI_LONG, dest, tag, comm, reqs);

    if (n)
    {
        nreq = 4;

        MPI_Isend(pd.m_m, n, MPI_DOUBLE, dest, ++tag, comm, ++reqs);
        MPI_Isend(pd.m_x, n, MPI_DOUBLE, dest, ++tag, comm, ++reqs);
        MPI_Isend(pd.m_y, n, MPI_DOUBLE, dest, ++tag, comm, ++reqs);
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
        MPI_Recv(pd.m_m, n, MPI_DOUBLE, src, ++tag, comm, MPI_STATUS_IGNORE);
        MPI_Recv(pd.m_x, n, MPI_DOUBLE, src, ++tag, comm, MPI_STATUS_IGNORE);
        MPI_Recv(pd.m_y, n, MPI_DOUBLE, src, ++tag, comm, MPI_STATUS_IGNORE);
    }
}


struct patch_force
{
    patch_force() : m_size(0), m_u(nullptr), m_v(nullptr) {}

    patch_force(const patch_force&) = delete;
    patch_force(patch_force&&) = delete;

    ~patch_force();

    void operator=(const patch_force &pd);
    void operator=(patch_force &&pd);

    long size() const { return m_size; }

    void dealloc();
    void alloc(long n);
    void resize(long n);
    void append(const patch_force &o);

    long m_size;

    double *m_u;   ///< body force x
    double *m_v;   ///< body force y
};

// --------------------------------------------------------------------------
patch_force::~patch_force()
{
    dealloc();
}

// --------------------------------------------------------------------------
void patch_force::operator=(const patch_force &pd)
{

    if (m_size < pd.m_size)
    {
        dealloc();
        alloc(pd.m_size);
    }
    else
    {
        m_size = pd.m_size;
    }

    long nb = m_size*sizeof(double);

    memcpy(m_u, pd.m_u, nb);
    memcpy(m_v, pd.m_v, nb);
}

// --------------------------------------------------------------------------
void patch_force::operator=(patch_force &&pd)
{
    dealloc();

    m_size = pd.m_size;

    m_u = pd.m_u;
    m_v = pd.m_v;

    pd.m_u = nullptr;
    pd.m_v = nullptr;

    pd.m_size = 0;
}

// --------------------------------------------------------------------------
void patch_force::alloc(long n)
{
    m_size = n;

    size_t nb = n*sizeof(double);

    m_u = (double*)malloc(nb);
    m_v = (double*)malloc(nb);
}


// --------------------------------------------------------------------------
void patch_force::dealloc()
{
    free(m_u);
    free(m_v);

    m_size = 0;

    m_u = nullptr;
    m_v = nullptr;
}

// --------------------------------------------------------------------------
void patch_force::resize(long n)
{
    if (n == 0)
    {
        dealloc();
    }
    else if (n > m_size)
    {
        size_t nbnew = n*sizeof(double);

        double *tmp_u = (double*)malloc(nbnew);
        double *tmp_v = (double*)malloc(nbnew);

        size_t nbold = m_size*sizeof(double);

        memcpy(tmp_u, m_u, nbold);
        memcpy(tmp_v, m_v, nbold);

        dealloc();

        m_u = tmp_u;
        m_v = tmp_v;
    }

    m_size = n;
}

// --------------------------------------------------------------------------
void patch_force::append(const patch_force &o)
{
    long n = size();
    long no = o.size();

    resize(n + no);

    for (long i = 0; i < no; ++i)
    {
        m_u[n + i] = o.m_u[i];
        m_v[n + i] = o.m_v[i];
    }
}









// --------------------------------------------------------------------------
void copy(const patch_data &pdi, const patch_force &pfi, const std::vector<long> &ids, const std::vector<int> &mask, patch_data &pdo, patch_force &pfo)
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


/** Calculates the forces from bodies on this MPI rank. This is written to
 * handle force initialization and should always occur before accumulating
 * remote forces
 */
// --------------------------------------------------------------------------
void forces(const patch_data &pd, patch_force &pf, double eps)
{
    long n = pd.size();

    assert(pf.size() == n);

    for (long i = 0; i < n; ++i)
    {
        double fx = 0.0;
        double fy = 0.0;

        for (long j = 0; j < n; ++j)
        {
            double mi = pd.m_m[i];
            double xi = pd.m_x[i];
            double yi = pd.m_y[i];

            double mj = pd.m_m[j];
            double rx = pd.m_x[j] - xi;
            double ry = pd.m_y[j] - yi;

            double r2e2 = rx*rx + ry*ry + eps*eps;
            double r2e23 = r2e2*r2e2*r2e2;

            double G = 6.67408e-11;
            double mf = G*mi*mj / sqrt(r2e23);

            fx += (i == j ? 0.0 : rx*mf);
            fy += (i == j ? 0.0 : ry*mf);
        }

        pf.m_u[i] = fx;
        pf.m_v[i] = fy;
    }
}

/** Accumulates the forces from bodies on another MPI rank.
 */
// --------------------------------------------------------------------------
void forces(const patch_data &lpd, const patch_data &rpd, patch_force &pf, double eps)
{
    long n = lpd.size();
    long m = rpd.size();

    assert(pf.size() == n);

    for (long i = 0; i < n; ++i)
    {
        double mi = lpd.m_m[i];
        double xi = lpd.m_x[i];
        double yi = lpd.m_y[i];

        double fx = 0.0;
        double fy = 0.0;

        for (long j = 0; j < m; ++j)
        {

            double mj = rpd.m_m[j];
            double rx = rpd.m_x[j] - xi;
            double ry = rpd.m_y[j] - yi;

            double r2e2 = rx*rx + ry*ry + eps*eps;
            double r2e23 = r2e2*r2e2*r2e2;

            double G = 6.67408e-11;
            double mf = G*mi*mj / sqrt(r2e23);

            fx += rx*mf;
            fy += ry*mf;
        }

        pf.m_u[i] = fx;
        pf.m_v[i] = fy;
    }
}


// --------------------------------------------------------------------------
void isend(MPI_Comm comm, const patch_force &pf, int dest, int tag)
{
    long n = pf.size();
    MPI_Send(&n, 1, MPI_LONG, dest, ++tag, comm);
    MPI_Send(pf.m_u, n, MPI_DOUBLE, dest, ++tag, comm);
    MPI_Send(pf.m_v, n, MPI_DOUBLE, dest, ++tag, comm);
}

// --------------------------------------------------------------------------
void recv(MPI_Comm comm, patch_force &pf, int src, int tag)
{
    long n = 0;
    MPI_Recv(&n, 1, MPI_LONG, src, ++tag, comm, MPI_STATUS_IGNORE);

    assert(pf.size() == n);

    MPI_Recv(pf.m_u, n, MPI_DOUBLE, src, ++tag, comm, MPI_STATUS_IGNORE);
    MPI_Recv(pf.m_v, n, MPI_DOUBLE, src, ++tag, comm, MPI_STATUS_IGNORE);
}

// --------------------------------------------------------------------------
void isend(MPI_Comm comm, const patch_data &pd, const patch_force &pf, int dest, int tag, MPI_Request reqs[8], int &nreqs)
{
    static long n;

    nreqs = 1;
    n = pd.size();
    MPI_Isend(&n, 1, MPI_LONG, dest, tag, comm, reqs);

    if (n)
    {
        nreqs = 8;

        MPI_Isend(pd.m_m, n, MPI_DOUBLE, dest, ++tag, comm, ++reqs);
        MPI_Isend(pd.m_x, n, MPI_DOUBLE, dest, ++tag, comm, ++reqs);
        MPI_Isend(pd.m_y, n, MPI_DOUBLE, dest, ++tag, comm, ++reqs);
        MPI_Isend(pd.m_u, n, MPI_DOUBLE, dest, ++tag, comm, ++reqs);
        MPI_Isend(pd.m_v, n, MPI_DOUBLE, dest, ++tag, comm, ++reqs);
        MPI_Isend(pf.m_u, n, MPI_DOUBLE, dest, ++tag, comm, ++reqs);
        MPI_Isend(pf.m_v, n, MPI_DOUBLE, dest, ++tag, comm, ++reqs);
    }
}

// --------------------------------------------------------------------------
void recv(MPI_Comm comm, patch_data &pd, patch_force &pf, int src, int tag)
{
    long n = 0;
    MPI_Recv(&n, 1, MPI_LONG, src, tag, comm, MPI_STATUS_IGNORE);

    pd.resize(n);
    pf.resize(n);

    if (n)
    {
        MPI_Recv(pd.m_m, n, MPI_DOUBLE, src, ++tag, comm, MPI_STATUS_IGNORE);
        MPI_Recv(pd.m_x, n, MPI_DOUBLE, src, ++tag, comm, MPI_STATUS_IGNORE);
        MPI_Recv(pd.m_y, n, MPI_DOUBLE, src, ++tag, comm, MPI_STATUS_IGNORE);
        MPI_Recv(pd.m_u, n, MPI_DOUBLE, src, ++tag, comm, MPI_STATUS_IGNORE);
        MPI_Recv(pd.m_v, n, MPI_DOUBLE, src, ++tag, comm, MPI_STATUS_IGNORE);
        MPI_Recv(pf.m_u, n, MPI_DOUBLE, src, ++tag, comm, MPI_STATUS_IGNORE);
        MPI_Recv(pf.m_v, n, MPI_DOUBLE, src, ++tag, comm, MPI_STATUS_IGNORE);
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

    nf.resize(n*n);

    for (long j = 0; j < n; ++j)
    {
        double cxj = (p[j].m_x[1] + p[j].m_x[0]) / 2.;
        double cyj = (p[j].m_x[3] + p[j].m_x[2]) / 2.;

        for (long i = 0; i < n; ++i)
        {
            double cxi = (p[i].m_x[1] + p[i].m_x[0]) / 2.;
            double cyi = (p[i].m_x[3] + p[i].m_x[2]) / 2.;

            double dx = cxi - cxj;
            double dy = cyi - cyj;

            double r = sqrt(dx*dx + dy*dy);

            nf[j*n + i] = (r < nfr ? 1 : 0);
        }
    }
}

// --------------------------------------------------------------------------
void forces(MPI_Comm comm, patch_data &pd, patch_force &pf, double eps, const std::vector<int> &nf)
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
            int nreqs = 0;
            MPI_Request reqs[4];

            if (rank == i)
            {
                // send data to j (may be reduced representation)
                if (nf[j*n_ranks +i])
                {
                    // near by send all data
                    isend_mp(comm, pd, j, 3000, reqs, nreqs);
                }
                else
                {
                    // far away send reduced representation
                    patch_data rpd;
                    reduce(pd, rpd);
                    isend_mp(comm, rpd, j, 3000, reqs, nreqs);
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
                    isend_mp(comm, pd, i, 4000, reqs, nreqs);
                }
                else
                {
                    // far away send reduced representation
                    patch_data rpd;
                    reduce(pd, rpd);
                    isend_mp(comm, rpd, i, 4000, reqs, nreqs);
                }

                // calc force on bodies in j from bodies in i
                forces(pd, pdi, pf, eps);
            }

            // wait for sends to complete
            MPI_Waitall(nreqs, reqs, MPI_STATUS_IGNORE);
        }
    }
}


// --------------------------------------------------------------------------
void append(patch_data &pdo, patch_force &pfo, const patch_data &pdi, const patch_force &pfi)
{
    pdo.append(pdi);
    pfo.append(pfi);
}




















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

// --------------------------------------------------------------------------
bool intersects(const patch &lp, const patch &rp)
{
    double lx = std::max(lp.m_x[0], rp.m_x[0]);
    double hx = std::min(lp.m_x[1], rp.m_x[1]);
    double ly = std::max(lp.m_x[2], rp.m_x[2]);
    double hy = std::min(lp.m_x[3], rp.m_x[3]);
    return (lx <= hx) && (ly <= hy);
}

// --------------------------------------------------------------------------
bool inside(const patch &p, double x, double y)
{
    return (x >= p.m_x[0]) && (x < p.m_x[1]) && (y >= p.m_x[2]) && (y < p.m_x[3]);
}

/** finds the list of patch neighbors and returns them in pn. returns the
 * non-neighbor patches in pnn
 */
// --------------------------------------------------------------------------
void neighbors(const std::vector<patch> &patches, double ofs, std::vector<std::vector<int>> &pn, std::vector<std::vector<int>> &pnn)
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


// --------------------------------------------------------------------------
void split(int dir, const patch &p0, patch &p1, patch &p2)
{
    p1.m_owner = p0.m_owner;
    p2.m_owner = p0.m_owner;

    if (dir == 0)
    {
        p1.m_x[0] = p0.m_x[0];
        p1.m_x[1] = (p0.m_x[0] + p0.m_x[1]) / 2.;
        p1.m_x[2] = p0.m_x[2];
        p1.m_x[3] = p0.m_x[3];

        p2.m_x[0] = p1.m_x[1];
        p2.m_x[1] = p0.m_x[1];
        p2.m_x[2] = p0.m_x[2];
        p2.m_x[3] = p0.m_x[3];
    }
    else
    {
        p1.m_x[0] = p0.m_x[0];
        p1.m_x[1] = p0.m_x[1];
        p1.m_x[2] = p0.m_x[2];
        p1.m_x[3] = (p0.m_x[2] + p0.m_x[3]) / 2.;

        p2.m_x[0] = p0.m_x[0];
        p2.m_x[1] = p0.m_x[1];
        p2.m_x[2] = p1.m_x[3];
        p2.m_x[3] = p0.m_x[3];
    }
}

// --------------------------------------------------------------------------
void area(const std::vector<patch> &p, std::vector<double> &area)
{
    double mxa = 0.0;
    long n = p.size();
    area.resize(n);
    for (long i = 0; i < n; ++i)
    {
        const patch &pi = p[i];
        double a = (pi.m_x[1] - pi.m_x[0]) * (pi.m_x[3] - pi.m_x[2]);
        mxa = std::max(mxa, a);
        area[i] = a;
    }

    for (long i = 0; i < n; ++i)
    {
        area[i] /= mxa;
    }
}

// --------------------------------------------------------------------------
std::ostream &operator<<(std::ostream &os, const patch &p)
{
    os << "{" << p.m_owner << " [" << p.m_x[0] << ", " << p.m_x[1] << ", "
        << p.m_x[2] << ", " << p.m_x[3] << "]}";
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
std::vector<patch> partiton(const patch &dom, size_t n_out)
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
void initialize_random(MPI_Comm comm, long n, const patch &p, double m0, double m1, double v0, double v1, patch_data &pd)
{
    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    std::mt19937 gen(rank);
    std::uniform_real_distribution<double> dist(0.,1.);

    pd.resize(n);

    double dx = p.m_x[1] - p.m_x[0];
    double dy = p.m_x[3] - p.m_x[2];
    double dm = m1 - m0;
    double dv = v1 - v0;

    for (long i = 0; i < n; ++i)
    {
        double x = dx * std::max(0., std::min(1., dist(gen))) + p.m_x[0];
        double y = dy * std::max(0., std::min(1., dist(gen))) + p.m_x[2];
        double r = sqrt(x*x + y*y);
        double m = dm * std::max(0., std::min(1., dist(gen))) + m0;
        double v = dv * std::max(0., std::min(1., dist(gen))) + v0;

        pd.m_x[i] = x;
        pd.m_y[i] = y;
        pd.m_m[i] = m;
        pd.m_u[i] = v*(-y - .1*x) / r;
        pd.m_v[i] = v*( x - .1*y) / r;
    }

    if (rank == 0)
    {
        pd.m_x[0] = 0.;
        pd.m_y[0] = 0.;
        pd.m_m[0] = 1.989e30;
        pd.m_u[0] = 0.;
        pd.m_v[0] = 0.;
    }
}

/** identifes bodies inside the given patch */
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
void partition(MPI_Comm comm, const patch &dom, const std::vector<patch> &ps, const std::vector<std::vector<int>> &pn, const std::vector<std::vector<int>> &pnn, const patch_data &pd, std::vector<int> &dest)
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

// --------------------------------------------------------------------------
void package(const patch_data &pdi, const patch_force &pfi, const std::vector<int> &dest, int rank, patch_data &pdo, patch_force &pfo)
{
    long ni = pdi.size();

    std::vector<int> mask;
    std::vector<long> ids;

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

    copy(pdi, pfi, ids, mask, pdo, pfo);
}

// --------------------------------------------------------------------------
void move(MPI_Comm comm, patch_data &pd, patch_force &pf, const std::vector<int> &dest)
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
            int nreqs = 0;
            MPI_Request reqs[8];

            if (rank == i)
            {
                // send data to j
                patch_data pdi;
                patch_force pfi;
                package(pd, pf, dest, j, pdi, pfi);
                isend(comm, pdi, pfi, j, 5000, reqs, nreqs);

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
                isend(comm, pdj, pfj, i, 6000, reqs, nreqs);

                // add to the output
                append(pdo, pfo, pdi, pfi);
            }

            // wait for sends to complete
            MPI_Waitall(nreqs, reqs, MPI_STATUS_IGNORE);
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
void velocity_verlet(MPI_Comm comm, patch_data &pd, patch_force &pf, double h, double eps, const std::vector<int> &nf)
{
    double h2 = h/2.;
    long n = pd.size();
    for (long i = 0; i < n; ++i)
    {
        // half step velocity
        pd.m_u[i] = pd.m_u[i] + h2 * pf.m_u[i] / pd.m_m[i];
        pd.m_v[i] = pd.m_v[i] + h2 * pf.m_v[i] / pd.m_m[i];

        // full step position
        pd.m_x[i] = pd.m_x[i] + h * pd.m_u[i];
        pd.m_y[i] = pd.m_y[i] + h * pd.m_v[i];
    }

    // update the forces
    forces(comm, pd, pf, eps, nf);

    for (long i = 0; i < n; ++i)
    {
        // half step velocity
        pd.m_u[i] = pd.m_u[i] + h2 * pf.m_u[i] / pd.m_m[i];
        pd.m_v[i] = pd.m_v[i] + h2 * pf.m_v[i] / pd.m_m[i];
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
        x[ii    ] = pd.m_x[i];
        x[ii + 1] = pd.m_y[i];
        x[ii + 2] = 0.;

        m[i] = pd.m_m[i];
        vu[i] = pd.m_u[i];
        vv[i] = pd.m_v[i];
        fu[i] = pf.m_u[i];
        fv[i] = pf.m_v[i];
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

        long ii = 12*i;
        x[ii    ] = pi.m_x[0];
        x[ii + 1] = pi.m_x[2];
        x[ii + 2] = 0.;

        ii += 3;
        x[ii    ] = pi.m_x[1];
        x[ii + 1] = pi.m_x[2];
        x[ii + 2] = 0.;

        ii += 3;
        x[ii    ] = pi.m_x[1];
        x[ii + 1] = pi.m_x[3];
        x[ii + 2] = 0.;

        ii += 3;
        x[ii    ] = pi.m_x[0];
        x[ii + 1] = pi.m_x[3];
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


int main(int argc, char **argv)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Init(&argc, &argv);

    int rank = 0;
    int n_ranks = 1;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &n_ranks);

    // initial condition
    double x0 = -5906.4e9;
    double x1 = 5906.4e9;
    double y0 = -5906.4e9;
    double y1 = 5906.4e9;
    double m0 = 10.0e24;
    double m1 = 100.0e24;
    double v0 = 1.0e3;
    double v1 = 10.0e3;
    double dx = x1 - x0;
    double dy = y1 - y0;
    double nfr = 2.*sqrt(dx*dx + dy*dy); // / 4.;
    double eps = 0.;
    double h = 4.*24.*3600.;
    long nb = 10;
    long nits = 1000;
    long io_int = 10;
    const char *odir = "output";

    // partition space
    patch dom(0, x0, x1, y0, y1);
    std::vector<patch> patches = partiton(dom, n_ranks);

    if (io_int)
       write(comm, patches, odir);

    // flag nearby patches
    std::vector<int> nf;
    near(patches, nfr, nf);

    // patch neighbors
    std::vector<std::vector<int>> pn;
    std::vector<std::vector<int>> pnn;
    neighbors(patches, fabs(x1 - x0)/1e3, pn, pnn);

    if (rank == 0)
    {
    std::cerr << "patches = " << patches << std::endl;
    std::cerr << "neighbors = " << pn << std::endl;
    std::cerr << "non-neighbors = " << pnn << std::endl;
    }

    // initialize bodies
    std::vector<double> pa;
    area(patches, pa);

    nb = std::max(1l, (long) (nb * pa[rank]));

    patch_data pd;
    pd.resize(nb);

    initialize_random(comm, nb, patches[rank], m0, m1, v0, v1, pd);

    // initialize forces
    patch_force pf;
    pf.resize(nb);

    forces(comm, pd, pf, eps, nf);

    // write initial state
    if (io_int)
       write(comm, pd, pf, odir);

    // iterate
    long it = 0;
    while (it < nits)
    {
        // update bodies
        velocity_verlet(comm, pd, pf, h, eps, nf);

        // update partition
        std::vector<int> dest;
        partition(comm, dom, patches, pn, pnn, pd, dest);
        move(comm, pd, pf, dest);

        // write current state
        if (io_int && (((it + 1) % io_int) == 0))
            write(comm, pd, pf, odir);

        it += 1;
    }


    MPI_Finalize();

    return 0;
}


/*
/// read n elements of type T
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
/// write n elements of type T
template <typename T>
int writen(int fh, T *buf, size_t n)
{
    ssize_t ierr = 0;
    ssize_t nb = n*sizeof(T);
    if ((ierr = write(fh, buf, nb)) != nb)
    {
        std::cerr << "Failed to write " << n << " elements of size "
            << sizeof(T) << std::endl << strerror(errno) << std::endl;
        return -1;
    }
    return 0;
}

/// memory for the nbody solver
struct solver_data
{
public:
    solver_data() = delete;

    solver_data(int devid) :
        m_devid(devid), m_t(0), m_dt(1), m_step(0), m_size(0), m_id(nullptr),
        m_m(nullptr), m_x(nullptr), m_y(nullptr), m_z(nullptr), m_u(nullptr),
        m_v(nullptr), m_uz(nullptr), m_u(nullptr), m_v(nullptr),
        m_fz(nullptr) {}

    ~solver_data();

    /// load state form disk. this is how you initialize the solver
    int load_state(const std::string &fn);

    /// save the current state to disk. use to make a checkpoint.
    int write_state(const std::string &fn);

    int m_devid;    ///< device owning the data
    double m_t;     ///< current simulated time
    double m_dt;    ///< time step for sovler
    long m_step;    ///< step number
    long m_size;    ///< number of bodies
    long *m_id;     ///< body id
    double *m_m;    ///< body mass
    double *m_x;    ///< body position x
    double *m_y;    ///< body position y
    double *m_z;    ///< body position z
    double *m_u;   ///< body velocity x
    double *m_v;   ///< body velocity y
    double *m_uz;   ///< body velocity z
    double *m_u;   ///< body force x
    double *m_v;   ///< body force y
    double *m_fz;   ///< body force z
};

// --------------------------------------------------------------------------
int solver_data::load_state(const std::string &fn)
{
    int h = open(fn.c_str(), O_RDONLY);
    if (h < 0)
    {
        std::cerr << "Failed to open \"" << fn << "\"" << std::endl
            << strerror(errno) << std::endl;
        return -1;
    }

    if (readn(h, &m_t, 1 ) ||
        readn(h, &m_dt, 1) ||
        readn(h, &m_step, 1) ||
        readn(h, &m_size, 1))
    {
        std::cerr << "Failed to read \"" << fn << "\"" << std::endl;
        close(h);
        return -1;
    }

    size_t nbytes = m_size*sizeof(double);
    m_id = (long*)malloc(m_size*sizeof(long));
    m_m = (double*)malloc(nbytes);
    m_x = (double*)malloc(nbytes);
    m_y = (double*)malloc(nbytes);
    m_z = (double*)malloc(nbytes);
    m_u = (double*)malloc(nbytes);
    m_v = (double*)malloc(nbytes);
    m_uz = (double*)malloc(nbytes);
    m_u = (double*)malloc(nbytes);
    m_v = (double*)malloc(nbytes);
    m_fz = (double*)malloc(nbytes);


    if (readn(h, m_id, m_size) ||
        readn(h, m_m, m_size) ||
        readn(h, m_x, m_size) ||
        readn(h, m_y, m_size) ||
        readn(h, m_z, m_size) ||
        readn(h, m_u, m_size) ||
        readn(h, m_v, m_size) ||
        readn(h, m_uz, m_size) ||
        readn(h, m_u, m_size) ||
        readn(h, m_v, m_size) ||
        readn(h, m_fz, m_size))
    {
        std::cerr << "Failed to read \"" << fn << "\"" << std::endl;
        close(h);
        return -1;
    }

    close(h);
    return 0;
}

// --------------------------------------------------------------------------
int solver_data::save_state(const std::string &fn)
{
    int h = open(fn.c_str(), O_WRONLY|O_CREAT);
    if (h < 0)
    {
        std::cerr << "Failed to open \"" << fn << "\"" << std::endl
            << strerror(errno) << std::endl;
        return -1;
    }

    if (writen(h, &m_t, 1 ) ||
        writen(h, &m_dt, 1) ||
        writen(h, &m_step, 1) ||
        writen(h, &m_size, 1) ||
        writen(h, m_id, m_size) ||
        writen(h, m_m, m_size) ||
        writen(h, m_x, m_size) ||
        writen(h, m_y, m_size) ||
        writen(h, m_z, m_size) ||
        writen(h, m_u, m_size) ||
        writen(h, m_v, m_size) ||
        writen(h, m_uz, m_size) ||
        writen(h, m_u, m_size) ||
        writen(h, m_v, m_size) ||
        writen(h, m_fz, m_size))
    {
        std::cerr << "Failed to write \"" << fn << "\"" << std::endl;
        close(h);
        return -1;
    }

    close(h);
    return 0;
}




*/
