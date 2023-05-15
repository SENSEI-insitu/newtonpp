#include "domain_decomp.h"
#include "communication.h"
#include "stream_compact.h"

#include <deque>
#include <math.h>

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
            double czj = (pj_x[5] + pj_x[4]) / 2.;

            double cxi = (pi_x[1] + pi_x[0]) / 2.;
            double cyi = (pi_x[3] + pi_x[2]) / 2.;
            double czi = (pi_x[5] + pi_x[4]) / 2.;

            double dx = cxi - cxj;
            double dy = cyi - cyj;
            double dz = czi - czj;

            double r = sqrt(dx*dx + dy*dy + dz*dz);

            pnf[j*n + i] = (r < nfr ? 1 : 0);
            }
        }
    }
    }
}

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
        p1_x[4] = p0_x[4];
        p1_x[5] = p0_x[5];

        p2_x[0] = p1_x[1];
        p2_x[1] = p0_x[1];
        p2_x[2] = p0_x[2];
        p2_x[3] = p0_x[3];
        p2_x[4] = p0_x[4];
        p2_x[5] = p0_x[5];
    }
    else if (dir == 1)
    {
        p1_x[0] = p0_x[0];
        p1_x[1] = p0_x[1];
        p1_x[2] = p0_x[2];
        p1_x[3] = (p0_x[2] + p0_x[3]) / 2.;
        p1_x[4] = p0_x[4];
        p1_x[5] = p0_x[5];

        p2_x[0] = p0_x[0];
        p2_x[1] = p0_x[1];
        p2_x[2] = p1_x[3];
        p2_x[3] = p0_x[3];
        p2_x[4] = p0_x[4];
        p2_x[5] = p0_x[5];
    }
    else
    {
        p1_x[0] = p0_x[0];
        p1_x[1] = p0_x[1];
        p1_x[2] = p0_x[2];
        p1_x[3] = p0_x[3];
        p1_x[4] = p0_x[4];
        p1_x[5] = (p0_x[4] + p0_x[5]) / 2.;

        p2_x[0] = p0_x[1];
        p2_x[1] = p0_x[1];
        p2_x[2] = p0_x[2];
        p2_x[3] = p0_x[3];
        p2_x[4] = p1_x[5];
        p2_x[5] = p0_x[5];

    }
}

// --------------------------------------------------------------------------
int inside(const double *p, double x, double y, double z)
{
    return (x >= p[0]) && (x < p[1]) && (y >= p[2]) && (y < p[3]) && (z >= p[4]) && (z < p[5]);
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
    atot = (dom_x[1] - dom_x[0]) * (dom_x[3] - dom_x[2]) * (dom_x[5] - dom_x[4]);
    }

    // compute fraction of the area covered by the patch
    for (long i = 0; i < n; ++i)
    {
        const patch &pi = p[i];
        const double *pi_x = pi.m_x.data();

        #pragma omp target map(alloc:atot), is_device_ptr(pa,pi_x,dom_x)
        {
        pa[i] = (pi_x[1] - pi_x[0]) * (pi_x[3] - pi_x[2]) * (pi_x[5] - pi_x[4]) / atot;
        }
    }

    #pragma omp target exit data map(release:atot)
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
        int dir = pass % 3;
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
#if defined(__NVCOMPILER)
    #pragma omp target teams loop is_device_ptr(pdest)
#else
    #pragma omp target teams distribute parallel for is_device_ptr(pdest)
#endif
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
    const double *pd_z = pd.m_z.data();

    int nps = ps.size();
    for (int j = 0; j < nps; ++j)
    {
        // get the patch corners
        const double *p_x = ps[j].m_x.data();

        // test each body to see if it's inside this patch
#if defined(__NVCOMPILER)
        #pragma omp target teams loop is_device_ptr(pd_x,pd_y,pd_z,p_x,pdest)
#else
        #pragma omp target teams distribute parallel for is_device_ptr(pd_x,pd_y,pd_z,p_x,pdest)
#endif
        for (long i = 0; i < n; ++i)
        {
            if ((pdest[i] < 0) &&
               (pd_x[i] >= p_x[0]) && (pd_x[i] < p_x[1]) &&
               (pd_y[i] >= p_x[2]) && (pd_y[i] < p_x[3]) &&
               (pd_z[i] >= p_x[4]) && (pd_z[i] < p_x[5]))
            {
                pdest[i] = j;
            }
        }
    }
}

// --------------------------------------------------------------------------
void package(const patch_data &pdi, const patch_force &pfi,
    const hamr::buffer<int> &dest, int rank, patch_data &pdo, patch_force &pfo)
{
    long ni = pdi.size();

    hamr::buffer<int> mask(def_alloc(), ni, 0);
    int *pmask = mask.data();

    const int *pdest = dest.data();

    // ideintify the bodies that are owned by the specified rank
#if defined(__NVCOMPILER)
    #pragma omp target teams loop is_device_ptr(pmask,pdest)
#else
    #pragma omp target teams distribute parallel for is_device_ptr(pmask,pdest)
#endif
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
    cuda::stream_compact(
        pdo.m_m.data(),
        pdo.m_x.data(), pdo.m_y.data(), pdo.m_z.data(),
        pdo.m_u.data(), pdo.m_v.data(), pdo.m_w.data(),
        pfo.m_u.data(), pfo.m_v.data(), pfo.m_w.data(),
        no,
        pdi.m_m.data(),
        pdi.m_x.data(), pdi.m_y.data(), pdi.m_z.data(),
        pdi.m_u.data(), pdi.m_v.data(), pdi.m_w.data(),
        pfi.m_u.data(), pfi.m_v.data(), pfi.m_w.data(),
        pmask, ni, threads_per_block);
#else
    cpu::stream_compact(
        pdo.m_m.data(),
        pdo.m_x.data(), pdo.m_y.data(), pdo.m_z.data(),
        pdo.m_u.data(), pdo.m_v.data(), pdo.m_w.data(),
        pfo.m_u.data(), pfo.m_v.data(), pfo.m_w.data(),
        no,
        pdi.m_m.data(),
        pdi.m_x.data(), pdi.m_y.data(), pdi.m_z.data(),
        pdi.m_u.data(), pdi.m_v.data(), pdi.m_w.data(),
        pfi.m_u.data(), pfi.m_v.data(), pfi.m_w.data(),
        pmask, ni);
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
                pdo.append(pdj);
                pfo.append(pfj);
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
                pdo.append(pdi);
                pfo.append(pfi);
            }

            // wait for sends to complete
            if (reqs.m_size)
                MPI_Waitall(reqs.m_size, reqs.m_req, MPI_STATUS_IGNORE);
        }
    }

    pd = std::move(pdo);
    pf = std::move(pfo);
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
