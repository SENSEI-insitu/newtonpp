#include "solver.h"
#include "communication.h"

#include <assert.h>
#include <math.h>

// --------------------------------------------------------------------------
void forces(const patch_data &pd, patch_force &pf, double G, double eps)
{
    long n = pd.size();

    assert(pf.size() == n);

    double eps2 = eps*eps;

#if defined(USE_STRUCTURED_BINDINGS)
    auto [pd_m, pd_x, pd_y, pd_z] = pd.get_mp_data();
    auto [pf_u, pf_v, pf_w] = pf.get_data();
#else
    const double *pd_m = pd.m_m.data();
    const double *pd_x = pd.m_x.data();
    const double *pd_y = pd.m_y.data();
    const double *pd_z = pd.m_z.data();

    double *pf_u = pf.m_u.data();
    double *pf_v = pf.m_v.data();
    double *pf_w = pf.m_w.data();
#endif

#if defined(NEWTONPP_USE_OMP_LOOP)
    #pragma omp target teams loop is_device_ptr(pf_u,pf_v,pf_w,pd_m,pd_x,pd_y,pd_z)
#else
    #pragma omp target teams distribute is_device_ptr(pf_u,pf_v,pf_w,pd_m,pd_x,pd_y,pd_z)
#endif
    for (long i = 0; i < n; ++i)
    {
        double fx = 0.;
        double fy = 0.;
        double fz = 0.;

#if defined(NEWTONPP_USE_OMP_LOOP)
        #pragma omp loop reduction(+: fx,fy,fz)
#else
        #pragma omp parallel for reduction(+: fx,fy,fz)
#endif
        for (long j = 0; j < n; ++j)
        {
            double rx = pd_x[j] - pd_x[i];
            double ry = pd_y[j] - pd_y[i];
            double rz = pd_z[j] - pd_z[i];

            double r2e2 = rx*rx + ry*ry + rz*rz + eps2;
            double r2e23 = r2e2*r2e2*r2e2;

            double mf = G*pd_m[i]*pd_m[j] / sqrt(r2e23);

            fx += (i == j ? 0. : rx*mf);
            fy += (i == j ? 0. : ry*mf);
            fz += (i == j ? 0. : rz*mf);
        }

        pf_u[i] = fx;
        pf_v[i] = fy;
        pf_w[i] = fz;
    }
}

// --------------------------------------------------------------------------
void forces(const patch_data &lpd, const patch_data &rpd, patch_force &pf, double G, double eps)
{
    long n = lpd.size();
    long m = rpd.size();

    pf.resize(n);

    double eps2 = eps*eps;

#if defined(USE_STRUCTURED_BINDINGS)
    auto [lpd_m, lpd_x, lpd_y, lpd_z] = lpd.get_mp_data();
    auto [rpd_m, rpd_x, rpd_y, rpd_z] = rpd.get_mp_data();
    auto [pf_u, pf_v, pf_w] = pf.get_data();
#else
    const double *lpd_m = lpd.m_m.data();
    const double *lpd_x = lpd.m_x.data();
    const double *lpd_y = lpd.m_y.data();
    const double *lpd_z = lpd.m_z.data();

    const double *rpd_m = rpd.m_m.data();
    const double *rpd_x = rpd.m_x.data();
    const double *rpd_y = rpd.m_y.data();
    const double *rpd_z = rpd.m_z.data();

    double *pf_u = pf.m_u.data();
    double *pf_v = pf.m_v.data();
    double *pf_w = pf.m_w.data();
#endif

#if defined(NEWTONPP_USE_OMP_LOOP)
    #pragma omp target teams loop is_device_ptr(pf_u,pf_v,pf_w,lpd_m,lpd_x,lpd_y,lpd_z, rpd_m,rpd_x,rpd_y,rpd_z)
#else
    #pragma omp target teams distribute is_device_ptr(pf_u,pf_v,pf_w,lpd_m,lpd_x,lpd_y,lpd_z, rpd_m,rpd_x,rpd_y,rpd_z)
#endif
    for (long i = 0; i < n; ++i)
    {
        double fx = 0.;
        double fy = 0.;
        double fz = 0.;

#if defined(NEWTONPP_USE_OMP_LOOP)
        #pragma omp loop reduction(+: fx,fy,fz)
#else
        #pragma omp parallel for reduction(+: fx,fy,fz)
#endif
        for (long j = 0; j < m; ++j)
        {
            double rx = rpd_x[j] - lpd_x[i];
            double ry = rpd_y[j] - lpd_y[i];
            double rz = rpd_z[j] - lpd_z[i];

            double r2e2 = rx*rx + ry*ry + rz*rz + eps2;
            double r2e23 = r2e2*r2e2*r2e2;

            double mf = G*lpd_m[i]*rpd_m[j] / sqrt(r2e23);

            fx += rx*mf;
            fy += ry*mf;
            fz += rz*mf;
        }

        pf_u[i] += fx;
        pf_v[i] += fy;
        pf_w[i] += fz;
    }
}

// --------------------------------------------------------------------------
void forces(MPI_Comm comm, patch_data &pd, patch_force &pf,
    double G, double eps, const std::vector<int> &nf)
{
    int rank = 0;
    int n_ranks = 1;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &n_ranks);

    long n = pd.size();
    pf.resize(n);

    // local forces
    forces(pd, pf, G, eps);

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
                forces(pd, pdj, pf, G, eps);
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
                forces(pd, pdi, pf, G, eps);
            }

            // wait for sends to complete
            if (reqs.m_size)
                MPI_Waitall(reqs.m_size, reqs.m_req, MPI_STATUS_IGNORE);
        }
    }
}

// --------------------------------------------------------------------------
void velocity_verlet(MPI_Comm comm,
    patch_data &pd, patch_force &pf, double G, double h,
    double eps, const std::vector<int> &nf)
{
    double h2 = h/2.;
    long n = pd.size();

#if defined(USE_STRUCTURED_BINDINGS)
    auto [pd_m, pd_x, pd_y, pd_z, pd_u, pd_v, pd_w] = pd.get_data();
    auto [pf_u, pf_v, pf_w] = pf.get_data();
#else
    double *pd_m = pd.m_m.data();
    double *pd_x = pd.m_x.data();
    double *pd_y = pd.m_y.data();
    double *pd_z = pd.m_z.data();
    double *pd_u = pd.m_u.data();
    double *pd_v = pd.m_v.data();
    double *pd_w = pd.m_w.data();

    double *pf_u = pf.m_u.data();
    double *pf_v = pf.m_v.data();
    double *pf_w = pf.m_w.data();
#endif

#if defined(NEWTONPP_USE_OMP_LOOP)
    #pragma omp target teams loop is_device_ptr(pd_m,pd_x,pd_y,pd_z,pd_u,pd_v,pd_w,pf_u,pf_v,pf_w)
#else
    #pragma omp target teams distribute parallel for is_device_ptr(pd_m,pd_x,pd_y,pd_z,pd_u,pd_v,pd_w,pf_u,pf_v,pf_w)
#endif
    for (long i = 0; i < n; ++i)
    {
        // half step velocity
        pd_u[i] = pd_u[i] + h2 * pf_u[i] / pd_m[i];
        pd_v[i] = pd_v[i] + h2 * pf_v[i] / pd_m[i];
        pd_w[i] = pd_w[i] + h2 * pf_w[i] / pd_m[i];

        // full step position
        pd_x[i] = pd_x[i] + h * pd_u[i];
        pd_y[i] = pd_y[i] + h * pd_v[i];
        pd_z[i] = pd_z[i] + h * pd_w[i];
    }

    // update the forces
    forces(comm, pd, pf, G, eps, nf);

#if defined(NEWTONPP_USE_OMP_LOOP)
    #pragma omp target teams loop is_device_ptr(pd_m,pd_u,pd_v,pd_w,pf_u,pf_v,pf_w)
#else
    #pragma omp target teams distribute parallel for is_device_ptr(pd_m,pd_u,pd_v,pd_w,pf_u,pf_v,pf_w)
#endif
    for (long i = 0; i < n; ++i)
    {
        // half step velocity
        pd_u[i] = pd_u[i] + h2 * pf_u[i] / pd_m[i];
        pd_v[i] = pd_v[i] + h2 * pf_v[i] / pd_m[i];
        pd_w[i] = pd_w[i] + h2 * pf_w[i] / pd_m[i];
    }
}
