#include "communication.h"
#include "memory_management.h"

#include <cassert>
#include <iostream>
#include <utility>

// --------------------------------------------------------------------------
void isend_mp(MPI_Comm comm, const patch_data &pd, int dest, int tag, requests &reqs)
{
    long n = pd.size();
    MPI_Send(&n, 1, MPI_LONG, dest, tag, comm);
    reqs.m_size = 0;

    if (n)
    {
        reqs.m_size = 4;

        auto [spm, pm, spx, px, spy, py, spz, pz] = pd.get_mp_cpu_accessible();

        MPI_Request *req = reqs.m_req;

        MPI_Isend(pm, n, MPI_DOUBLE, dest, ++tag, comm,   req);
        MPI_Isend(px, n, MPI_DOUBLE, dest, ++tag, comm, ++req);
        MPI_Isend(py, n, MPI_DOUBLE, dest, ++tag, comm, ++req);
        MPI_Isend(pz, n, MPI_DOUBLE, dest, ++tag, comm, ++req);

        reqs.m_data = {spm, spx, spy, spz};
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
        hamr::buffer<double> z(hamr::buffer_allocator::malloc, n);

        MPI_Recv(m.data(), n, MPI_DOUBLE, src, ++tag, comm, MPI_STATUS_IGNORE);
        MPI_Recv(x.data(), n, MPI_DOUBLE, src, ++tag, comm, MPI_STATUS_IGNORE);
        MPI_Recv(y.data(), n, MPI_DOUBLE, src, ++tag, comm, MPI_STATUS_IGNORE);
        MPI_Recv(z.data(), n, MPI_DOUBLE, src, ++tag, comm, MPI_STATUS_IGNORE);

        pd.m_m = std::move(m);
        pd.m_x = std::move(x);
        pd.m_y = std::move(y);
        pd.m_z = std::move(z);
    }
}

// --------------------------------------------------------------------------
void isend(MPI_Comm comm, const patch_force &pf, int dest, int tag)
{
    long n = pf.size();

    auto [spf_u, pf_u, spf_v, pf_v, spf_w, pf_w] = pf.get_cpu_accessible();

    MPI_Send(&n, 1, MPI_LONG, dest, ++tag, comm);
    MPI_Send(pf_u, n, MPI_DOUBLE, dest, ++tag, comm);
    MPI_Send(pf_v, n, MPI_DOUBLE, dest, ++tag, comm);
    MPI_Send(pf_w, n, MPI_DOUBLE, dest, ++tag, comm);
}

// --------------------------------------------------------------------------
void recv(MPI_Comm comm, patch_force &pf, int src, int tag)
{
    long n = 0;
    MPI_Recv(&n, 1, MPI_LONG, src, ++tag, comm, MPI_STATUS_IGNORE);

    assert(pf.size() == n);

    hamr::buffer<double> pf_u(hamr::buffer_allocator::malloc, n);
    hamr::buffer<double> pf_v(hamr::buffer_allocator::malloc, n);
    hamr::buffer<double> pf_w(hamr::buffer_allocator::malloc, n);

    MPI_Recv(pf_u.data(), n, MPI_DOUBLE, src, ++tag, comm, MPI_STATUS_IGNORE);
    MPI_Recv(pf_v.data(), n, MPI_DOUBLE, src, ++tag, comm, MPI_STATUS_IGNORE);
    MPI_Recv(pf_w.data(), n, MPI_DOUBLE, src, ++tag, comm, MPI_STATUS_IGNORE);

    pf.m_u = std::move(pf_u);
    pf.m_v = std::move(pf_v);
    pf.m_w = std::move(pf_w);
}

// --------------------------------------------------------------------------
void isend(MPI_Comm comm, const patch_data &pd,
    const patch_force &pf, int dest, int tag, requests &reqs)
{
    long n = pd.size();
    MPI_Send(&n, 1, MPI_LONG, dest, tag, comm);
    reqs.m_size = 0;

    if (n)
    {
        auto [spd_m, pd_m,
              spd_x, pd_x,
              spd_y, pd_y,
              spd_z, pd_z,
              spd_u, pd_u,
              spd_v, pd_v,
              spd_w, pd_w] = pd.get_cpu_accessible();

        auto [spf_u, pf_u,
              spf_v, pf_v,
              spf_w, pf_w] = pf.get_cpu_accessible();

        reqs.m_data = {spd_m, spd_x, spd_y, spd_z, spd_u, spd_v, spd_w, spf_u, spf_v, spf_w};

        reqs.m_size = 10;
        MPI_Request *req = reqs.m_req;

        MPI_Isend(pd_m, n, MPI_DOUBLE, dest, ++tag, comm,   req);
        MPI_Isend(pd_x, n, MPI_DOUBLE, dest, ++tag, comm, ++req);
        MPI_Isend(pd_y, n, MPI_DOUBLE, dest, ++tag, comm, ++req);
        MPI_Isend(pd_z, n, MPI_DOUBLE, dest, ++tag, comm, ++req);
        MPI_Isend(pd_u, n, MPI_DOUBLE, dest, ++tag, comm, ++req);
        MPI_Isend(pd_v, n, MPI_DOUBLE, dest, ++tag, comm, ++req);
        MPI_Isend(pd_w, n, MPI_DOUBLE, dest, ++tag, comm, ++req);
        MPI_Isend(pf_u, n, MPI_DOUBLE, dest, ++tag, comm, ++req);
        MPI_Isend(pf_v, n, MPI_DOUBLE, dest, ++tag, comm, ++req);
        MPI_Isend(pf_w, n, MPI_DOUBLE, dest, ++tag, comm, ++req);
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
        hamr::buffer<double> tpd_m(cpu_alloc(), n);
        hamr::buffer<double> tpd_x(cpu_alloc(), n);
        hamr::buffer<double> tpd_y(cpu_alloc(), n);
        hamr::buffer<double> tpd_z(cpu_alloc(), n);
        hamr::buffer<double> tpd_u(cpu_alloc(), n);
        hamr::buffer<double> tpd_v(cpu_alloc(), n);
        hamr::buffer<double> tpd_w(cpu_alloc(), n);
        hamr::buffer<double> tpf_u(cpu_alloc(), n);
        hamr::buffer<double> tpf_v(cpu_alloc(), n);
        hamr::buffer<double> tpf_w(cpu_alloc(), n);

        MPI_Recv(tpd_m.data(), n, MPI_DOUBLE, src, ++tag, comm, MPI_STATUS_IGNORE);
        MPI_Recv(tpd_x.data(), n, MPI_DOUBLE, src, ++tag, comm, MPI_STATUS_IGNORE);
        MPI_Recv(tpd_y.data(), n, MPI_DOUBLE, src, ++tag, comm, MPI_STATUS_IGNORE);
        MPI_Recv(tpd_z.data(), n, MPI_DOUBLE, src, ++tag, comm, MPI_STATUS_IGNORE);
        MPI_Recv(tpd_u.data(), n, MPI_DOUBLE, src, ++tag, comm, MPI_STATUS_IGNORE);
        MPI_Recv(tpd_v.data(), n, MPI_DOUBLE, src, ++tag, comm, MPI_STATUS_IGNORE);
        MPI_Recv(tpd_w.data(), n, MPI_DOUBLE, src, ++tag, comm, MPI_STATUS_IGNORE);
        MPI_Recv(tpf_u.data(), n, MPI_DOUBLE, src, ++tag, comm, MPI_STATUS_IGNORE);
        MPI_Recv(tpf_v.data(), n, MPI_DOUBLE, src, ++tag, comm, MPI_STATUS_IGNORE);
        MPI_Recv(tpf_w.data(), n, MPI_DOUBLE, src, ++tag, comm, MPI_STATUS_IGNORE);

        pd.m_m = std::move(tpd_m);
        pd.m_x = std::move(tpd_x);
        pd.m_y = std::move(tpd_y);
        pd.m_z = std::move(tpd_z);
        pd.m_u = std::move(tpd_u);
        pd.m_v = std::move(tpd_v);
        pd.m_w = std::move(tpd_w);
        pf.m_u = std::move(tpf_u);
        pf.m_v = std::move(tpf_v);
        pf.m_w = std::move(tpf_w);
    }
}
