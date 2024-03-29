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

#if defined(NEWTONPP_GPU_DIRECT)
        auto [pm, px, py, pz] = pd.get_mp_data();
#else
        auto [spm, pm, spx, px, spy, py, spz, pz] = pd.get_mp_host_accessible();
        reqs.m_data = {spm, spx, spy, spz};
#endif

        MPI_Request *req = reqs.m_req;

        MPI_Isend(pm, n, MPI_DOUBLE, dest, ++tag, comm,   req);
        MPI_Isend(px, n, MPI_DOUBLE, dest, ++tag, comm, ++req);
        MPI_Isend(py, n, MPI_DOUBLE, dest, ++tag, comm, ++req);
        MPI_Isend(pz, n, MPI_DOUBLE, dest, ++tag, comm, ++req);
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
#if defined(NEWTONPP_GPU_DIRECT)
        auto alloc = gpu_alloc();
#else
        auto alloc = cpu_alloc();
#endif
        hamr::buffer<double> m(alloc, n);
        hamr::buffer<double> x(alloc, n);
        hamr::buffer<double> y(alloc, n);
        hamr::buffer<double> z(alloc, n);

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

#if defined(NEWTONPP_GPU_DIRECT)
    auto [pf_u, pf_v, pf_w] = pf.get_data();
#else
    auto [spf_u, pf_u, spf_v, pf_v, spf_w, pf_w] = pf.get_host_accessible();
#endif

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

#if defined(NEWTONPP_GPU_DIRECT)
        auto alloc = gpu_alloc();
#else
        auto alloc = cpu_alloc();
#endif
    hamr::buffer<double> pf_u(alloc, n);
    hamr::buffer<double> pf_v(alloc, n);
    hamr::buffer<double> pf_w(alloc, n);

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
#if defined(NEWTONPP_GPU_DIRECT)
        auto [pd_m, pd_x, pd_y, pd_z, pd_u, pd_v, pd_w, pd_id] = pd.get_data();
        auto [pf_u, pf_v, pf_w] = pf.get_data();
#else
        auto [spd_m, pd_m, spd_x, pd_x, spd_y, pd_y, spd_z, pd_z,
              spd_u, pd_u, spd_v, pd_v, spd_w, pd_w, spd_id, pd_id] = pd.get_host_accessible();

        auto [spf_u, pf_u, spf_v, pf_v, spf_w, pf_w] = pf.get_host_accessible();

        reqs.m_data = {spd_m, spd_x, spd_y, spd_z, spd_u, spd_v, spd_w, spf_u, spf_v, spf_w};
        reqs.m_idata = spd_id;
#endif
        reqs.m_size = 11;
        MPI_Request *req = reqs.m_req;

        MPI_Isend(pd_m, n, MPI_DOUBLE, dest, ++tag, comm,   req);
        MPI_Isend(pd_x, n, MPI_DOUBLE, dest, ++tag, comm, ++req);
        MPI_Isend(pd_y, n, MPI_DOUBLE, dest, ++tag, comm, ++req);
        MPI_Isend(pd_z, n, MPI_DOUBLE, dest, ++tag, comm, ++req);
        MPI_Isend(pd_u, n, MPI_DOUBLE, dest, ++tag, comm, ++req);
        MPI_Isend(pd_v, n, MPI_DOUBLE, dest, ++tag, comm, ++req);
        MPI_Isend(pd_w, n, MPI_DOUBLE, dest, ++tag, comm, ++req);
        MPI_Isend(pd_id,n, MPI_INT,    dest, ++tag, comm, ++req);
        MPI_Isend(pf_u, n, MPI_DOUBLE, dest, ++tag, comm, ++req);
        MPI_Isend(pf_v, n, MPI_DOUBLE, dest, ++tag, comm, ++req);
        MPI_Isend(pf_w, n, MPI_DOUBLE, dest, ++tag, comm, ++req);
    }
}

// --------------------------------------------------------------------------
void send(MPI_Comm comm, const patch_data &pd,
    const patch_force &pf, int dest, int tag)
{
    long n = pd.size();
    MPI_Send(&n, 1, MPI_LONG, dest, tag, comm);

    if (n)
    {
#if defined(NEWTONPP_GPU_DIRECT)
        auto [pd_m, pd_x, pd_y, pd_z, pd_u, pd_v, pd_w, pd_id] = pd.get_data();
        auto [pf_u, pf_v, pf_w] = pf.get_data();
#else
        auto [spd_m, pd_m, spd_x, pd_x, spd_y, pd_y, spd_z, pd_z,
              spd_u, pd_u, spd_v, pd_v, spd_w, pd_w, spd_id, pd_id] = pd.get_host_accessible();

        auto [spf_u, pf_u, spf_v, pf_v, spf_w, pf_w] = pf.get_host_accessible();
#endif
        MPI_Send(pd_m, n, MPI_DOUBLE, dest, ++tag, comm);
        MPI_Send(pd_x, n, MPI_DOUBLE, dest, ++tag, comm);
        MPI_Send(pd_y, n, MPI_DOUBLE, dest, ++tag, comm);
        MPI_Send(pd_z, n, MPI_DOUBLE, dest, ++tag, comm);
        MPI_Send(pd_u, n, MPI_DOUBLE, dest, ++tag, comm);
        MPI_Send(pd_v, n, MPI_DOUBLE, dest, ++tag, comm);
        MPI_Send(pd_w, n, MPI_DOUBLE, dest, ++tag, comm);
        MPI_Send(pd_id,n, MPI_INT,    dest, ++tag, comm);
        MPI_Send(pf_u, n, MPI_DOUBLE, dest, ++tag, comm);
        MPI_Send(pf_v, n, MPI_DOUBLE, dest, ++tag, comm);
        MPI_Send(pf_w, n, MPI_DOUBLE, dest, ++tag, comm);
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
#if defined(NEWTONPP_GPU_DIRECT)
        auto alloc = gpu_alloc();
#else
        auto alloc = cpu_alloc();
#endif
        hamr::buffer<double> tpd_m(alloc, n);
        hamr::buffer<double> tpd_x(alloc, n);
        hamr::buffer<double> tpd_y(alloc, n);
        hamr::buffer<double> tpd_z(alloc, n);
        hamr::buffer<double> tpd_u(alloc, n);
        hamr::buffer<double> tpd_v(alloc, n);
        hamr::buffer<double> tpd_w(alloc, n);
        hamr::buffer<int>   tpd_id(alloc, n);
        hamr::buffer<double> tpf_u(alloc, n);
        hamr::buffer<double> tpf_v(alloc, n);
        hamr::buffer<double> tpf_w(alloc, n);

        MPI_Recv(tpd_m.data(), n, MPI_DOUBLE, src, ++tag, comm, MPI_STATUS_IGNORE);
        MPI_Recv(tpd_x.data(), n, MPI_DOUBLE, src, ++tag, comm, MPI_STATUS_IGNORE);
        MPI_Recv(tpd_y.data(), n, MPI_DOUBLE, src, ++tag, comm, MPI_STATUS_IGNORE);
        MPI_Recv(tpd_z.data(), n, MPI_DOUBLE, src, ++tag, comm, MPI_STATUS_IGNORE);
        MPI_Recv(tpd_u.data(), n, MPI_DOUBLE, src, ++tag, comm, MPI_STATUS_IGNORE);
        MPI_Recv(tpd_v.data(), n, MPI_DOUBLE, src, ++tag, comm, MPI_STATUS_IGNORE);
        MPI_Recv(tpd_w.data(), n, MPI_DOUBLE, src, ++tag, comm, MPI_STATUS_IGNORE);
        MPI_Recv(tpd_id.data(),n, MPI_INT,    src, ++tag, comm, MPI_STATUS_IGNORE);
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
        pd.m_id = std::move(tpd_id);
        pf.m_u = std::move(tpf_u);
        pf.m_v = std::move(tpf_v);
        pf.m_w = std::move(tpf_w);
    }
}

// --------------------------------------------------------------------------
void gather(MPI_Comm comm, patch_data &pdo, patch_force &pfo,
    const patch_data &pdi, const patch_force &pfi, int rank,
    int *lengths, int *offsets, long n_total)
{
#if defined(NEWTONPP_GPU_DIRECT)
    auto alloc = gpu_alloc();
#else
    auto alloc = cpu_alloc();
#endif
    hamr::buffer<double> tpd_m(alloc, n_total);
    hamr::buffer<double> tpd_x(alloc, n_total);
    hamr::buffer<double> tpd_y(alloc, n_total);
    hamr::buffer<double> tpd_z(alloc, n_total);
    hamr::buffer<double> tpd_u(alloc, n_total);
    hamr::buffer<double> tpd_v(alloc, n_total);
    hamr::buffer<double> tpd_w(alloc, n_total);
    hamr::buffer<int>   tpd_id(alloc, n_total);
    hamr::buffer<double> tpf_u(alloc, n_total);
    hamr::buffer<double> tpf_v(alloc, n_total);
    hamr::buffer<double> tpf_w(alloc, n_total);

#if defined(NEWTONPP_GPU_DIRECT)
    auto [pdi_m, pdi_x, pdi_y, pdi_z, pdi_u, pdi_v, pdi_w, pdi_id] = pdi.get_data();
    auto [pfi_u, pfi_v, pfi_w] = pfi.get_data();
#else
    auto [spdi_m, pdi_m, spdi_x, pdi_x, spdi_y, pdi_y, spdi_z, pdi_z,
          spdi_u, pdi_u, spdi_v, pdi_v, spdi_w, pdi_w, spdi_id, pdi_id] = pdi.get_host_accessible();

    auto [spfi_u, pfi_u, spfi_v, pfi_v, spfi_w, pfi_w] = pfi.get_host_accessible();
#endif

    int n_in = pdi.size();

    MPI_Gatherv(pdi_m, n_in, MPI_DOUBLE, tpd_m.data(), lengths, offsets, MPI_DOUBLE, rank, comm);
    MPI_Gatherv(pdi_x, n_in, MPI_DOUBLE, tpd_x.data(), lengths, offsets, MPI_DOUBLE, rank, comm);
    MPI_Gatherv(pdi_y, n_in, MPI_DOUBLE, tpd_y.data(), lengths, offsets, MPI_DOUBLE, rank, comm);
    MPI_Gatherv(pdi_z, n_in, MPI_DOUBLE, tpd_z.data(), lengths, offsets, MPI_DOUBLE, rank, comm);
    MPI_Gatherv(pdi_u, n_in, MPI_DOUBLE, tpd_u.data(), lengths, offsets, MPI_DOUBLE, rank, comm);
    MPI_Gatherv(pdi_v, n_in, MPI_DOUBLE, tpd_v.data(), lengths, offsets, MPI_DOUBLE, rank, comm);
    MPI_Gatherv(pdi_w, n_in, MPI_DOUBLE, tpd_w.data(), lengths, offsets, MPI_DOUBLE, rank, comm);
    MPI_Gatherv(pdi_id,n_in, MPI_INT,    tpd_id.data(),lengths, offsets, MPI_INT,    rank, comm);
    MPI_Gatherv(pfi_u, n_in, MPI_DOUBLE, tpf_u.data(), lengths, offsets, MPI_DOUBLE, rank, comm);
    MPI_Gatherv(pfi_v, n_in, MPI_DOUBLE, tpf_v.data(), lengths, offsets, MPI_DOUBLE, rank, comm);
    MPI_Gatherv(pfi_w, n_in, MPI_DOUBLE, tpf_w.data(), lengths, offsets, MPI_DOUBLE, rank, comm);

    pdo.m_m = std::move(tpd_m);
    pdo.m_x = std::move(tpd_x);
    pdo.m_y = std::move(tpd_y);
    pdo.m_z = std::move(tpd_z);
    pdo.m_u = std::move(tpd_u);
    pdo.m_v = std::move(tpd_v);
    pdo.m_w = std::move(tpd_w);
    pdo.m_id = std::move(tpd_id);
    pfo.m_u = std::move(tpf_u);
    pfo.m_v = std::move(tpf_v);
    pfo.m_w = std::move(tpf_w);
}

// --------------------------------------------------------------------------
void gather(MPI_Comm comm, const patch_data &pdi, const patch_force &pfi, int rank)
{
#if defined(NEWTONPP_GPU_DIRECT)
    auto [pdi_m, pdi_x, pdi_y, pdi_z, pdi_u, pdi_v, pdi_w, pdi_id] = pdi.get_data();
    auto [pfi_u, pfi_v, pfi_w] = pfi.get_data();
#else
    auto [spdi_m, pdi_m, spdi_x, pdi_x, spdi_y, pdi_y, spdi_z, pdi_z,
          spdi_u, pdi_u, spdi_v, pdi_v, spdi_w, pdi_w, spdi_id, pdi_id] = pdi.get_host_accessible();

    auto [spfi_u, pfi_u, spfi_v, pfi_v, spfi_w, pfi_w] = pfi.get_host_accessible();
#endif

    int n_in = pdi.size();

    MPI_Gatherv(pdi_m, n_in, MPI_DOUBLE, nullptr, nullptr, nullptr, MPI_DOUBLE, rank, comm);
    MPI_Gatherv(pdi_x, n_in, MPI_DOUBLE, nullptr, nullptr, nullptr, MPI_DOUBLE, rank, comm);
    MPI_Gatherv(pdi_y, n_in, MPI_DOUBLE, nullptr, nullptr, nullptr, MPI_DOUBLE, rank, comm);
    MPI_Gatherv(pdi_z, n_in, MPI_DOUBLE, nullptr, nullptr, nullptr, MPI_DOUBLE, rank, comm);
    MPI_Gatherv(pdi_u, n_in, MPI_DOUBLE, nullptr, nullptr, nullptr, MPI_DOUBLE, rank, comm);
    MPI_Gatherv(pdi_v, n_in, MPI_DOUBLE, nullptr, nullptr, nullptr, MPI_DOUBLE, rank, comm);
    MPI_Gatherv(pdi_w, n_in, MPI_DOUBLE, nullptr, nullptr, nullptr, MPI_DOUBLE, rank, comm);
    MPI_Gatherv(pdi_id,n_in, MPI_INT,    nullptr, nullptr, nullptr, MPI_INT,    rank, comm);
    MPI_Gatherv(pfi_u, n_in, MPI_DOUBLE, nullptr, nullptr, nullptr, MPI_DOUBLE, rank, comm);
    MPI_Gatherv(pfi_v, n_in, MPI_DOUBLE, nullptr, nullptr, nullptr, MPI_DOUBLE, rank, comm);
    MPI_Gatherv(pfi_w, n_in, MPI_DOUBLE, nullptr, nullptr, nullptr, MPI_DOUBLE, rank, comm);
}
