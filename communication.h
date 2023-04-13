#ifndef communication_h
#define communication_h

#include "patch_data.h"
#include "patch_force.h"

#include <mpi.h>
#include <memory>
#include <vector>

/// a collection of MPI requests from non-blocking comminucation
struct requests
{
    requests() : m_size(0) {}

    requests(const requests &) = delete;
    void operator=(const requests &) = delete;

    int m_size;                                        ///< number of requests actually made
    MPI_Request m_req[11];                             ///< the requests (11 needed to send all fields and size)
    std::vector<std::shared_ptr<const double>> m_data; ///< keeps buffers alive durring communication
};

/// send/receive patch data's mass and position
void isend_mp(MPI_Comm comm, const patch_data &pd, int dest, int tag, requests &reqs);
void recv_mp(MPI_Comm comm, patch_data &pd, int src, int tag);

/// send/receive per-body forces
void isend(MPI_Comm comm, const patch_force &pf, int dest, int tag);
void recv(MPI_Comm comm, patch_force &pf, int src, int tag);

/// send/receive bodies and per-body forces
void isend(MPI_Comm comm, const patch_data &pd, const patch_force &pf, int dest, int tag, requests &reqs);
void recv(MPI_Comm comm, patch_data &pd, patch_force &pf, int src, int tag);

#endif
