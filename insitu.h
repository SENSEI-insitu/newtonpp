#ifndef insitu_h
#define insitu_h

#include "patch.h"
#include "patch_data.h"
#include "patch_force.h"
#include "sensei_adaptor.h"

#include <ConfigurableAnalysis.h>
#include <vector>
#include <mpi.h>

struct insitu_data
{
    insitu_data() : m_data(nullptr), m_analysis(nullptr) {}

    operator bool () const { return m_analysis; }

    sensei_adaptor *m_data;
    sensei::ConfigurableAnalysis *m_analysis;
};

int init_insitu(MPI_Comm comm, const char *is_conf, insitu_data &is_data);

int update_insitu(MPI_Comm comm, insitu_data &is_data,
    long step, double time, const std::vector<patch> &patches,
    const patch_data &pd, const patch_force &pf);

int finalize_insitu(MPI_Comm comm, insitu_data &is_data);

#endif
