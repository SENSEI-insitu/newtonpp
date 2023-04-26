#include "insitu.h"

#include "sensei_adaptor.h"


// --------------------------------------------------------------------------
int init_insitu(MPI_Comm comm, const char *is_conf, insitu_data &is_data)
{
    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    is_data.m_analysis = sensei::ConfigurableAnalysis::New();

    if (is_data.m_analysis->Initialize(is_conf))
    {
        if (rank == 0)
            std::cerr << "Failed to initialize the in-situ backend" << std::endl;

        is_data.m_analysis->Delete();
        is_data.m_analysis = nullptr;

        return -1;
    }

    is_data.m_data = sensei_adaptor::New();

    if (rank == 0)
        std::cerr << "Initialized the in-situ backend" << std::endl;

    return 0;
}

// --------------------------------------------------------------------------
int update_insitu(MPI_Comm comm, insitu_data &is_data,
  long step, double time, const std::vector<patch> &patches,
  const patch_data &pd, const patch_force &pf)
{
    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    is_data.m_data->SetData(step, time, patches, pd, pf);

    int ierr = 0;
    if (!is_data.m_analysis->Execute(is_data.m_data, nullptr))
        ierr = -1;

    if (rank == 0)
    {
        if (ierr)
            std::cerr << "Failed to update the in-situ backend" << std::endl;
        else
            std::cerr << "Updated the in-situ backend" << std::endl;
    }

    return ierr;
}

// --------------------------------------------------------------------------
int finalize_insitu(MPI_Comm comm, insitu_data &is_data)
{
    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    int ierr = 0;

    if (is_data.m_analysis->Finalize())
        ierr = -1;

    if (rank == 0)
    {
        if (ierr)
            std::cerr << "Failed to finalize the in-situ backend" << std::endl;
        else
            std::cerr << "Finalized the in-situ backend" << std::endl;
    }

    is_data.m_analysis->Delete();
    is_data.m_data->Delete();

    return ierr;
}
