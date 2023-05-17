#include "patch.h"
#include "patch_data.h"
#include "memory_management.h"

#include <hdf5.h>
#include <mpi.h>
#include <limits>

// --------------------------------------------------------------------------
int check_err(int rank, const char *desc, herr_t ret)
{
    if (ret < 0)
    {
        if (rank == 0)
        {
            std::cerr << "Operation failed: " << desc << std::endl;
            //H5Eprint2(H5E_DEFAULT, nullptr);
        }
        return -1;
    }
    return 0;
}

// --------------------------------------------------------------------------
int check_hid(int rank, const char *desc, hid_t handle)
{
    if (handle == H5I_INVALID_HID)
    {
        if (rank == 0)
        {
            std::cerr << "Operation failed: " << desc << std::endl;
            //H5Eprint2(H5E_DEFAULT, nullptr);
        }
        return -1;
    }
    return 0;
}

/** this reads data generated by MAGI (MAny-component Galaxy Initializer)
 * https://bitbucket.org/ymiki/magi/src/master/
 * cmake -DOMP_THREADS=8 -DUSE_HDF5=ON -DUSE_TIPSY_FORMAT=OFF ..
 * edit and use run.sh to generate the IC, it it located in dat/name.tmp0.h5
 * in doc/unit.txt find G for these computational units.
 */
// --------------------------------------------------------------------------
int read_magi(MPI_Comm comm, const char *file, patch &dom, patch_data &pd)
{
    int rank = 0;
    int n_ranks = 1;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &n_ranks);

    // open the file
    hid_t fh = H5Fopen(file, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (check_hid(rank, "open file", fh))
        return -1;

    // open the main group
    hid_t gh = H5Gopen(fh, "/nbody", H5P_DEFAULT);
    if (check_hid(rank, "open group /nbody", gh))
        return -1;

    // get the number of bodies
    hid_t lenh = H5Aopen(gh, "number", H5P_DEFAULT);
    if (check_hid(rank, "open attribute /nbody/number", lenh))
        return -1;

    long nb_total = 0;
    if (check_err(rank, "read attribute /nbody/number",
        H5Aread(lenh, H5T_NATIVE_LONG, &nb_total)))
        return -1;

    H5Aclose(lenh);

    if (rank == 0)
        std::cerr << " === total number of particles loaded " << nb_total << std::endl;

    // split particle data among the ranks. this won't have the correct spatial
    // decomposition so the data will also need to be partitioned later
    long bsz = nb_total / n_ranks;
    long nlg = nb_total % n_ranks;
    long nb_local = bsz + (rank < nlg ? 1 : 0);

    hsize_t start[1] = {0};
    hsize_t block[1] = {0};
    hsize_t stride[1] = {1};
    hsize_t count[1] = {1};

    start[0] = 4 *( bsz * rank + (rank < nlg ? rank : nlg) );
    block[0] = 4 * nb_local;

    // read mass and position
    hid_t posh = H5Dopen(gh, "position", H5P_DEFAULT);
    if (check_hid(rank, "open dataset /nbody/position", posh))
        return -1;

    hid_t poss = H5Dget_space(posh);
    if (check_hid(rank, "get space /nbody/position", poss))
        return -1;

    if (check_err(rank, "select hyperslab /nbody/position",
        H5Sselect_hyperslab(poss, H5S_SELECT_SET, start, stride, count, block)))
        return -1;

    hamr::buffer<double> tmp(cpu_alloc(), block[0]);

    hid_t mems = H5Screate_simple(1, block, nullptr);
    if (check_hid(rank, "create the mass and pos memspace", mems))
        return -1;

    if (check_err(rank, "read /nbody/position",
        H5Dread(posh, H5T_NATIVE_DOUBLE, mems, poss, H5P_DEFAULT, tmp.data())))
        return -1;

    H5Sclose(poss);
    H5Dclose(posh);
    H5Sclose(mems);

    // unpack mass and position
    hamr::buffer<double> tmp_m(cpu_alloc(), nb_local);
    hamr::buffer<double> tmp_x(cpu_alloc(), nb_local);
    hamr::buffer<double> tmp_y(cpu_alloc(), nb_local);
    hamr::buffer<double> tmp_z(cpu_alloc(), nb_local);

    for (long i = 0; i < nb_local; ++i)
        tmp_x.data()[i] = tmp.data()[4*i    ];

    for (long i = 0; i < nb_local; ++i)
        tmp_y.data()[i] = tmp.data()[4*i + 1];

    for (long i = 0; i < nb_local; ++i)
        tmp_z.data()[i] = tmp.data()[4*i + 2];

    for (long i = 0; i < nb_local; ++i)
        tmp_m.data()[i] = tmp.data()[4*i + 3];

    // get the local bounding box
    double mnx = std::numeric_limits<double>::max();
    double mxx = std::numeric_limits<double>::lowest();
    for (long i = 0; i < nb_local; ++i)
    {
        mnx = std::min(mnx, tmp_x.data()[i]);
        mxx = std::max(mxx, tmp_x.data()[i]);
    }
    double mny = std::numeric_limits<double>::max();
    double mxy = std::numeric_limits<double>::lowest();
    for (long i = 0; i < nb_local; ++i)
    {
        mny = std::min(mny, tmp_y.data()[i]);
        mxy = std::max(mxy, tmp_y.data()[i]);
    }
    double mnz = std::numeric_limits<double>::max();
    double mxz = std::numeric_limits<double>::lowest();
    for (long i = 0; i < nb_local; ++i)
    {
        mnz = std::min(mnz, tmp_z.data()[i]);
        mxz = std::max(mxz, tmp_z.data()[i]);
    }

    // compute the global bounding box
    MPI_Request reqs[6];
    MPI_Iallreduce(MPI_IN_PLACE, &mnx, 1, MPI_DOUBLE, MPI_MIN, comm, reqs  );
    MPI_Iallreduce(MPI_IN_PLACE, &mny, 1, MPI_DOUBLE, MPI_MIN, comm, reqs+1);
    MPI_Iallreduce(MPI_IN_PLACE, &mnz, 1, MPI_DOUBLE, MPI_MIN, comm, reqs+2);
    MPI_Iallreduce(MPI_IN_PLACE, &mxx, 1, MPI_DOUBLE, MPI_MAX, comm, reqs+3);
    MPI_Iallreduce(MPI_IN_PLACE, &mxy, 1, MPI_DOUBLE, MPI_MAX, comm, reqs+4);
    MPI_Iallreduce(MPI_IN_PLACE, &mxz, 1, MPI_DOUBLE, MPI_MAX, comm, reqs+5);

    // read velocity
    hamr::buffer<double> tmp_u(cpu_alloc(), nb_local);
    hamr::buffer<double> tmp_v(cpu_alloc(), nb_local);
    hamr::buffer<double> tmp_w(cpu_alloc(), nb_local);

    start[0] = bsz * rank + (rank < nlg ? rank : nlg);
    block[0] = nb_local;

    mems = H5Screate_simple(1, block, nullptr);
    if (check_hid(rank, "create the vel memspace", mems))
        return -1;

    // vx
    hid_t vxh = H5Dopen(gh, "vx", H5P_DEFAULT);
    if (check_hid(rank, "open dataset /nbody/vx", vxh))
        return -1;

    hid_t vxs = H5Dget_space(vxh);
    if (check_hid(rank, "get space /nbody/vx", vxs))
        return -1;

    if (check_err(rank, "select hyperslab /nbody/vx",
        H5Sselect_hyperslab(vxs, H5S_SELECT_SET, start, stride, count, block)))
        return -1;

    if (check_err(rank, "read /nbody/vx",
        H5Dread(vxh, H5T_NATIVE_DOUBLE, mems, vxs, H5P_DEFAULT, tmp_u.data())))
        return -1;

    H5Sclose(vxs);
    H5Dclose(vxh);

    // vy
    hid_t vyh = H5Dopen(gh, "vy", H5P_DEFAULT);
    if (check_hid(rank, "open dataset /nbody/vy", vyh))
        return -1;

    hid_t vys = H5Dget_space(vyh);
    if (check_hid(rank, "get space /nbody/vy", vys))
        return -1;

    if (check_err(rank, "select hyperslab /nbody/vy",
        H5Sselect_hyperslab(vys, H5S_SELECT_SET, start, stride, count, block)))
        return -1;

    if (check_err(rank, "read /nbody/vy",
        H5Dread(vyh, H5T_NATIVE_DOUBLE, mems, vys, H5P_DEFAULT, tmp_v.data())))
        return -1;

    H5Sclose(vys);
    H5Dclose(vyh);

    // vz
    hid_t vzh = H5Dopen(gh, "vz", H5P_DEFAULT);
    if (check_hid(rank, "open dataset /nbody/vz", vzh))
        return -1;

    hid_t vzs = H5Dget_space(vzh);
    if (check_hid(rank, "get space /nbody/vz", vzs))
        return -1;

    if (check_err(rank, "select hyperslab /nbody/vz",
        H5Sselect_hyperslab(vzs, H5S_SELECT_SET, start, stride, count, block)))
        return -1;

    if (check_err(rank, "read /nbody/vz",
        H5Dread(vzh, H5T_NATIVE_DOUBLE, mems, vzs, H5P_DEFAULT, tmp_w.data())))
        return -1;

    H5Sclose(vzs);
    H5Dclose(vzh);

    H5Sclose(mems);

    H5Gclose(gh);
    H5Fclose(fh);

    // finalize the global bounding box
    MPI_Waitall(6, reqs, MPI_STATUSES_IGNORE);

    // grow the box a bit
    double dx = mxx - mnx;
    double dy = mxy - mny;
    double dz = mxz - mnz;

    hamr::buffer<double> tmp_dom(cpu_alloc(), 6);

    tmp_dom.data()[0] = mnx - dx / 8.;
    tmp_dom.data()[1] = mxx + dx / 8.;
    tmp_dom.data()[2] = mny - dy / 8.;
    tmp_dom.data()[3] = mxy + dy / 8.;
    tmp_dom.data()[4] = mnz - dz / 8.;
    tmp_dom.data()[5] = mxz + dz / 8.;

    dom.m_x = std::move(tmp_dom);

    // pass back local data
    pd.m_m = std::move(tmp_m);
    pd.m_x = std::move(tmp_x);
    pd.m_y = std::move(tmp_y);
    pd.m_z = std::move(tmp_z);
    pd.m_u = std::move(tmp_u);
    pd.m_v = std::move(tmp_v);
    pd.m_w = std::move(tmp_w);

    return 0;
}