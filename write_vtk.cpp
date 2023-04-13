#include "write_vtk.h"

#include "memory_management.h"

#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <cstring>

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
void write_vtk(MPI_Comm comm, const patch_data &pd, const patch_force &pf, const char *dir)
{
    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    auto [spd_m, pd_m,
          spd_x, pd_x, spd_y, pd_y, spd_z, pd_z,
          spd_u, pd_u, spd_v, pd_v, spd_w, pd_w] = pd.get_cpu_accessible();

    auto [spf_u, pf_u, spf_v, pf_v, spf_w, pf_w] = pf.get_cpu_accessible();

    // package in the vtk layout
    long n = pd.size();

    std::vector<double> x(3*n);
    std::vector<double> m(n);
    std::vector<double> vu(n);
    std::vector<double> vv(n);
    std::vector<double> vw(n);
    std::vector<double> fu(n);
    std::vector<double> fv(n);
    std::vector<double> fw(n);
    std::vector<int> r(n);
    std::vector<int> id(2*n);

    for (long i = 0; i < n; ++i)
    {
        long ii = 3*i;
        x[ii    ] = pd_x[i];
        x[ii + 1] = pd_y[i];
        x[ii + 2] = pd_z[i];

        m[i] = pd_m[i];
        vu[i] = pd_u[i];
        vv[i] = pd_v[i];
        vw[i] = pd_w[i];
        fu[i] = pf_u[i];
        fv[i] = pf_v[i];
        fw[i] = pf_w[i];
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

    uint64_t *pvw = (uint64_t*)vw.data();
    for (size_t i = 0; i < vw.size(); ++i)
        pvw[i] = __builtin_bswap64(pvw[i]);

    uint64_t *pfu = (uint64_t*)fu.data();
    for (size_t i = 0; i < fu.size(); ++i)
        pfu[i] = __builtin_bswap64(pfu[i]);

    uint64_t *pfv = (uint64_t*)fv.data();
    for (size_t i = 0; i < vv.size(); ++i)
        pfv[i] = __builtin_bswap64(pfv[i]);

    uint64_t *pfw = (uint64_t*)fw.data();
    for (size_t i = 0; i < vw.size(); ++i)
        pfw[i] = __builtin_bswap64(pfw[i]);

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

    fprintf(fh, "SCALARS vw double 1\n"
                "LOOKUP_TABLE default\n");

    fwrite(vw.data(), sizeof(double), vw.size(), fh);

    fprintf(fh, "SCALARS fu double 1\n"
                "LOOKUP_TABLE default\n");

    fwrite(fu.data(), sizeof(double), fu.size(), fh);

    fprintf(fh, "SCALARS fv double 1\n"
                "LOOKUP_TABLE default\n");

    fwrite(fv.data(), sizeof(double), fv.size(), fh);

    fprintf(fh, "SCALARS fw double 1\n"
                "LOOKUP_TABLE default\n");

    fwrite(fw.data(), sizeof(double), fw.size(), fh);

    fclose(fh);
}

// --------------------------------------------------------------------------
void write_vtk(MPI_Comm comm, const std::vector<patch> &patches, const char *dir)
{
    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    if (rank != 0)
        return;

    // package in the vtk layout
    long n = patches.size();
    std::vector<double> x(24*n);
    std::vector<int> id(9*n);
    std::vector<int> ct(n);
    std::vector<int> r(n);

    for (long i = 0; i < n; ++i)
    {
        const patch &pi = patches[i];

        auto [spi_x, pi_x] = hamr::get_cpu_accessible(pi.m_x);

        long ii = 24*i;
        x[ii    ] = pi_x[0];
        x[ii + 1] = pi_x[2];
        x[ii + 2] = pi_x[4];

        ii += 3;
        x[ii    ] = pi_x[1];
        x[ii + 1] = pi_x[2];
        x[ii + 2] = pi_x[4];

        ii += 3;
        x[ii    ] = pi_x[1];
        x[ii + 1] = pi_x[3];
        x[ii + 2] = pi_x[4];

        ii += 3;
        x[ii    ] = pi_x[0];
        x[ii + 1] = pi_x[3];
        x[ii + 2] = pi_x[4];

        ii += 3;
        x[ii    ] = pi_x[0];
        x[ii + 1] = pi_x[2];
        x[ii + 2] = pi_x[5];

        ii += 3;
        x[ii    ] = pi_x[1];
        x[ii + 1] = pi_x[2];
        x[ii + 2] = pi_x[5];

        ii += 3;
        x[ii    ] = pi_x[1];
        x[ii + 1] = pi_x[3];
        x[ii + 2] = pi_x[5];

        ii += 3;
        x[ii    ] = pi_x[0];
        x[ii + 1] = pi_x[3];
        x[ii + 2] = pi_x[5];

        ii = 9*i;
        long pid = 8*i;
        id[ii    ] = 8;
        id[ii + 1] = pid;
        id[ii + 2] = pid + 1;
        id[ii + 3] = pid + 2;
        id[ii + 4] = pid + 3;
        id[ii + 5] = pid + 4;
        id[ii + 6] = pid + 5;
        id[ii + 7] = pid + 6;
        id[ii + 8] = pid + 7;

        ct[i] = 12; // VTK_HEXAHEDRON

        r[i] = i;
    }

    // convert to big endian (required by vtk)
    uint64_t *px = (uint64_t*)x.data();
    for (long i = 0; i < 24*n; ++i)
        px[i] = __builtin_bswap64(px[i]);

    uint32_t *pid = (uint32_t*)id.data();
    for (long i = 0; i < 9*n; ++i)
        pid[i] = __builtin_bswap32(pid[i]);

    uint32_t *pct = (uint32_t*)ct.data();
    for (long i = 0; i < n; ++i)
        pct[i] = __builtin_bswap32(pct[i]);

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
                "DATASET UNSTRUCTURED_GRID\n"
                "POINTS %ld double\n", 8*n);

    fwrite(x.data(), sizeof(double), x.size(), fh);

    fprintf(fh, "CELLS %ld %ld\n", n, 9*n);

    fwrite(id.data(), sizeof(int), id.size(), fh);

    fprintf(fh, "CELL_TYPES %ld\n", n);

    fwrite(ct.data(), sizeof(int), ct.size(), fh);

    fprintf(fh, "CELL_DATA %ld\n"
                "SCALARS rank int 1\n"
                "LOOKUP_TABLE default\n", n);

    fwrite(r.data(), sizeof(int), r.size(), fh);

    fclose(fh);
}
