#include <cstddef>

namespace cpu
{
// --------------------------------------------------------------------------
int stream_compact(
    double *pdo_m, double *pdo_x, double *pdo_y, double *pdo_z,
    double *pdo_u, double *pdo_v, double *pdo_w,
    double *pfo_u, double *pfo_v, double *pfo_w,
    int &outCount,
    const double *pdi_m, const double *pdi_x, const double *pdi_y, const double *pdi_z,
    const double *pdi_u, const double *pdi_v, const double *pdi_w,
    const double *pfi_u, const double *pfi_v, const double *pfi_w,
    const int *mask, size_t N )
{
    size_t q = 0;
    for (size_t i = 0; i < N; ++i)
    {
        if (mask[i])
        {
            pdo_m[q] = pdi_m[i];
            pdo_x[q] = pdi_x[i];
            pdo_y[q] = pdi_y[i];
            pdo_z[q] = pdi_z[i];
            pdo_u[q] = pdi_u[i];
            pdo_v[q] = pdi_v[i];
            pdo_w[q] = pdi_w[i];

            pfo_u[q] = pfi_u[i];
            pfo_v[q] = pfi_v[i];
            pfo_w[q] = pfi_w[i];

            ++q;
        }
    }

    outCount = q;
    return 0;
}
}
