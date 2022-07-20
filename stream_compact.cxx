#include <cstddef>

namespace cpu
{
// --------------------------------------------------------------------------
int stream_compact( double *pdo_m, double *pdo_x, double *pdo_y,
    double *pdo_u, double *pdo_v, double *pfo_u, double *pfo_v,
    int &outCount, const double *pdi_m, const double *pdi_x,
    const double *pdi_y, const double *pdi_u, const double *pdi_v,
    const double *pfi_u, const double *pfi_v, const int *mask,
    size_t N )
{
    size_t q = 0;
    for (size_t i = 0; i < N; ++i)
    {
        if (mask[i])
        {
            pdo_m[q] = pdi_m[i];
            pdo_x[q] = pdi_x[i];
            pdo_y[q] = pdi_y[i];
            pdo_u[q] = pdi_u[i];
            pdo_v[q] = pdi_v[i];

            pfo_u[q] = pfi_u[i];
            pfo_v[q] = pfi_v[i];

            ++q;
        }
    }

    outCount = q;
    return 0;
}
}
