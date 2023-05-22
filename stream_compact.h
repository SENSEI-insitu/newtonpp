#ifndef stream_compact_h
#define stream_compact_h

namespace cuda
{
/** The algorithm is implemented using the 2-pass scan algorithm, counting the
 * true predicates with a reduction pass; scanning the predicates; then passing
 * over the data again, evaluating the predicates again and using the scanned
 * predicate values as indices to write the output for which the predicate is
 * true.
 *
 * @param[out] pdo_X/pfo_X compacted arrays (device)
 * @param[out] outCount the length of the compacted arrays (host)
 * @param[in] pdi_X/pfi_X arrays of size N to be compacted (device)
 * @param[in] mask flags that are set to 1 where the input should be copied and zero elsewhere (device)
 * @param[in] nIn the number of arrays to compact (host)
 * @param[in] N the length of the arrays to be compacted (host)
 * @param[in] b the number of threads per block (host)
 * @returns zero if successful
 */
int stream_compact(
    double *pdo_m, double *pdo_x, double *pdo_y, double *pdo_z,
    double *pdo_u, double *pdo_v, double *pdo_w, int *pdo_id,
    double *pfo_u, double *pfo_v, double *pfo_w,
    int &outCount,
    const double *pdi_m, const double *pdi_x, const double *pdi_y, const double *pdi_z,
    const double *pdi_u, const double *pdi_v, const double *pdi_w, const int *pdi_id,
    const double *pfi_u, const double *pfi_v, const double *pfi_w,
    const int *mask, size_t N, int b );
}

namespace cpu
{

int stream_compact(
    double *pdo_m, double *pdo_x, double *pdo_y, double *pdo_z,
    double *pdo_u, double *pdo_v, double *pdo_w, int *pdo_id,
    double *pfo_u, double *pfo_v, double *pfo_w,
    int &outCount,
    const double *pdi_m, const double *pdi_x, const double *pdi_y, const double *pdi_z,
    const double *pdi_u, const double *pdi_v, const double *pdi_w, const int *pdi_id,
    const double *pfi_u, const double *pfi_v, const double *pfi_w,
    const int *mask, size_t N );
}


#endif
