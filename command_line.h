#ifndef command_line_h
#define command_line_h

#include <mpi.h>

int parse_command_line(int argc, char **argv, MPI_Comm comm,
    double &G, double &dt, double &eps, double &theta,
    long &n_its, long &n_bodies, const char *&magi_h5,
    const char *&magi_sum, const char *&out_dir, long &io_int,
    const char *&is_conf, long &is_int);

#endif
