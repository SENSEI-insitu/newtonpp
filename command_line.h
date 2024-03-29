#ifndef command_line_h
#define command_line_h

#include <mpi.h>

int parse_command_line(int argc, char **argv, MPI_Comm comm,
    int &num_devs, int &start_dev, int &dev_stride,
    double &G, double &dt, double &eps, double &theta,
    long &n_its, long &n_bodies, long &part_int, const char *&magi_h5,
    const char *&magi_sum, const char *&out_dir, long &io_int,
    const char *&is_conf, long &is_int);

#endif
