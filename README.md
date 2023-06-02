## Newton++

An MPI + OpenMP offload parallel n-body code written in C++.
The code currently implements a direct method but we are working on a tree based method.
Platofmr portable data struictures and memory management is handled by [HAMR](https://github.com/LBL-EESA/HAMR).
The code can write its state to disk in [VTK](https://www.vtk.org) format for visualization and is instrumented with [SENSEI](https://github.com/SENSEI-insitu/SENSEI) for in situ visualziation.
The unit system is very flexible and can be specified on the command line.
The code can ingest initial conditions in the HDF5 format defined by [MAGI](https://academic.oup.com/mnras/article/475/2/2269/4791573).
A simple uniform random initial condition is built in and lets the code run with no dependencies for testing and validation.

<table>
<tr><td>
<img src="s15_4_gpu_mass.1250.png" width="400px"/>
</td><td>
 <img src="s15_4_gpu_rank.1250.png" width="400px"/>
</td> </tr>
</table>

## Compiling

The build system is GNU make based.  Make files exist for different compilers
and systems. Select one of the Makefiles and modify for your system.

### Debendencies

**required:** MPI, HAMR <br>
**optional:** OpenMP, CUDA, HDF5, SENSEI <br>

It is recommended to use OpenMP offload as the CPU implemnentation is not full optimized.

### Makefiles

| Makefile | Status |
| -------- | ------ |
| Makefile.debug | GCC build for the CPU only. This is a good start |
| Makefile.nvidia | NVIDIA HPC SDK build for OpenMP offlaod and CUDA. |
| Makefile.clang15 | Mainline clang build for OpenMP offload and CUDA. |

Other makefiles found in the repo may or may not work and be maintained.

### Makefile variables

| Option | Description |
| ------ | ----------- |
| HAMR | The path to the top level directory of the HAMR install. Required. |
| CUDA | The path to the CUDA library directory. |
| USE_SENSEI | Enables the SENSEI in situ instrumentation. Optional. |
| SENSEI | The path to the top level directory (prefix) of the SENSEI install. Optional. |
| USE_MAGI | Enables the MAGI HDF5 reader. Optional. |
| HDF5_FLAGS | Sets the path for HDF5 incudes. Optional. |
| HDF5_LINK | Sets the path for HDF5 libraries. Optional. |

### Preprocessor definitions

| Defiintion | Description |
| ---------- | ----------- |
| `-DNEWTONPP_ENABLE_OMP` | Enables OpenMP offload. |
| `-DNEWTONPP_ENABLE_CUDA` | Enable CUDA optimized stream compact algorithm. OpenMP must also be enabled. |
| `-DNEWTONPP_ENABLE_MAGI` | Enable MAGI reader for initial conditions. |


## Running

### newtonpp command line arguments:

| Option | Description |
| ------ | ----------- |
| --G | gravitational constant|
| --dt | time step size|
| --eps | softening length|
| --theta | threshold for reduced representation|
| --n_its | how many iterations to perform|
| --n_bodies | the total number of bodies|
| --magi_h5 | MAGI file with particle positions|
| --magi_sum | MAGI file with component sizes|
| --out_dir | where to write the results|
| --out_int | how often to write results|
| --sensei_xml | a sensei configuration file|
| --sensei_int | how often to invoke in situ|

### Using the built in uniform random initial condition
```
mpiexec -np 4 ./newtonpp_clang15_omp --n_its 1000 --out_dir ruic_out --out_int 10 --n_bodies 10000
```


### Using a MAGI initial condition
```
mpiexec -np 4 ./newtonpp_clang15_omp --magi_file /work/SENSEI/magi/bin/dat/s15.tmp0.h5 --G 4.498466e+00 --dt 0.00001 --n_its 100000000 --out_dir s15_4_gpu --out_int 1000 --eps 0.0
```
