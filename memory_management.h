#ifndef memory_management_h
#define memory_management_h

#include "hamr_config.h"
#include "hamr_buffer.h"
#include "hamr_buffer_util.h"

#if defined(HAMR_ENABLE_CUDA)
inline constexpr hamr::buffer_allocator cpu_alloc() { return hamr::buffer_allocator::cuda_host; }
#else
inline constexpr hamr::buffer_allocator cpu_alloc() { return hamr::buffer_allocator::malloc; }
#endif

inline constexpr hamr::buffer_allocator gpu_alloc() { return hamr::buffer_allocator::openmp; }

#if defined(NEWTONPP_ENABLE_OMP)
inline constexpr hamr::buffer_allocator def_alloc() { return gpu_alloc(); }
#else
inline constexpr hamr::buffer_allocator def_alloc() { return cpu_alloc(); }
#endif

#endif
