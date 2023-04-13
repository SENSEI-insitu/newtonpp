#ifndef memory_management_h
#define memory_management_h

#include "hamr_buffer.h"
#include "hamr_buffer_util.h"

inline constexpr hamr::buffer_allocator cpu_alloc() { return hamr::buffer_allocator::malloc; }
inline constexpr hamr::buffer_allocator gpu_alloc() { return hamr::buffer_allocator::openmp; }

#if defined(ENABLE_OMP)
inline constexpr hamr::buffer_allocator def_alloc() { return gpu_alloc(); }
#else
inline constexpr hamr::buffer_allocator def_alloc() { return cpu_alloc(); }
#endif

#endif
