#ifndef patch_h
#define patch_h

#include "memory_management.h"

#include <vector>
#include <iostream>

/// a spatial patch
struct patch
{
    ~patch() {}
    patch() : m_owner(-1), m_x(def_alloc(), 6, 0.) {}

    /// construct a new patch
    patch(int owner, double x0, double x1, double y0, double y1, double z0, double z1);

    patch(const patch &p);
    void operator=(const patch &p);

    int m_owner;              ///< the MPI rank to which the patch belongs
    hamr::buffer<double> m_x; ///< bounding box defining a spatial patch
};

/// sends a patch to the output stream
std::ostream &operator<<(std::ostream &os, const patch &p);

#endif
