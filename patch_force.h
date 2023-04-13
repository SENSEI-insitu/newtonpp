#ifndef patch_force_h
#define patch_force_h

#include "memory_management.h"


/// the forces on a set of bodies
struct patch_force
{
    patch_force(hamr::buffer_allocator alloc = def_alloc());

    patch_force(const patch_force&) = delete;
    patch_force(patch_force&&) = delete;

    ~patch_force();

    void operator=(const patch_force &pd);
    void operator=(patch_force &&pd);

    long size() const { return m_u.size(); }

    void resize(long n);
    void append(const patch_force &o);

    auto get_cpu_accessible() const { return hamr::get_cpu_accessible(m_u, m_v, m_w); }
    auto get_openmp_accessible() const { return hamr::get_openmp_accessible(m_u, m_v, m_w); }

    auto get_data() { return  hamr::data(m_u, m_v, m_w); }

    hamr::buffer<double> m_u;   ///< body force x
    hamr::buffer<double> m_v;   ///< body force y
    hamr::buffer<double> m_w;   ///< body force z
};

#endif
