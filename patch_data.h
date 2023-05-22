#ifndef newton_patch_data_h
#define newton_patch_data_h

#include "memory_management.h"

/// a collection of bodies associated with a spatial patch
struct patch_data
{
    patch_data(hamr::buffer_allocator alloc = def_alloc());

    patch_data(const patch_data&) = delete;
    patch_data(patch_data&&) = delete;

    ~patch_data();

    void operator=(const patch_data &pd);
    void operator=(patch_data &&pd);

    /// returns the number of bodies
    long size() const { return m_m.size(); }

    /// set the number of bodies
    void resize(long n);

    /// add bodies at the end
    void append(const patch_data &o);

    /// read only access to mass and position data
    auto get_mp_cpu_accessible() const { return hamr::get_cpu_accessible(m_m, m_x, m_y, m_z); }
    auto get_mp_openmp_accessible() const { return hamr::get_openmp_accessible(m_m, m_x, m_y, m_z); }

    /// write access to mass and position data
    auto get_mp_data() { return  hamr::data(m_m, m_x, m_y, m_z); }
    auto get_mp_data() const { return  hamr::data(m_m, m_x, m_y, m_z); }

    /// read only access to data
    auto get_cpu_accessible() const {
        return std::tuple_cat(hamr::get_cpu_accessible(m_m, m_x, m_y, m_z, m_u, m_v, m_w),
             hamr::get_cpu_accessible(m_id)); }

    auto get_openmp_accessible() const {
        return std::tuple_cat(hamr::get_openmp_accessible(m_m, m_x, m_y, m_z, m_u, m_v, m_w),
            hamr::get_openmp_accessible(m_id)); }

    /// write access to data
    auto get_data() {
        return std::tuple_cat(hamr::data(m_m, m_x, m_y, m_z, m_u, m_v, m_w),
            hamr::data(m_id)); }

    auto get_data() const {
        return std::tuple_cat(hamr::data(m_m, m_x, m_y, m_z, m_u, m_v, m_w),
            hamr::data(m_id)); }

    hamr::buffer<double> m_m; ///< body mass
    hamr::buffer<double> m_x; ///< body position x
    hamr::buffer<double> m_y; ///< body position y
    hamr::buffer<double> m_z; ///< body position y
    hamr::buffer<double> m_u; ///< body velocity x
    hamr::buffer<double> m_v; ///< body velocity y
    hamr::buffer<double> m_w; ///< body velocity y
    hamr::buffer<int> m_id;   ///< body type or id
};

/// reduce the input to a single body located at its center of mass
void reduce(const patch_data &pdi, patch_data &pdo);

#endif
