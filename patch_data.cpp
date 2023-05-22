#include "patch_data.h"
#include <iostream>

// --------------------------------------------------------------------------
patch_data::patch_data(hamr::buffer_allocator alloc) : m_m(alloc),
    m_x(alloc), m_y(alloc), m_z(alloc), m_u(alloc), m_v(alloc), m_w(alloc),
    m_id(alloc)
{
    #ifdef DEBUG
    std::cerr << "patch_data::patch_data " << this << std::endl;
    #endif
}

// --------------------------------------------------------------------------
patch_data::~patch_data()
{
    #ifdef DEBUG
    std::cerr << "patch_data::~patch_data " << this << std::endl;
    #endif
}

// --------------------------------------------------------------------------
void patch_data::operator=(const patch_data &pd)
{
    #ifdef DEBUG
    std::cerr << "patch_data::operator= " << this << " <-- " << &pd << std::endl;
    #endif

    m_m.assign(pd.m_m);
    m_x.assign(pd.m_x);
    m_y.assign(pd.m_y);
    m_z.assign(pd.m_z);
    m_u.assign(pd.m_u);
    m_v.assign(pd.m_v);
    m_w.assign(pd.m_w);
    m_id.assign(pd.m_id);
}

// --------------------------------------------------------------------------
void patch_data::operator=(patch_data &&pd)
{
    #ifdef DEBUG
    std::cerr << "patch_data::operator= && " << this << " <-- " << &pd << std::endl;
    #endif

    m_m = std::move(pd.m_m);
    m_x = std::move(pd.m_x);
    m_y = std::move(pd.m_y);
    m_z = std::move(pd.m_z);
    m_u = std::move(pd.m_u);
    m_v = std::move(pd.m_v);
    m_w = std::move(pd.m_w);
    m_id = std::move(pd.m_id);
}

// --------------------------------------------------------------------------
void patch_data::resize(long n)
{
    #ifdef DEBUG
    std::cerr << "patch_data::resize " << this << std::endl;
    #endif

    m_m.resize(n);
    m_x.resize(n);
    m_y.resize(n);
    m_z.resize(n);
    m_u.resize(n);
    m_v.resize(n);
    m_w.resize(n);
    m_id.resize(n);
}

// --------------------------------------------------------------------------
void patch_data::append(const patch_data &o)
{
    #ifdef DEBUG
    std::cerr << "patch_data::append " << this << std::endl;
    #endif

    m_m.append(o.m_m);
    m_x.append(o.m_x);
    m_y.append(o.m_y);
    m_z.append(o.m_z);
    m_u.append(o.m_u);
    m_v.append(o.m_v);
    m_w.append(o.m_w);
    m_id.append(o.m_id);
}

// --------------------------------------------------------------------------
void reduce(const patch_data &pdi, patch_data &pdo)
{
    const double *mi = pdi.m_m.data();
    const double *xi = pdi.m_x.data();
    const double *yi = pdi.m_y.data();
    const double *zi = pdi.m_z.data();

    pdo.resize(1);
    double *mo = pdo.m_m.data();
    double *xo = pdo.m_x.data();
    double *yo = pdo.m_y.data();
    double *zo = pdo.m_z.data();

    long n = pdi.size();

    double m, x, y, z;
    #pragma omp target enter data map(alloc: m,x,y,z)

    #pragma omp target map(alloc: m,x,y,z)
    {
    m = 0.;
    x = 0.;
    y = 0.;
    z = 0.;
    }

#if defined(NEWTONPP_USE_OMP_LOOP)
    #pragma omp target teams loop reduction(+: m,x,y,z), is_device_ptr(mi,xi,yi,zi), map(alloc: m,x,y,z)
#else
    #pragma omp target teams distribute parallel for reduction(+: m,x,y,z), is_device_ptr(mi,xi,yi,zi), map(alloc: m,x,y,z)
#endif
    for (long i = 0; i < n; ++i)
    {
        m += mi[i];
        x += mi[i]*xi[i];
        y += mi[i]*yi[i];
        z += mi[i]*zi[i];
    }

    #pragma omp target is_device_ptr(mo,xo,yo,zo), map(alloc: m,x,y,z)
    {
    mo[0] = m;
    xo[0] = x/m;
    yo[0] = y/m;
    zo[0] = z/m;
    }

    #pragma omp target exit data map(release: m,x,y)
}
