#include "patch_force.h"
#include <iostream>


// --------------------------------------------------------------------------
patch_force::patch_force(hamr::buffer_allocator alloc) : m_u(alloc), m_v(alloc), m_w(alloc)
{
    #ifdef DEBUG
    std::cerr << "patch_force::patch_force " << this << std::endl;
    #endif
}

// --------------------------------------------------------------------------
patch_force::~patch_force()
{
    #ifdef DEBUG
    std::cerr << "patch_force::~patch_force " << this << std::endl;
    #endif
}

// --------------------------------------------------------------------------
void patch_force::operator=(const patch_force &pd)
{
    #ifdef DEBUG
    std::cerr << "patch_force::operator= " << this << " <-- " << &pd << std::endl;
    #endif

    m_u.assign(pd.m_u);
    m_v.assign(pd.m_v);
    m_w.assign(pd.m_w);
}

// --------------------------------------------------------------------------
void patch_force::operator=(patch_force &&pd)
{
    #ifdef DEBUG
    std::cerr << "patch_force::operator=&& " << this << " <-- " << &pd << std::endl;
    #endif

    m_u = std::move(pd.m_u);
    m_v = std::move(pd.m_v);
    m_w = std::move(pd.m_w);
}

// --------------------------------------------------------------------------
void patch_force::resize(long n)
{
    #ifdef DEBUG
    std::cerr << "patch_force::resize " << this << std::endl;
    #endif

    m_u.resize(n);
    m_v.resize(n);
    m_w.resize(n);
}

// --------------------------------------------------------------------------
void patch_force::append(const patch_force &o)
{
    #ifdef DEBUG
    std::cerr << "patch_force::append " << this << std::endl;
    #endif

    m_u.append(o.m_u);
    m_v.append(o.m_v);
    m_w.append(o.m_w);
}

