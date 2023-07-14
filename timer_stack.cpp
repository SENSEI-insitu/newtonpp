#include "timer_stack.h"
#include <iostream>

// **************************************************************************
timer_stack::timer_stack(int active) : m_active(active)
{
    if (!m_active) return;

    timeval now{};
    gettimeofday(&now, nullptr);
    m_stack.push_back(now);
}

// **************************************************************************
timer_stack::~timer_stack()
{
    if (!m_active) return;

    pop("total time");
}

// **************************************************************************
void timer_stack::push()
{
    if (!m_active) return;

    timeval now{};
    gettimeofday(&now, nullptr);
    m_stack.push_back(now);
}

// **************************************************************************
void timer_stack::pop(const char *event)
{
    if (!m_active) return;

    timeval now{};
    gettimeofday(&now, nullptr);

    timeval &prev = m_stack.back();

    double dt = (now.tv_sec * 1e6 + now.tv_usec) -
                (prev.tv_sec * 1e6 + prev.tv_usec);

    m_stack.pop_back();

    std::cerr << " === newton++ === : " << event << " : "
        << dt / 1e6 << "s" << std::endl;
}
