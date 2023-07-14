#ifndef timer_stack_h
#define timer_stack_h

#include <vector>
#include <sys/time.h>

class timer_stack
{
public:
    timer_stack() = delete;
    timer_stack(int active);
    ~timer_stack();

    void push();
    void pop(const char *event);
    void pop_push(const char *event) { pop(event); push(); }

private:
    int m_active;
    std::vector<timeval> m_stack;
};

#endif
