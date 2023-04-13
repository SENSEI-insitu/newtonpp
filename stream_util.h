#ifndef stream_util_h
#define stream_util_h

#include <iostream>
#include <vector>

/// sends a vector to the output stream
template <typename T>
std::ostream &operator<<(std::ostream &os, const std::vector<T> &v)
{
    size_t n = v.size();
    if (n)
    {
        os << v[0];
        for (size_t i = 1; i < n; ++i)
            os << ", " << v[i];
    }
    return os;
}

/// sends a vector of vectors to the output stream
template <typename T>
std::ostream &operator<<(std::ostream &os, const std::vector<std::vector<T>> &v)
{
    size_t n = v.size();
    for (size_t i = 0; i < n; ++i)
    {
        os << i << " : ";
        size_t m = v[i].size();
        if (m)
        {
            os << v[i][0];
            for (size_t j = 1; j < m; ++j)
                os << ", " << v[i][j];
        }
        os << std::endl;
    }
    return os;
}

#endif
