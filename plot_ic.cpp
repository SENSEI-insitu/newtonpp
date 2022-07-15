#include <iostream>
#include <cstring>
#include <vector>
#include <random>
#include <deque>
#include <limits>
#include "unistd.h"
#include "fcntl.h"
#include "sys/stat.h"

//---------------------------------------------------------------------------
void plot(const std::vector<double> &parts, long n_parts,
    const std::vector<double> &m,
    const std::vector<double> &x, const std::vector<double> &y,
    const std::vector<double> &u, const std::vector<double> &v,
    const std::vector<double> &c, const std::vector<double> &o)
{
    long nt = x.size();

    std::cout << "import numpy as np" << std::endl
        << "import matplotlib.pyplot as plt" << std::endl
        << "from matplotlib.patches import Rectangle" << std::endl;

    std::cout << "x = np.array([";
    for (long i = 0; i < nt; ++i)
    {
        std::cout << x[i] << ", ";
    }
    std::cout <<  "])" << std::endl;

    std::cout << "y = np.array([";
    for (long i = 0; i < nt; ++i)
    {
        std::cout << y[i] << ", ";
    }
    std::cout <<  "])" << std::endl;

    std::cout << "u = np.array([";
    for (long i = 0; i < nt; ++i)
    {
        std::cout << u[i] << ", ";
    }
    std::cout <<  "])" << std::endl;

    std::cout << "v = np.array([";
    for (long i = 0; i < nt; ++i)
    {
        std::cout << v[i] << ", ";
    }
    std::cout <<  "])" << std::endl;

    std::cout << "c = np.array([";
    for (long i = 0; i < nt; ++i)
    {
        std::cout << c[i] << ", ";
    }
    std::cout <<  "])" << std::endl;

    std::cout << "o = np.array([";
    for (long i = 0; i < nt; ++i)
    {
        std::cout << o[i] << ", ";
    }
    std::cout <<  "])" << std::endl;

    std::cout << "plt.scatter(x,y, c=o, alpha=0.7)" << std::endl
        << "plt.quiver(x,y, u,v)" << std::endl;

    for (long i = 0; i < n_parts; ++i)
    {
        const double *pi = parts.data() + 4*i;

        std::cout << "plt.gca().add_patch(Rectangle((" << pi[0] << ", " << pi[2] << "), "
            << pi[1] - pi[0] << ", " << pi[3] - pi[2]
            << ", edgecolor='red', facecolor='none', lw=2))" << std::endl;
    }

    std::cout << "plt.axis('square')" << std::endl
        << "plt.show()" << std::endl;
}

/// read n elements of type T
template <typename T>
int readn(int fh, T *buf, size_t n)
{
    ssize_t ierr = 0;
    ssize_t nb = n*sizeof(T);
    if ((ierr = read(fh, buf, nb)) != nb)
    {
        std::cerr << "Failed to read " << n << " elements of size "
            << sizeof(T) << std::endl << strerror(errno) << std::endl;
        return -1;
    }
    return 0;
}

// --------------------------------------------------------------------------
int read(const std::string &idir, std::vector<double> &parts, long &n)
{
    std::string fn = idir + "/patches.npp";
    int h = open(fn.c_str(), O_RDONLY);
    if (h < 0)
    {
        std::cerr << "Failed to open \"" << fn << "\"" << std::endl
            << strerror(errno) << std::endl;
        return -1;
    }

    n = 0;
    if (readn(h, &n, 1 ))
    {
        std::cerr << "Failed to read \"" << fn << "\"" << std::endl;
        close(h);
        return -1;
    }

    parts.resize(4*n);
    if (readn(h, parts.data(), 4*n ))
    {
        std::cerr << "Failed to read \"" << fn << "\"" << std::endl;
        close(h);
        return -1;
    }

    close(h);
    return 0;
}

// --------------------------------------------------------------------------
int read(const std::string &idir, long np,
    std::vector<double> &m, std::vector<double> &x,
    std::vector<double> &y, std::vector<double> &u,
    std::vector<double> &v, std::vector<double> &c,
    std::vector<double> &o)
{
    for (long j = 0; j < np; ++j)
    {
        std::string fn = idir + "/patch_data_" + std::to_string(j) + ".npp";
        int h = open(fn.c_str(), O_RDONLY);
        if (h < 0)
        {
            std::cerr << "Failed to open \"" << fn << "\"" << std::endl
                << strerror(errno) << std::endl;
            return -1;
        }

        long n = 0;
        if (readn(h, &n, 1 ))
        {
            std::cerr << "Failed to read \"" << fn << "\"" << std::endl;
            close(h);
            return -1;
        }

        if (n)
        {
            std::vector<double> tm(n);
            std::vector<double> tx(n);
            std::vector<double> ty(n);
            std::vector<double> tu(n);
            std::vector<double> tv(n);
            std::vector<double> tc(n);
            std::vector<double> to(n,j);

            if (readn(h, tm.data(), n ) ||
                readn(h, tx.data(), n ) || readn(h, ty.data(), n ) ||
                readn(h, tu.data(), n ) || readn(h, tv.data(), n ) ||
                readn(h, tc.data(), n ))
            {
                std::cerr << "Failed to read \"" << fn << "\"" << std::endl;
                close(h);
                return -1;
            }

            m.insert(m.end(), tm.begin(), tm.end());
            x.insert(x.end(), tx.begin(), tx.end());
            y.insert(y.end(), ty.begin(), ty.end());
            u.insert(u.end(), tu.begin(), tu.end());
            v.insert(v.end(), tv.begin(), tv.end());
            c.insert(c.end(), tc.begin(), tc.end());
            o.insert(o.end(), to.begin(), to.end());
        }

        close(h);

    }

    return 0;
}


int main(int argc, char **argv)
{
    std::string idir = argv[1];

    long n_parts = 0;
    std::vector<double> parts;
    read(idir, parts, n_parts);

    std::vector<double> m;
    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> u;
    std::vector<double> v;
    std::vector<double> c;
    std::vector<double> o;
    read(idir, n_parts, m,x,y,u,v,c,o);

    plot(parts, n_parts, m,x,y,u,v,c,o);

    return 0;
}
