#include <iostream>
#include <cstring>
#include <vector>
#include <random>
#include <deque>
#include <limits>
#include "unistd.h"
#include "fcntl.h"
#include "sys/stat.h"

struct patch
{
    patch() : m_x{0., 0., 0., 0.} {}
    patch(double x0, double x1, double y0, double y1) : m_x{x0, x1, y0, y1} {}
    patch(const patch &o) : m_x{o.m_x[0], o.m_x[1], o.m_x[2], o.m_x[3]} {}
    void operator=(const patch &o) { memcpy(m_x, o.m_x, 4*sizeof(double)); }

    double m_x[4];
};

// --------------------------------------------------------------------------
std::ostream &operator<<(std::ostream &os, const patch &p)
{
    os << "[" << p.m_x[0] << ", " << p.m_x[1]
        << ", " << p.m_x[2] << ", " << p.m_x[3] << "]";
    return os;
}


// --------------------------------------------------------------------------
void split(int dir, const double *p0_x, double *p1_x, double *p2_x)
{
    if (dir == 0)
    {
        p1_x[0] = p0_x[0];
        p1_x[1] = (p0_x[0] + p0_x[1]) / 2.;
        p1_x[2] = p0_x[2];
        p1_x[3] = p0_x[3];

        p2_x[0] = p1_x[1];
        p2_x[1] = p0_x[1];
        p2_x[2] = p0_x[2];
        p2_x[3] = p0_x[3];
    }
    else
    {
        p1_x[0] = p0_x[0];
        p1_x[1] = p0_x[1];
        p1_x[2] = p0_x[2];
        p1_x[3] = (p0_x[2] + p0_x[3]) / 2.;

        p2_x[0] = p0_x[0];
        p2_x[1] = p0_x[1];
        p2_x[2] = p1_x[3];
        p2_x[3] = p0_x[3];
    }
}

// --------------------------------------------------------------------------
int inside(const double *p, double x, double y)
{
    return (x >= p[0]) && (x < p[1]) && (y >= p[2]) && (y < p[3]);
}

// --------------------------------------------------------------------------
void split(int dir, const patch &p0, patch &p1, patch &p2)
{
    split(dir, p0.m_x, p1.m_x, p2.m_x);
}
// --------------------------------------------------------------------------
std::vector<patch> partition(const patch &dom, size_t n_out)
{
    std::deque<patch> patches;
    patches.push_back(dom);

    int pass = 0;
    while (patches.size() != n_out)
    {
        size_t n = patches.size();
        int dir = pass % 2;
        for (size_t i = 0; i < n; ++i)
        {
            patch p0 = patches.front();
            patches.pop_front();

            patch p1, p2;
            split(dir, p0, p1, p2);

            patches.push_back(p1);
            patches.push_back(p2);

            if (patches.size() == n_out)
                break;
        }
        pass += 1;
    }

    return std::vector<patch>(patches.begin(), patches.end());
}

// --------------------------------------------------------------------------
void spiral(double cx, double cy,
    double vx, double vy,
    double a, double k, double rot, double w0, double w1,
    double t0, double t1, long ns, long nb,
    double x0, double y0,
    double m0, double m1, double v0,
    std::vector<double> &mo,
    std::vector<double> &xo, std::vector<double> &yo,
    std::vector<double> &uo, std::vector<double> &vo,
    std::vector<double> &co)
{
    // spread nb points acros ns spiral segments
    long nps = nb / ns;

    // use symetry to generate the lower branch
    long nt = 2*nps*ns;

    mo.resize(nt);
    xo.resize(nt);
    yo.resize(nt);
    uo.resize(nt);
    vo.resize(nt);
    co.resize(nt);

    double dt = (t1 - t0) / (ns - 1);
    double dm = m1 - m0;
    //double dw = (w1 - w0) / (ns - 1);
    double w = w0;

    std::mt19937 gen(1); // seed
    std::uniform_real_distribution<double> udist(0.,1.);
    std::normal_distribution<double> ndist(0.,.6);

    for (long q = 0; q < ns; ++q)
    {
        double t = t0 + q * dt;

        // spread more further aloong the curve
        //double w = w0 + q * dw;
        w *= w1;

        // spiral
        double r = a * exp( k * t );

        double x = r * cos( t );
        double y = r * sin( t );

        // tangent to spiral
        double u = y + k * x;
        double v = -x + k * y;
        double muv = sqrt( u*u + v*v );

        u /= muv;
        v /= muv;

        // upper branch
        for (long i = 0; i < nps; ++i)
        {
            long qq = q*nps + i;
            mo[qq] = m0 + dm * std::max(0., std::min(1., udist(gen)));
            xo[qq] = ( x + w * ndist(gen) );
            yo[qq] = ( y + w * ndist(gen) );
            uo[qq] = ( v0 * u + vx );
            vo[qq] = ( v0 * v + vy );
            co[qq] = 0.;
        }

        // lower branch
        for (long i = 0; i < nps; ++i)
        {
            long qq = nps*ns + q*nps + i;
            mo[qq] = m0 + dm * std::max(0., std::min(1., udist(gen)));
            xo[qq] = -1. * ( x + w * ndist(gen) );
            yo[qq] = -1. * ( y + w * ndist(gen) );
            uo[qq] = -1. * ( v0 * u )  + vx;
            vo[qq] = -1. * ( v0 * v )  + vy;
            co[qq] = 1.;
        }
    }


    // position
    // rotate
    for (long i = 0; i < nt; ++i)
    {
        double xr = xo[i] * cos( rot ) - yo[i] * sin( rot );
        double yr = xo[i] * sin( rot ) + yo[i] * cos( rot );

        xo[i] = xr;
        yo[i] = yr;
    }

    // normalize positions to -1 to 1
    double mxr = 0.;
    for (long i = 0; i < nt; ++i)
    {
        mxr = std::max( mxr, sqrt( xo[i]*xo[i] + yo[i]*yo[i] ) );
    }

    for (long i = 0; i < nt; ++i)
    {
        xo[i] /= mxr;
        yo[i] /= mxr;
    }

    // translate
    for (long i = 0; i < nt; ++i)
    {
        xo[i] += cx;
        yo[i] += cy;
    }

    // scale
    for (long i = 0; i < nt; ++i)
    {
        xo[i] *= x0;
        yo[i] *= y0;
    }

    // velocity
    // rotate
    for (long i = 0; i < nt; ++i)
    {
        double ur = uo[i] * cos( rot ) - vo[i] * sin( rot );
        double vr = uo[i] * sin( rot ) + vo[i] * cos( rot );

        uo[i] = ur;
        vo[i] = vr;
    }

    // scale
    double sx = x0 < 0 ? -1. : 1.;
    double sy = y0 < 0 ? -1. : 1.;
    for (long i = 0; i < nt; ++i)
    {
        uo[i] *= sx*v0;
        vo[i] *= sy*v0;
    }

    // translate
    for (long i = 0; i < nt; ++i)
    {
        uo[i] += vx;
        vo[i] += vy;
    }
}

// --------------------------------------------------------------------------
patch domain(std::vector<double> &x, std::vector<double> &y, double pad = 1.2)
{
    long np = x.size();

    double mxx = std::numeric_limits<double>::lowest();
    double mnx = std::numeric_limits<double>::max();
    double mxy = std::numeric_limits<double>::lowest();
    double mny = std::numeric_limits<double>::max();

    for (long i = 0; i < np; ++i)
    {
        mnx = std::min( mnx, x[i] );
        mxx = std::max( mxx, x[i] );

        mny = std::min( mny, y[i] );
        mxy = std::max( mxy, y[i] );
    }

    double dx = (mxx - mnx);
    double dy = (mxy - mny);

    double ds = std::max( dx, dy ) * pad / 2.;

    double cx = (mnx + mxx) / 2.;
    double cy = (mny + mxy) / 2.;

    return patch(cx - ds, cx + ds, cy - ds, cy + ds);
}



/// write n elements of type T
// --------------------------------------------------------------------------
template <typename T>
int writen(int fh, T *buf, size_t n)
{
    ssize_t ierr = 0;
    ssize_t nb = n*sizeof(T);
    if ((ierr = write(fh, buf, nb)) != nb)
    {
        std::cerr << "Failed to write " << n << " elements of size "
            << sizeof(T) << std::endl << strerror(errno) << std::endl;
        return -1;
    }
    return 0;
}

// --------------------------------------------------------------------------
int write(const std::string &odir, const std::vector<patch> &parts)
{
    std::string fn = odir + "/patches.npp";
    int h = open(fn.c_str(), O_WRONLY|O_CREAT|O_TRUNC, S_IRUSR|S_IWUSR|S_IRGRP|S_IROTH );
    if (h < 0)
    {
        std::cerr << "Failed to open \"" << fn << "\"" << std::endl
            << strerror(errno) << std::endl;
        return -1;
    }

    long np = parts.size();
    std::vector<double> buf(4*np);
    double *pbuf = buf.data();
    for (long i = 0; i < np; ++i)
    {
        const patch &pi = parts[i];
        memcpy(pbuf + 4*i, pi.m_x, 4*sizeof(double));
    }

    if (writen(h, &np, 1 ) || writen(h, pbuf, 4*np))
    {
        std::cerr << "Failed to write patches" << std::endl;
        close(h);
        return -1;
    }

    close(h);
    return 0;
}

// --------------------------------------------------------------------------
int write(const std::string &fn,
    const std::vector<double> &m,
    const std::vector<double> &x, const std::vector<double> &y,
    const std::vector<double> &u, const std::vector<double> &v,
    const std::vector<double> &c)
{
    int h = open(fn.c_str(), O_WRONLY|O_CREAT|O_TRUNC, S_IRUSR|S_IWUSR|S_IRGRP|S_IROTH );
    if (h < 0)
    {
        std::cerr << "Failed to open \"" << fn << "\"" << std::endl
            << strerror(errno) << std::endl;
        return -1;
    }

    long np = m.size();
    if (writen(h, &np, 1 ) ||
        writen(h, m.data(), np) || writen(h, x.data(), np) || writen(h, y.data(), np) ||
        writen(h, u.data(), np) || writen(h, v.data(), np) || writen(h, c.data(), np))
    {
        std::cerr << "Failed to write data" << std::endl;
        close(h);
        return -1;
    }

    close(h);
    return 0;
}

// --------------------------------------------------------------------------
int write(const std::string &odir,
    const std::vector<patch> &parts,
    const std::vector<double> &m,
    const std::vector<double> &x, const std::vector<double> &y,
    const std::vector<double> &u, const std::vector<double> &v,
    const std::vector<double> &c)
{
    long n = m.size();

    std::vector<double> pm(n);
    std::vector<double> px(n);
    std::vector<double> py(n);
    std::vector<double> pu(n);
    std::vector<double> pv(n);
    std::vector<double> pc(n);

    std::vector<int> vis(n,0);

    long no = 0;
    long np = parts.size();
    for (long j = 0; j < np; ++j)
    {
        // select the particles associated with this patch
        pm.resize(0);
        px.resize(0);
        py.resize(0);
        pu.resize(0);
        pv.resize(0);
        pc.resize(0);

        const patch &pj = parts[j];

        for (long i = 0; i < n; ++i)
        {
            if (!vis[i] && inside(pj.m_x, x[i], y[i]))
            {
                vis[i] = 1;
                pm.push_back(m[i]);
                px.push_back(x[i]);
                py.push_back(y[i]);
                pu.push_back(u[i]);
                pv.push_back(v[i]);
                pc.push_back(c[i]);
            }
        }

        // write it
        std::string fn = odir + "/patch_data_" + std::to_string(j) + ".npp";
        if (write(fn, pm,px,py,pu,pv,pc))
        {
            std::cerr << "Failed to write data for patch " << j << std::endl;
            return -1;
        }
    }

    return 0;
}


int main(int argc, char **argv)
{
    if (argc < 20)
    {
        std::cerr << "usage:" << std::endl
            << "spiral [ns] {  [cx] [cy] [vx] [vy] [a] [k] [rot] [w0] [w1] [t0] [t1] [ns] [nb] [x0] [y0] [m0] [m1] [v0] ... } [n ranks] [odir]" << std::endl;
        return -1;
    }

    std::vector<double> m;
    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> u;
    std::vector<double> v;
    std::vector<double> c;

    int q = 0;

    int  ns = atoi(argv[++q]);

    for (int i = 0; i < ns; ++i)
    {
        double cx = atof(argv[++q]);
        double cy = atof(argv[++q]);
        double vx = atof(argv[++q]);
        double vy = atof(argv[++q]);
        double a = atof(argv[++q]);
        double k = atof(argv[++q]);
        double rot = atof(argv[++q]);
        double w0 = atof(argv[++q]);
        double w1 = atof(argv[++q]);
        double t0 = atof(argv[++q]);
        double t1 = atof(argv[++q]);
        long ns = atoi(argv[++q]);
        long nb = atoi(argv[++q]);
        double x0 = atof(argv[++q]);
        double y0 = atof(argv[++q]);
        double m0 = atof(argv[++q]);
        double m1 = atof(argv[++q]);
        double v0 = atof(argv[++q]);

        std::vector<double> tm;
        std::vector<double> tx;
        std::vector<double> ty;
        std::vector<double> tu;
        std::vector<double> tv;
        std::vector<double> tc;

        spiral(cx,cy,vx,vy,a,k,rot,w0,w1,t0,t1,ns,nb,x0,y0,m0,m1,v0,tm,tx,ty,tu,tv,tc);

        m.insert(m.end(), tm.begin(), tm.end());
        x.insert(x.end(), tx.begin(), tx.end());
        y.insert(y.end(), ty.begin(), ty.end());
        u.insert(u.end(), tu.begin(), tu.end());
        v.insert(v.end(), tv.begin(), tv.end());
        c.insert(c.end(), tc.begin(), tc.end());
    }

    // partition the domain
    int n_parts = atoi(argv[++q]);
    patch dom = domain(x, y);
    std::vector<patch> parts = partition(dom, n_parts);

    // write the data
    std::string odir = argv[++q];
    write(odir, parts);
    write(odir, parts, m,x,y,u,v,c);

    //matplotlib(parts, m,x,y,u,v,c);

    return 0;
}

/*
//---------------------------------------------------------------------------
void matplotlib(const std::vector<patch> &parts,
    const std::vector<double> &mo,
    const std::vector<double> &xo, const std::vector<double> &yo,
    const std::vector<double> &uo, const std::vector<double> &vo,
    const std::vector<double> &co)
{
    long nt = xo.size();

    std::cout << "import numpy as np" << std::endl
        << "import matplotlib.pyplot as plt" << std::endl
        << "from matplotlib.patches import Rectangle" << std::endl;

    std::cout << "x = np.array([";
    for (long i = 0; i < nt; ++i)
    {
        std::cout << xo[i] << ", ";
    }
    std::cout <<  "])" << std::endl;

    std::cout << "y = np.array([";
    for (long i = 0; i < nt; ++i)
    {
        std::cout << yo[i] << ", ";
    }
    std::cout <<  "])" << std::endl;

    std::cout << "u = np.array([";
    for (long i = 0; i < nt; ++i)
    {
        std::cout << uo[i] << ", ";
    }
    std::cout <<  "])" << std::endl;

    std::cout << "v = np.array([";
    for (long i = 0; i < nt; ++i)
    {
        std::cout << vo[i] << ", ";
    }
    std::cout <<  "])" << std::endl;

    std::cout << "c = np.array([";
    for (long i = 0; i < nt; ++i)
    {
        std::cout << co[i] << ", ";
    }
    std::cout <<  "])" << std::endl;

    std::cout << "plt.scatter(x,y, c=c, alpha=0.7)" << std::endl
        << "plt.quiver(x,y, u,v)" << std::endl;


    long np = parts.size();
    for (long i = 0; i < np; ++i)
    {
        const patch &pi = parts[i];

        std::cout << "plt.gca().add_patch(Rectangle((" << pi.m_x[0] << ", " << pi.m_x[2] << "), "
            << pi.m_x[1] - pi.m_x[0] << ", " << pi.m_x[3] - pi.m_x[2]
            << ", edgecolor='red', facecolor='none', lw=2))" << std::endl;
    }

    std::cout << "plt.axis('square')" << std::endl
        << "plt.show()" << std::endl;
}
*/
