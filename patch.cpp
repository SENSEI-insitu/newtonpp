#include "patch.h"
#include <iostream>

// --------------------------------------------------------------------------
patch::patch(int owner, double x0, double x1,
    double y0, double y1, double z0, double z1) : m_owner(owner), m_x(def_alloc())
{
    double bds[] = {x0, x1, y0, y1, z0, z1};
    hamr::buffer<double> x(def_alloc(), 6, bds);
    m_x.swap(x);
}

// --------------------------------------------------------------------------
patch::patch(const patch &p) : m_x(p.m_x)
{
    m_owner = p.m_owner;
}

// --------------------------------------------------------------------------
void patch::operator=(const patch &p)
{
    m_owner = p.m_owner;
    m_x.assign(p.m_x);
}


// --------------------------------------------------------------------------
std::ostream &operator<<(std::ostream &os, const patch &p)
{
    auto spx = p.m_x.get_cpu_accessible();
    const double *px = spx.get();

    os << "{" << p.m_owner << " [" << px[0] << ", " << px[1] << ", "
        << px[2] << ", " << px[3] << ", " << px[4] << ", " << px[5] << "]}";

    return os;
}

/*
// --------------------------------------------------------------------------
void grow(patch &p, double amt)
{
    p.m_x[0] -= amt;
    p.m_x[1] += amt;
    p.m_x[2] -= amt;
    p.m_x[3] += amt;
}

// --------------------------------------------------------------------------
void shrink(patch &p, double amt)
{
    p.m_x[0] += amt;
    p.m_x[1] -= amt;
    p.m_x[2] += amt;
    p.m_x[3] -= amt;
}
*/
/*
// --------------------------------------------------------------------------
bool intersects(const patch &lp, const patch &rp)
{
    double lx = std::max(lp.m_x[0], rp.m_x[0]);
    double hx = std::min(lp.m_x[1], rp.m_x[1]);
    double ly = std::max(lp.m_x[2], rp.m_x[2]);
    double hy = std::min(lp.m_x[3], rp.m_x[3]);
    return (lx <= hx) && (ly <= hy);
}
*/
/*
// --------------------------------------------------------------------------
bool inside(const patch &p, double x, double y)
{
    return (x >= p.m_x[0]) && (x < p.m_x[1]) && (y >= p.m_x[2]) && (y < p.m_x[3]);
}
*/

/** finds the list of patch neighbors and returns them in pn. returns the
 * non-neighbor patches in pnn
// --------------------------------------------------------------------------
void neighbors(const std::vector<patch> &patches,
    double ofs, std::vector<std::vector<int>> &pn, std::vector<std::vector<int>> &pnn)
{
    std::vector<patch> smallp(patches);

    // make patches disjoint by shrinking a small amount
    int n = patches.size();
    for (int i = 0; i < n; ++i)
        shrink(smallp[i], ofs);

    double ofs2 = 2*ofs;

    // a neighbor list for each patch
    pn.resize(n);
    pnn.resize(n);

    for (int i = 0; i < n; ++i)
    {
        // grow active patch by 2x the small amount
        patch actp = patches[i];
        grow(actp, ofs2);

        // see which of the others this patch intersects
        for (int j = 0; j < n; ++j)
        {
            // dont test a patch against itself
            if (i == j) continue;

            const patch &spj = smallp[j];

            if (intersects(actp, spj))
            {
                // j is a neighbor of i
                pn[i].push_back(j);
            }
            else
            {
                pnn[i].push_back(j);
            }
        }
    }
}
 */
