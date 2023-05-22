#include "sensei_adaptor.h"

#include "svtkDataArray.h"
#include "svtkHAMRDataArray.h"
#include "svtkUnsignedCharArray.h"
#include "svtkIntArray.h"
#include "svtkDoubleArray.h"
#include "svtkHAMRDataArray.h"
#include "svtkTable.h"
#include "svtkUnstructuredGrid.h"
#include "svtkMultiBlockDataSet.h"
#include "svtkDataSetAttributes.h"
#include "svtkCellData.h"
#include "svtkPointData.h"


// given an array name get a pointer to the associated buffer instance or null
// if the array name is not known
const hamr::buffer<double>*
GetBuffer(const std::string &anm, const patch_data *pd, const patch_force *pf)
{
    switch (anm[0])
    {
        case 'm':
            return &pd->m_m;
        case 'x':
            return &pd->m_x;
        case 'y':
            return &pd->m_y;
        case 'z':
            return &pd->m_z;
        case 'v':
            switch (anm[1])
            {
                case 'u':
                    return &pd->m_u;
                case 'v':
                    return &pd->m_v;
                case 'w':
                    return &pd->m_w;
            }
            break;
        case 'f':
            switch (anm[1])
            {
                case 'u':
                    return &pf->m_u;
                case 'v':
                    return &pf->m_v;
                case 'w':
                    return &pf->m_w;
            }
            break;
    }

    std::cerr << "Error: no array named \"" << anm << "\"" << std::endl;
    return nullptr;
}

// computes the min/max of the data.
void GetMinMax(const hamr::buffer<double> &buf, long n_elem, double &mn, double &mx)
{
    mn = std::numeric_limits<double>::max();
    mx = std::numeric_limits<double>::lowest();

    const double *pbuf = buf.data();

/*#if defined(NEWTONPP_USE_OMP_LOOP)
    #pragma omp target teams loop is_device_ptr(pbuf), reduction(min:mn), reduction(max:mx)
#else*/
    #pragma omp target teams distribute parallel for is_device_ptr(pbuf), reduction(min:mn), reduction(max:mx)
//#endif
    for (long i = 0; i < n_elem; ++i)
    {
        mn = pbuf[i] > mn ? mn : pbuf[i];
        mx = pbuf[i] < mx ? mx : pbuf[i];
    }
}

//-----------------------------------------------------------------------------
senseiNewMacro(sensei_adaptor);

// --------------------------------------------------------------------------
void sensei_adaptor::SetData(long step, double time,
    const std::vector<patch> &patches, const patch_data &pd,
    const patch_force &pf)
{
    m_patches = &patches;
    m_bodies = &pd;
    m_body_forces = &pf;

    this->SetDataTimeStep(step);
    this->SetDataTime(time);
}

// --------------------------------------------------------------------------
int sensei_adaptor::GetNumberOfMeshes(unsigned int &numMeshes)
{
    numMeshes = 3;
    return 0;
}

// --------------------------------------------------------------------------
int sensei_adaptor::GetMeshMetadata(unsigned int id, sensei::MeshMetadataPtr &md)
{
    const int pCen = svtkDataObject::POINT;
    const int cCen = svtkDataObject::CELL;
    const int f64 = SVTK_DOUBLE;
    const int i32 = SVTK_INT;

    int rank = 0;
    MPI_Comm_rank(this->GetCommunicator(), &rank);

    // patches
    // mesh 0 has the patches. each patch is represented by an unstructured
    //        grid with one hexhedra cell.
    //
    // bodies
    // mesh 1 is a table where collumns are bodies and per-body forces. This
    //        is the zero copy option that passes pointers to data in place
    //        without requiring any restructring or movement. Note: in ParaView
    //        one can use the TableToPoints filter.
    //
    // bodies_ug
    // mesh 2 is a polydata with bodies and per-body forces. The coordinates
    //        are deep coppied, to put them into the layout required by SVTK.
    //        The cell arrays are alway generated on the CPU. The other arrays
    //        are passed in place without movement or modificaiton.

    if (id > 2)
    {
        std::cerr << "ERROR: invalid mesh id " << id << std::endl;
        return -1;
    }

    md->GlobalView = 0;
    md->StaticMesh = 0;
    md->MeshType = SVTK_MULTIBLOCK_DATA_SET;

    if (id == 0)
    {
        md->MeshName = "patches";
        md->BlockType = SVTK_UNSTRUCTURED_GRID;
    }
    else if (id == 1)
    {
        md->MeshName = "bodies";
        md->BlockType = SVTK_TABLE;
    }
    else
    {
        md->MeshName = "bodies_ug";
        md->BlockType = SVTK_UNSTRUCTURED_GRID;
    }

    md->CoordinateType = f64;
    md->NumBlocks = 1;
    md->NumBlocksLocal = {1};
    md->NumGhostCells = 0;

    if (id == 0)
    {
        md->NumArrays = 1;
        md->ArrayName = {"owner"};
        md->ArrayCentering = {cCen};
        md->ArrayComponents = {1};
        md->ArrayType = {i32};
    }
    else
    {
        md->NumArrays = 11;
        md->ArrayName = {"rank", "m", "x", "y", "z",
                         "vu", "vv", "vw", "fu", "fv", "fw"};

        md->ArrayCentering = {pCen, pCen, pCen, pCen, pCen,
                              pCen, pCen, pCen, pCen, pCen, pCen};

        md->ArrayComponents = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

        md->ArrayType = {i32, f64, f64, f64, f64,
                         f64, f64, f64, f64, f64, f64};
    }


    // block extents can't be provided.
    if (md->Flags.BlockExtentsSet())
        md->Flags.ClearBlockExtents();

    if (md->Flags.BlockBoundsSet())
    {
        auto [spx, px] = hamr::get_cpu_accessible(m_patches->at(rank).m_x);
        md->Bounds = {px[0], px[1], px[2], px[3], px[4]};
        md->BlockBounds = {{px[0], px[1], px[2], px[3], px[4]}};
    }

    if (md->Flags.BlockSizeSet())
    {
        if (id == 0)
        {
            md->BlockNumPoints = {8}; // one cell per patch, each w/ 8 points
            md->BlockNumCells = {1};
            md->BlockCellArraySize = {8};
        }
        else
        {
            md->BlockNumPoints = {m_bodies->size()}; // one cell per body, each w/ 1 point
            md->BlockNumCells = {m_bodies->size()};
            md->BlockCellArraySize = {m_bodies->size()};
        }
    }

    if (md->Flags.BlockDecompSet())
    {
        md->BlockOwner = {rank};
        md->BlockIds = {rank};
    }

    if (md->Flags.BlockArrayRangeSet())
    {
        // rank
        md->ArrayRange.push_back({{(double)rank, (double)rank}});
        md->BlockArrayRange.push_back({{(double)rank, (double)rank}});

        if (id != 0)
        {
            double mn, mx;
            long nb = m_bodies->size();

            // m
            GetMinMax(m_bodies->m_m, nb, mn, mx);
            md->ArrayRange.push_back({{mn, mx}});
            md->BlockArrayRange.push_back({{mn, mx}});

            // px
            GetMinMax(m_bodies->m_x, nb, mn, mx);
            md->ArrayRange.push_back({{mn, mx}});
            md->BlockArrayRange.push_back({{mn, mx}});

            // py
            GetMinMax(m_bodies->m_y, nb, mn, mx);
            md->ArrayRange.push_back({{mn, mx}});
            md->BlockArrayRange.push_back({{mn, mx}});

            // pz
            GetMinMax(m_bodies->m_z, nb, mn, mx);
            md->ArrayRange.push_back({{mn, mx}});
            md->BlockArrayRange.push_back({{mn, mx}});

            // vu
            GetMinMax(m_bodies->m_u, nb, mn, mx);
            md->ArrayRange.push_back({{mn, mx}});
            md->BlockArrayRange.push_back({{mn, mx}});

            // vv
            GetMinMax(m_bodies->m_v, nb, mn, mx);
            md->ArrayRange.push_back({{mn, mx}});
            md->BlockArrayRange.push_back({{mn, mx}});

            // vw
            GetMinMax(m_bodies->m_w, nb, mn, mx);
            md->ArrayRange.push_back({{mn, mx}});
            md->BlockArrayRange.push_back({{mn, mx}});

            // fu
            GetMinMax(m_body_forces->m_u, nb, mn, mx);
            md->ArrayRange.push_back({{mn, mx}});
            md->BlockArrayRange.push_back({{mn, mx}});

            // fv
            GetMinMax(m_body_forces->m_v, nb, mn, mx);
            md->ArrayRange.push_back({{mn, mx}});
            md->BlockArrayRange.push_back({{mn, mx}});

            // fw
            GetMinMax(m_body_forces->m_w, nb, mn, mx);
            md->ArrayRange.push_back({{mn, mx}});
            md->BlockArrayRange.push_back({{mn, mx}});
        }
    }
    return 0;
}

// --------------------------------------------------------------------------
int sensei_adaptor::GetMesh(const std::string &meshName,
    bool structureOnly, svtkDataObject *&mesh)
{
    (void)structureOnly;

    int rank = 0;
    MPI_Comm_rank(this->GetCommunicator(), &rank);

    mesh = nullptr;

    svtkMultiBlockDataSet *mbds = svtkMultiBlockDataSet::New();
    mbds->SetNumberOfBlocks(m_patches->size());

    if (meshName == "patches")
    {
        // This mesh shows the region of space covered by each patch and the
        // MPI rank that owns it

        // topology
        auto con = svtkTypeInt64Array::New();
        con->SetName("connectivity");
        con->SetNumberOfTuples(8);
        auto pcon = con->GetPointer(0);
        for (long i = 0; i < 8; ++i)
            pcon[i] = i;

        auto off = svtkTypeInt64Array::New();
        off->SetName("offsets");
        off->SetNumberOfTuples(2);
        auto poff = off->GetPointer(0);
        poff[0] = 0;
        poff[1] = 8;

        // coordinates
        auto pts = svtkDoubleArray::New();
        pts->SetName("coordinates");
        pts->SetNumberOfComponents(3);
        pts->SetNumberOfTuples(8);
        auto ppts = pts->GetPointer(0);

        const patch &pi = m_patches->at(rank);

        auto [spi_x, pi_x] = hamr::get_cpu_accessible(pi.m_x);

        ppts[0 ] = pi_x[0];
        ppts[1 ] = pi_x[2];
        ppts[2 ] = pi_x[4];

        ppts[3 ] = pi_x[1];
        ppts[4 ] = pi_x[2];
        ppts[5 ] = pi_x[4];

        ppts[6 ] = pi_x[1];
        ppts[7 ] = pi_x[3];
        ppts[8 ] = pi_x[4];

        ppts[9 ] = pi_x[0];
        ppts[10] = pi_x[3];
        ppts[11] = pi_x[4];

        ppts[12] = pi_x[0];
        ppts[13] = pi_x[2];
        ppts[14] = pi_x[5];

        ppts[15] = pi_x[1];
        ppts[16] = pi_x[2];
        ppts[17] = pi_x[5];

        ppts[18] = pi_x[1];
        ppts[19] = pi_x[3];
        ppts[20] = pi_x[5];

        ppts[21] = pi_x[0];
        ppts[22] = pi_x[3];
        ppts[23] = pi_x[5];

        // cell type
        auto ct = svtkUnsignedCharArray::New();
        ct->SetName("cell_types");
        ct->SetNumberOfTuples(1);
        auto pct = ct->GetPointer(0);
        pct[0] = SVTK_HEXAHEDRON;

        // package this into SVTK data model
        svtkCellArray *ca = svtkCellArray::New();
        ca->SetData(off, con);
        off->Delete();
        con->Delete();

        auto ug = svtkUnstructuredGrid::New();
        ug->SetCells(ct, ca);
        ug->GetPoints()->SetData(pts);
        ct->Delete();
        ca->Delete();
        pts->Delete();

        mbds->SetBlock(rank, ug);
        ug->Delete();

        mesh = mbds;

        return 0;
    }
    else if (meshName == "bodies")
    {
        // this is the zero-copy mesh that passes data as columns of a table.
        // the dataset is empty until arrays are added.
        auto tab = svtkTable::New();

        mbds->SetBlock(rank, tab);
        tab->Delete();

        mesh = mbds;

        return 0;
    }
    else if (meshName == "bodies_ug")
    {
        // this is the unstructured mesh that can be readily visualized in
        // Ascent, VisIt, and ParaView but cannot be zero-copied due to
        // limitations in VTK data model.

        long nb = m_bodies->size();

        auto [spd_x, pd_x] = hamr::get_cpu_accessible(m_bodies->m_x);
        auto [spd_y, pd_y] = hamr::get_cpu_accessible(m_bodies->m_y);
        auto [spd_z, pd_z] = hamr::get_cpu_accessible(m_bodies->m_z);

        // topology
        long ncon = nb;
        auto con = svtkTypeInt64Array::New();
        con->SetName("connectivity");
        con->SetNumberOfTuples(ncon);
        auto pcon = con->GetPointer(0);
        for (long i = 0; i < ncon; ++i)
            pcon[i] = i;

        long noff = ncon + 1;
        auto off = svtkTypeInt64Array::New();
        off->SetName("offsets");
        off->SetNumberOfTuples(noff);
        auto poff = off->GetPointer(0);
        for (long i = 0; i < noff; ++i)
            poff[i] = i;

        // coordinates
        auto pts = svtkDoubleArray::New();
        pts->SetName("coordinates");
        pts->SetNumberOfComponents(3);
        pts->SetNumberOfTuples(nb);
        auto ppts = pts->GetPointer(0);
        for (long i = 0; i < nb; ++i)
        {
            long ii = 3*i;
            ppts[ii    ] = pd_x[i];
            ppts[ii + 1] = pd_y[i];
            ppts[ii + 2] = pd_z[i];
        }

        auto pa = svtkPoints::New();
        pa->SetData(pts);
        pts->Delete();

        // cell type
        auto ct = svtkUnsignedCharArray::New();
        ct->SetName("cell_types");
        ct->SetNumberOfTuples(nb);
        auto pct = ct->GetPointer(0);
        for (long i = 0; i < nb; ++i)
            pct[i] = SVTK_VERTEX;

        // package this into SVTK data model
        svtkCellArray *ca = svtkCellArray::New();
        ca->SetData(off, con);
        off->Delete();
        con->Delete();

        auto ug = svtkUnstructuredGrid::New();
        ug->SetCells(ct, ca);
        ug->SetPoints(pa);
        ct->Delete();
        ca->Delete();
        pa->Delete();

        // set the block associated with this rank
        mbds->SetBlock(rank, ug);
        ug->Delete();

        mesh = mbds;

        return 0;
    }

    mesh = nullptr;
    std::cerr << "Failed to get mesh named \"" << meshName << "\"" << std::endl;
    mbds->Delete();

    return -1;
}

// --------------------------------------------------------------------------
int sensei_adaptor::AddArray(svtkDataObject* mesh,
     const std::string &meshName, int association, const std::string &arrayName)
{
    const int pCen = svtkDataObject::POINT;
    const int cCen = svtkDataObject::CELL;

    int rank = 0;
    MPI_Comm_rank(this->GetCommunicator(), &rank);

    auto mbds = dynamic_cast<svtkMultiBlockDataSet*>(mesh);
    if (!mbds)
    {
        std::cerr << "Invalid mesh passed to AddArray" << std::endl;
        return -1;
    }

    if (meshName == "patches")
    {
        if ((arrayName == "owner") && (association == cCen))
        {
            auto ug = dynamic_cast<svtkUnstructuredGrid*>(mbds->GetBlock(rank));
            if (ug)
            {
                auto aout = svtkIntArray::New();
                aout->SetName("owner");
                aout->SetNumberOfTuples(1);
                auto paout = aout->GetPointer(0);
                paout[0] = rank;

                ug->GetCellData()->AddArray(aout);
                aout->Delete();
                return 0;
            }
        }
    }
    else if (meshName == "bodies")
    {
        // zero-copy into a table with a column for each array
        auto tab = dynamic_cast<svtkTable*>(mbds->GetBlock(rank));
        if (tab)
        {
            long nb = m_bodies->size();
            svtkDataArray *aout = nullptr;

            if (arrayName == "owner")
            {
                auto ao = svtkIntArray::New();
                ao->SetName("owner");
                ao->SetNumberOfTuples(nb);
                ao->FillValue(rank);
                aout = ao;
            }
            else
            {
                auto buf = GetBuffer(arrayName, m_bodies, m_body_forces);
                if (buf)
                {
                    auto pbuf = buf->pointer();
                    auto alloc = m_bodies->m_x.get_allocator();
                    int owner = m_bodies->m_x.get_owner();
                    aout = svtkHAMRDoubleArray::New(arrayName, pbuf, nb, 1, alloc, owner);
                }
            }

            if (aout)
            {
               tab->AddColumn(aout);
               aout->Delete();
               return 0;
            }
        }
    }
    else if (meshName == "bodies_ug")
    {
        // deep copy into a mesh structure that is compatible with VisIt
        // libsim/ParaView
        auto ug = dynamic_cast<svtkUnstructuredGrid*>(mbds->GetBlock(rank));
        if (ug)
        {
            auto dsa = ug->GetAttributes(association);
            if (dsa)
            {
                long nb = m_bodies->size();

                svtkDataArray *aout = nullptr;

                if (arrayName == "owner")
                {
                    auto ao = svtkIntArray::New();
                    ao->SetName("owner");
                    ao->SetNumberOfTuples(nb);
                    ao->FillValue(rank);
                    aout = ao;
                }
                else
                {
                    auto buf = GetBuffer(arrayName, m_bodies, m_body_forces);
                    if (buf)
                    {
                        auto [spbuf, pbuf] = hamr::get_cpu_accessible(*buf);

                        auto ao = svtkDoubleArray::New();
                        ao->SetName(arrayName);
                        ao->SetNumberOfTuples(nb);
                        auto pao = ao->GetPointer(0);
                        aout = ao;

                        for (long i = 0; i < nb; ++i)
                            pao[i] = pbuf[i];
                    }
                }

                if (aout)
                {
                   dsa->AddArray(aout);
                   aout->Delete();
                   return 0;
                }
            }
        }
    }

    std::cerr << "The mesh \"" << meshName << "\" has no "
        << (association == pCen ? "point" : "cell")
        << " centered array named \"" << arrayName << "\""
        << std::endl;
    return -1;
}
