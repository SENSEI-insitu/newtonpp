#ifndef sensei_adaptor_h
#define sensei_adaptor_h

#include "patch.h"
#include "patch_data.h"
#include "patch_force.h"

#include <DataAdaptor.h>

/// Serves requested data to the in situ analysis back end
class sensei_adaptor : public sensei::DataAdaptor
{
public:
  static sensei_adaptor* New();
  senseiTypeMacro(sensei_adaptor, sensei::DataAdaptor);

  /// SENSEI API
  int GetNumberOfMeshes(unsigned int &numMeshes) override;
  int GetMeshMetadata(unsigned int id, sensei::MeshMetadataPtr &metadata) override;
  int GetMesh(const std::string &meshName, bool structureOnly, svtkDataObject *&mesh) override;
  int AddArray(svtkDataObject* mesh, const std::string &meshName, int association, const std::string &arrayName) override;

  /// pass simulation data here at each time step
  void SetData(long step, double time,
    const std::vector<patch> &patches, const patch_data &pd,
    const patch_force &pf);

protected:
  sensei_adaptor() : m_patches(nullptr), m_bodies(nullptr), m_body_forces(nullptr) {}
  ~sensei_adaptor() {}

  sensei_adaptor(const sensei_adaptor&) = delete;
  void operator=(const sensei_adaptor&) = delete;

  const std::vector<patch> *m_patches;
  const patch_data *m_bodies;
  const patch_force *m_body_forces;
};

#endif
