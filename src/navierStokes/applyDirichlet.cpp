#include "nrs.hpp"
#include "bcMap.hpp"

void createZeroNormalMask(nrs_t *nrs, mesh_t *mesh, occa::memory &o_EToB, occa::memory& o_EToBV, occa::memory &o_mask)
{
  nrs->initializeZeroNormalMaskKernel(mesh->Nlocal, nrs->fieldOffset, o_EToBV, o_mask);

#if 1
  // normal + count (4 fields)
  auto o_avgNormal = platform->o_mempool.slice0;
  int bcType = ZERO_NORMAL;
  nrs->averageNormalBcTypeKernel(mesh->Nelements,
                                 nrs->fieldOffset,
                                 bcType,
                                 mesh->o_sgeo,
                                 mesh->o_vmapM,
                                 o_EToB,
                                 o_avgNormal);

  oogs::startFinish(o_avgNormal, nrs->NVfields+1, nrs->fieldOffset, ogsDfloat, ogsAdd, mesh->oogs);

  nrs->fixZeroNormalMaskKernel(mesh->Nelements,
                     nrs->fieldOffset,
                     mesh->o_sgeo,
                     mesh->o_vmapM,
                     o_EToB,
                     o_avgNormal,
                     o_mask);
#endif

  oogs::startFinish(o_mask, nrs->NVfields, nrs->fieldOffset, ogsDfloat, ogsMin, mesh->oogs);
}

void applyZeroNormalMask(nrs_t *nrs,
                         mesh_t *mesh,
                         dlong Nelements,
                         occa::memory &o_elementList,
                         occa::memory &o_EToB,
                         occa::memory &o_mask,
                         occa::memory &o_x)
{
  if (Nelements == 0)
    return;

  nrs->applyZeroNormalMaskKernel(Nelements,
                                 nrs->fieldOffset,
                                 o_elementList,
                                 mesh->o_sgeo,
                                 o_mask,
                                 mesh->o_vmapM,
                                 o_EToB,
                                 o_x);
}

void applyZeroNormalMask(nrs_t *nrs, mesh_t *mesh, occa::memory &o_EToB, occa::memory &o_mask, occa::memory &o_x)
{
  nrs->applyZeroNormalMaskKernel(mesh->Nelements,
                                 nrs->fieldOffset,
                                 mesh->o_elementList,
                                 mesh->o_sgeo,
                                 o_mask,
                                 mesh->o_vmapM,
                                 o_EToB,
                                 o_x);
}

void applyDirichletVelocity(nrs_t *nrs, double time)
{
  if (bcMap::unalignedMixedBoundary("velocity")) {
    applyZeroNormalMask(nrs, nrs->meshV, nrs->uvwSolver->o_EToB, nrs->o_zeroNormalMaskVelocity, nrs->o_U);
    applyZeroNormalMask(nrs, nrs->meshV, nrs->uvwSolver->o_EToB, nrs->o_zeroNormalMaskVelocity, nrs->o_Ue);
  }

  mesh_t *mesh = nrs->meshV;

  platform->linAlg->fill((1 + nrs->NVfields) * nrs->fieldOffset,
                         -1.0 * std::numeric_limits<dfloat>::max(),
                         platform->o_mempool.slice6);

  for (int sweep = 0; sweep < 2; sweep++) {
    nrs->pressureDirichletBCKernel(mesh->Nelements,
                                   time,
                                   nrs->fieldOffset,
                                   mesh->o_sgeo,
                                   mesh->o_x,
                                   mesh->o_y,
                                   mesh->o_z,
                                   mesh->o_vmapM,
                                   mesh->o_EToB,
                                   nrs->o_EToB,
                                   nrs->o_rho,
                                   nrs->o_mue,
                                   nrs->o_usrwrk,
                                   nrs->o_Ue,
                                   platform->o_mempool.slice6);

    nrs->velocityDirichletBCKernel(mesh->Nelements,
                                   nrs->fieldOffset,
                                   time,
                                   mesh->o_sgeo,
                                   nrs->o_zeroNormalMaskVelocity,
                                   mesh->o_x,
                                   mesh->o_y,
                                   mesh->o_z,
                                   mesh->o_vmapM,
                                   mesh->o_EToB,
                                   nrs->o_EToB,
                                   nrs->o_rho,
                                   nrs->o_mue,
                                   nrs->neknek->o_pointMap,
                                   nrs->neknek->o_U,
                                   nrs->o_usrwrk,
                                   nrs->o_U,
                                   platform->o_mempool.slice7);

    oogs::startFinish(platform->o_mempool.slice6,
                      1 + nrs->NVfields,
                      nrs->fieldOffset,
                      ogsDfloat,
                      (sweep == 0) ? ogsMax : ogsMin,
                      nrs->gsh);
  }

  if (nrs->pSolver->Nmasked)
    nrs->maskCopyKernel(nrs->pSolver->Nmasked,
                        0,
                        nrs->pSolver->o_maskIds,
                        platform->o_mempool.slice6,
                        nrs->o_P);

  if (nrs->uvwSolver) {
    if (nrs->uvwSolver->Nmasked)
      nrs->maskCopy2Kernel(nrs->uvwSolver->Nmasked,
                          0 * nrs->fieldOffset,
                          nrs->uvwSolver->o_maskIds,
                          platform->o_mempool.slice7,
                          nrs->o_U, nrs->o_Ue);
  }
  else {
    if (nrs->uSolver->Nmasked)
      nrs->maskCopy2Kernel(nrs->uSolver->Nmasked,
                          0 * nrs->fieldOffset,
                          nrs->uSolver->o_maskIds,
                          platform->o_mempool.slice7,
                          nrs->o_U, nrs->o_Ue);
    if (nrs->vSolver->Nmasked)
      nrs->maskCopy2Kernel(nrs->vSolver->Nmasked,
                          1 * nrs->fieldOffset,
                          nrs->vSolver->o_maskIds,
                          platform->o_mempool.slice7,
                          nrs->o_U, nrs->o_Ue);
    if (nrs->wSolver->Nmasked)
      nrs->maskCopy2Kernel(nrs->wSolver->Nmasked,
                          2 * nrs->fieldOffset,
                          nrs->wSolver->o_maskIds,
                          platform->o_mempool.slice7,
                          nrs->o_U, nrs->o_Ue);
  }
}

void applyDirichletScalars(nrs_t *nrs, double time)
{
  cds_t *cds = nrs->cds;
  for (int is = 0; is < cds->NSfields; is++) {
    if (!cds->compute[is])
      continue;
    mesh_t *mesh = cds->mesh[0];
    oogs_t *gsh = cds->gshT;
    if (is) {
      mesh = cds->meshV;
      gsh = cds->gsh;
    }

    auto o_diff_i = cds->o_diff + cds->fieldOffsetScan[is] * sizeof(dfloat);
    auto o_rho_i = cds->o_rho + cds->fieldOffsetScan[is] * sizeof(dfloat);

    platform->linAlg->fill(cds->fieldOffset[is],
                           -1.0 * std::numeric_limits<dfloat>::max(),
                           platform->o_mempool.slice2);
    for (int sweep = 0; sweep < 2; sweep++) {
      cds->dirichletBCKernel(mesh->Nelements,
                             cds->fieldOffset[is],
                             is,
                             time,
                             mesh->o_sgeo,
                             mesh->o_x,
                             mesh->o_y,
                             mesh->o_z,
                             mesh->o_vmapM,
                             mesh->o_EToB,
                             cds->o_EToB[is],
                             cds->o_Ue,
                             o_diff_i,
                             o_rho_i,
                             cds->neknek->o_pointMap,
                             cds->neknek->o_U,
                             cds->neknek->o_S,
                             *(cds->o_usrwrk),
                             platform->o_mempool.slice2);

      if (sweep == 0)
        oogs::startFinish(platform->o_mempool.slice2, 1, cds->fieldOffset[is], ogsDfloat, ogsMax, gsh);
      if (sweep == 1)
        oogs::startFinish(platform->o_mempool.slice2, 1, cds->fieldOffset[is], ogsDfloat, ogsMin, gsh);
    }
    occa::memory o_Si =
        cds->o_S.slice(cds->fieldOffsetScan[is] * sizeof(dfloat), cds->fieldOffset[is] * sizeof(dfloat));
    occa::memory o_Si_e =
        cds->o_Se.slice(cds->fieldOffsetScan[is] * sizeof(dfloat), cds->fieldOffset[is] * sizeof(dfloat));

    if (cds->solver[is]->Nmasked)
      cds->maskCopy2Kernel(cds->solver[is]->Nmasked,
                          0,
                          cds->solver[is]->o_maskIds,
                          platform->o_mempool.slice2,
                          o_Si, o_Si_e);
  }
}

void applyDirichletMesh(nrs_t *nrs, double time)
{
  mesh_t *mesh = nrs->_mesh;
  if (bcMap::unalignedMixedBoundary("mesh")) {
    applyZeroNormalMask(nrs, mesh, nrs->meshSolver->o_EToB, nrs->o_zeroNormalMaskMeshVelocity, mesh->o_U);
    applyZeroNormalMask(nrs, mesh, nrs->meshSolver->o_EToB, nrs->o_zeroNormalMaskMeshVelocity, mesh->o_Ue);
  }
  platform->linAlg->fill(nrs->NVfields * nrs->fieldOffset,
                         -1.0 * std::numeric_limits<dfloat>::max(),
                         platform->o_mempool.slice3);

  for (int sweep = 0; sweep < 2; sweep++) {
    mesh->velocityDirichletKernel(mesh->Nelements,
                                  nrs->fieldOffset,
                                  time,
                                  bcMap::useDerivedMeshBoundaryConditions(),
                                  mesh->o_sgeo,
                                  nrs->o_zeroNormalMaskMeshVelocity,
                                  mesh->o_x,
                                  mesh->o_y,
                                  mesh->o_z,
                                  mesh->o_vmapM,
                                  mesh->o_EToB,
                                  nrs->o_EToBMeshVelocity,
                                  nrs->o_meshRho,
                                  nrs->o_meshMue,
                                  nrs->o_usrwrk,
                                  nrs->o_U,
                                  platform->o_mempool.slice3);

    if (sweep == 0)
      oogs::startFinish(platform->o_mempool.slice3,
                        nrs->NVfields,
                        nrs->fieldOffset,
                        ogsDfloat,
                        ogsMax,
                        nrs->gshMesh);
    if (sweep == 1)
      oogs::startFinish(platform->o_mempool.slice3,
                        nrs->NVfields,
                        nrs->fieldOffset,
                        ogsDfloat,
                        ogsMin,
                        nrs->gshMesh);
  }

  if (nrs->meshSolver->Nmasked)
    nrs->maskCopy2Kernel(nrs->meshSolver->Nmasked,
                        0 * nrs->fieldOffset,
                        nrs->meshSolver->o_maskIds,
                        platform->o_mempool.slice3,
                        mesh->o_U, mesh->o_Ue);
}

void applyDirichlet(nrs_t *nrs, double time)
{
  if (nrs->flow) {
    if (bcMap::unalignedMixedBoundary("velocity")) {
      applyZeroNormalMask(nrs, nrs->meshV, nrs->uvwSolver->o_EToB, nrs->o_zeroNormalMaskVelocity, nrs->o_U);
      applyZeroNormalMask(nrs, nrs->meshV, nrs->uvwSolver->o_EToB, nrs->o_zeroNormalMaskVelocity, nrs->o_Ue);
    }
  }

  if (nrs->Nscalar) applyDirichletScalars(nrs, time); 

  if (nrs->flow) applyDirichletVelocity(nrs, time);  

  if (!platform->options.compareArgs("MESH SOLVER", "NONE"))
    applyDirichletMesh(nrs, time);
}
