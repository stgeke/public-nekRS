#include "nrs.hpp"
#include "platform.hpp"
#include "linAlg.hpp"
#include "postProcessing.hpp"

void postProcessing::strainRate(nrs_t *nrs, bool smooth, occa::memory& o_Sij)
{
  mesh_t *mesh = nrs->meshV;
 
  nrsCheck(o_Sij.size() < 2*nrs->NVfields*nrs->fieldOffset*sizeof(dfloat),
           platform->comm.mpiComm, EXIT_FAILURE, "%s\n", "input buffer for o_Sij < 2*nrs->NVfields*nrs->fieldOffet"); 
 
  nrs->SijOijKernel(mesh->Nelements,
		    nrs->fieldOffset,
		    0,
		    mesh->o_vgeo,
		    mesh->o_D,
		    nrs->o_U,
		    o_Sij);

  if(smooth) { 
    oogs::startFinish(o_Sij, 
                      2*nrs->NVfields,
                      nrs->fieldOffset,
                      ogsDfloat,
                      ogsAdd,
                      nrs->gsh);

    platform->linAlg->axmyMany(mesh->Nlocal, 
                               2*nrs->NVfields, 
                               nrs->fieldOffset, 
                               0, 
                               1.0, 
                               mesh->o_invLMM, 
                               o_Sij);
  }
}

