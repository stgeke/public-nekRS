#include <math.h>
#include <vector>
#include "nrs.hpp"
#include "platform.hpp"
#include "linAlg.hpp"
#include "nekInterfaceAdapter.hpp"
#include "postProcessing.hpp"

// private members
namespace {
  static int nbID = 0;
  static dfloat mueLam;

  static dfloat *drag;
    
  static occa::memory o_bID;
  static occa::memory o_drag;
}

occa::memory postProcessing::computeSij(nrs_t *nrs)
{
  mesh_t *mesh = nrs->meshV;
  
  occa::memory o_Sij = platform->o_mempool.slice2;
  nrs->SijOijKernel(mesh->Nelements,
		    nrs->fieldOffset,
		    0,
		    mesh->o_vgeo,
		    mesh->o_D,
		    nrs->o_U,
		    o_Sij);
  oogs::startFinish(o_Sij, 6, nrs->fieldOffset, ogsDfloat, ogsAdd, nrs->gsh);

  return o_Sij;
}

dfloat postProcessing::dragCalc(nrs_t *nrs, std::vector<int> bID)
{
  occa::memory o_Sij = postProcessing::computeSij(nrs);
    
  dfloat dragIntegral = postProcessing::dragCalc(nrs, bID, o_Sij);

  return dragIntegral;
}

dfloat postProcessing::dragCalc(nrs_t *nrs, std::vector<int> bID, occa::memory& o_Sij)
{
  mesh_t *mesh = nrs->meshV;

  if(nbID == 0){ //setup
    drag = (dfloat *)calloc(mesh->Nelements, sizeof(dfloat));
    o_drag = platform->device.malloc(mesh->Nelements * sizeof(dfloat), drag);

    platform->options.getArgs("VISCOSITY",mueLam);
  }
  
  if(nbID != bID.size()){
    
    if(nbID != 0) o_bID.free();

    nbID = bID.size();
    
    o_bID = platform->device.malloc(nbID * sizeof(int), bID.data());
  }
  
  o_bID.copyFrom(bID.data());
  
  platform->linAlg->fillKernel(mesh->Nelements, 0.0, o_drag);

  auto dragKernel = platform->kernels.get("drag");
  
  dragKernel(mesh->Nelements,
	     nrs->fieldOffset,
	     nbID,
	     mueLam,
	     o_bID,
	     mesh->o_sgeo,
	     mesh->o_vmapM,
	     mesh->o_EToB,
	     mesh->o_invLMM,
	     o_Sij,
	     o_drag);

  o_drag.copyTo(drag);

  dfloat dragIntegral = 0;
  for (dlong n = 0; n < mesh->Nelements; n++) {
    dragIntegral += drag[n];
  }

  MPI_Allreduce(MPI_IN_PLACE, &dragIntegral, 1, MPI_DFLOAT, MPI_SUM, platform->comm.mpiComm);
    
  return dragIntegral;
}
