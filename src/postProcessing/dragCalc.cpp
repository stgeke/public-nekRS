#include <math.h>
#include <vector>
#include "nrs.hpp"
#include "platform.hpp"
#include "linAlg.hpp"
#include "nekInterfaceAdapter.hpp"
#include "postProcessing.hpp"

// private members
namespace {
  static dfloat *dragIntegral;

  static int nbID = 0;
  static dfloat mueLam;

  static dfloat *drag;
    
  static occa::memory o_bID;
  static occa::memory o_drag;
}

dfloat* postProcessing::dragCalc(nrs_t *nrs, std::vector<int> bID)
{
  mesh_t *mesh = nrs->meshV;

  if(nbID != bID.size()){
    if(nbID != 0){
      o_bID.free();
      o_drag.free();
    }
    nbID = bID.size();
    
    o_bID = platform->device.malloc(nbID * sizeof(int), bID.data());
    
    drag = (dfloat *)realloc(drag, mesh->Nelements * nbID * sizeof(dfloat));
    o_drag = platform->device.malloc(mesh->Nelements * nbID * sizeof(dfloat), drag);
    
    dragIntegral = (dfloat *)realloc(dragIntegral, nbID * sizeof(dfloat));
  }
  
  o_bID.copyFrom(bID.data());
  
  platform->options.getArgs("VISCOSITY",mueLam);
  
  platform->linAlg->fillKernel(mesh->Nelements * nbID, 0.0, o_drag);
  
  occa::memory o_SijOij = platform->o_mempool.slice2;
  nrs->SijOijKernel(mesh->Nelements,
		    nrs->fieldOffset,
		    0,
		    mesh->o_vgeo,
		    mesh->o_D,
		    nrs->o_U,
		    o_SijOij);
  oogs::startFinish(o_SijOij, 6, nrs->fieldOffset, ogsDfloat, ogsAdd, nrs->gsh);

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
	     o_SijOij,
	     o_drag);

  o_drag.copyTo(drag);
  
  for (dlong ibID = 0; ibID < nbID; ibID++) {
    dragIntegral[ibID] = 0;
  }

  for (dlong n = 0; n < mesh->Nelements; n++) {
    for (dlong ibID = 0; ibID < nbID; ibID++) {
      dragIntegral[ibID] += drag[n + ibID * mesh->Nelements];
    }
  }

  MPI_Allreduce(MPI_IN_PLACE, dragIntegral, nbID, MPI_DFLOAT, MPI_SUM, platform->comm.mpiComm);
    
  return dragIntegral;
}
