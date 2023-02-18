#include <math.h>
#include <vector>
#include "nrs.hpp"
#include "platform.hpp"
#include "linAlg.hpp"
#include "nekInterfaceAdapter.hpp"
#include "postProcessing.hpp"

// private members
namespace {
  static dfloat dragIntegral;

  static int nbID;
  static int NfpTotal;
  static dfloat mueLam;
  
  static occa::memory o_bID;
  static occa::memory o_drag;
  static occa::memory o_area;

  static bool setup = 0;
}

dfloat postProcessing::getDrag()
{
  return dragIntegral;
}

void postProcessing::dragCalc(nrs_t *nrs, std::vector<int> bID, bool verbose)
{
  mesh_t *mesh = nrs->meshV;

  if(!setup){
    int bIDmin = *min_element(std::begin(bID), std::end(bID));
    nrsCheck(bIDmin < 1, platform->comm.mpiComm, EXIT_FAILURE,
	     "Boundary ID for drag calc must be >=1","");
    
    nbID = bID.size();
    o_bID = platform->device.malloc(nbID * sizeof(int), bID.data());

    NfpTotal = mesh->Nelements * mesh->Nfaces * mesh->Nfp;
    
    o_drag = platform->device.malloc(mesh->Nelements * nbID * sizeof(dfloat));
    o_area = platform->device.malloc(mesh->Nelements * nbID * sizeof(dfloat));

    platform->options.getArgs("VISCOSITY",mueLam);
  }
 
  platform->linAlg->fillKernel(mesh->Nelements * nbID, 0.0, o_drag);
  platform->linAlg->fillKernel(mesh->Nelements * nbID, 0.0, o_area);

  occa::memory o_SijOij = platform->o_mempool.slice2;
  nrs->SijOijKernel(mesh->Nelements,
		    nrs->fieldOffset,
		    1,
		    mesh->o_vgeo,
		    mesh->o_D,
		    nrs->o_U,
		    o_SijOij);
  oogs::startFinish(o_SijOij, 9, nrs->fieldOffset, ogsDfloat, ogsAdd, nrs->gsh);

  auto dragKernel = platform->kernels.get("drag");
  
  dragKernel(mesh->Nelements,
	     nrs->fieldOffset,
	     NfpTotal,
	     nbID,
	     mueLam,
	     1,
	     mesh->o_sgeo,
	     mesh->o_vmapM,
	     mesh->o_EToB,
	     mesh->o_invLMM,
	     o_SijOij,
	     o_drag,
	     o_area);
  
  const dfloat areasum = platform->linAlg->sum(mesh->Nelements, o_area, platform->comm.mpiComm);
    
  dragIntegral = platform->linAlg->sum(mesh->Nelements, o_drag, platform->comm.mpiComm) / areasum;
      
  setup = 1;
}
